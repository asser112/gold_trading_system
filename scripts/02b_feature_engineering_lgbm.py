#!/usr/bin/env python3
"""
LightGBM Feature Engineering Module (Separate Pipeline — Approach 1)
- Same classical indicators as XGBoost pipeline
- Adds session binary flags: is_london, is_ny, is_overlap
- Adds H1 trend features: h1_ema20, h1_ema50, h1_trend
- Saves to data/processed/features_lgbm_m5.parquet (separate from XGBoost parquet)
- Saves scaler to models/scalers/robust_scaler_lgbm.pkl
"""
import os
import sys
import pandas as pd
import numpy as np
import ta
from sklearn.preprocessing import RobustScaler
import joblib
import yaml
import logging

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
os.chdir(PROJECT_ROOT)
sys.path.insert(0, SCRIPT_DIR)

from utils import get_db_connection, load_ohlc, compute_atr

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

logging.basicConfig(
    level=getattr(logging, config['logging']['level']),
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Classical technical indicators (identical to XGBoost pipeline)."""
    df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
    df['atr'] = compute_atr(df, period=14)
    df['ema20'] = ta.trend.EMAIndicator(close=df['close'], window=20).ema_indicator()
    df['ema50'] = ta.trend.EMAIndicator(close=df['close'], window=50).ema_indicator()

    typical_price = (df['high'] + df['low'] + df['close']) / 3
    df['vwap'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()

    bb = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_middle'] = bb.bollinger_mavg()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

    adx = ta.trend.ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['adx'] = adx.adx()
    return df


def compute_smart_money(df: pd.DataFrame) -> pd.DataFrame:
    """Smart Money Concepts — vectorized (identical to XGBoost pipeline)."""
    df['range'] = df['high'] - df['low']
    rolling_mean_range = df['range'].rolling(50, min_periods=1).mean()
    rolling_mean_vol = df['volume'].rolling(50, min_periods=1).mean()
    df['large_candle'] = (df['range'] > rolling_mean_range * 1.5) & (df['volume'] > rolling_mean_vol * 1.2)
    df['order_block'] = df['large_candle'].astype(int)
    df.drop(columns=['large_candle', 'range'], inplace=True)

    fvg_bullish = (df['high'] < df['low'].shift(2)) & (df['low'].shift(1) > df['high'].shift(1))
    fvg_bearish = (df['low'] > df['high'].shift(2)) & (df['high'].shift(1) < df['low'].shift(1))
    df['fvg_distance'] = np.where(
        fvg_bullish,
        df['low'].shift(2) - df['high'],
        np.where(fvg_bearish, df['low'] - df['high'].shift(2), 0)
    )

    vol_avg = df['volume'].rolling(50, min_periods=1).mean()
    df['liquidity_zone'] = (df['volume'] > 2 * vol_avg).astype(int)

    window = 10
    rolling_high = df['high'].shift(1).rolling(window).max()
    rolling_low = df['low'].shift(1).rolling(window).min()
    sweep_high = (df['high'] > rolling_high) & (df['close'] < rolling_high)
    sweep_low = (df['low'] < rolling_low) & (df['close'] > rolling_low)
    df['sweep'] = (sweep_high | sweep_low).astype(int)

    return df


def add_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    NEW — LightGBM-specific session binary flags.
    London: 08:00–16:59 UTC
    NY:     13:00–20:59 UTC
    Overlap (London+NY): 13:00–16:59 UTC
    """
    hour = df.index.hour
    df['hour'] = hour
    df['day_of_week'] = df.index.dayofweek
    df['is_london'] = ((hour >= 8) & (hour < 17)).astype(int)
    df['is_ny'] = ((hour >= 13) & (hour < 21)).astype(int)
    df['is_overlap'] = ((hour >= 13) & (hour < 17)).astype(int)
    return df


def add_h1_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    NEW — Resample M5 → H1, compute EMA20 and EMA50, merge back via forward-fill.
    h1_trend: 1 (bullish) / -1 (bearish) / 0 (flat)
    """
    h1 = df['close'].resample('1h').last().dropna()
    h1_ema20 = h1.ewm(span=20, adjust=False).mean()
    h1_ema50 = h1.ewm(span=50, adjust=False).mean()

    h1_df = pd.DataFrame({'h1_ema20': h1_ema20, 'h1_ema50': h1_ema50}, index=h1.index)
    h1_df = h1_df.reindex(df.index, method='ffill')

    df['h1_ema20'] = h1_df['h1_ema20']
    df['h1_ema50'] = h1_df['h1_ema50']
    df['h1_trend'] = np.where(df['h1_ema20'] > df['h1_ema50'], 1,
                     np.where(df['h1_ema20'] < df['h1_ema50'], -1, 0))
    return df


def aggregate_sentiment(df: pd.DataFrame, conn) -> pd.DataFrame:
    """Synthetic news sentiment (identical to XGBoost pipeline)."""
    logger.info("Computing sentiment features...")
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(20).std().clip(1e-6)
    df['atr_norm'] = df['returns'].abs() / df['volatility']

    np.random.seed(42)
    synthetic = np.where(
        df['atr_norm'] > 1.5,
        np.clip(df['returns'] * 50 + np.random.normal(0, 0.1, len(df)), -1, 1),
        0.0
    )
    df['sentiment_score'] = pd.Series(synthetic).rolling(5, min_periods=1).mean().values
    df.drop(columns=['returns', 'volatility', 'atr_norm'], inplace=True, errors='ignore')

    try:
        news_df = pd.read_sql("SELECT * FROM news_processed", conn, parse_dates=['timestamp'])
        if not news_df.empty:
            news_df['candle_time'] = news_df['timestamp'].dt.floor('5min')
            sentiment_agg = news_df.groupby('candle_time')['sentiment_score'].mean()
            for idx, val in sentiment_agg.items():
                if idx in df.index:
                    df.loc[idx, 'sentiment_score'] = val
            logger.info(f"Real news overlaid ({len(news_df)} records)")
    except Exception as e:
        logger.warning(f"Could not load real news: {e} — using synthetic only")

    return df


FEATURE_COLS = [
    'rsi', 'atr', 'ema20', 'ema50', 'vwap',
    'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'adx',
    'order_block', 'fvg_distance', 'liquidity_zone', 'sweep',
    'sentiment_score',
    'hour', 'day_of_week',
    'is_london', 'is_ny', 'is_overlap',
    'h1_ema20', 'h1_ema50', 'h1_trend',
]


def main():
    logger.info("=" * 60)
    logger.info("LGBM FEATURE ENGINEERING — START")
    logger.info("=" * 60)

    conn = get_db_connection()
    df = load_ohlc('m5', conn=conn)
    logger.info(f"Loaded {len(df):,} M5 bars")

    df = aggregate_sentiment(df, conn)
    df = compute_indicators(df)
    df = compute_smart_money(df)
    df = add_session_features(df)
    df = add_h1_trend_features(df)

    df.dropna(inplace=True)
    logger.info(f"Rows after dropping NaN: {len(df):,}")

    # Build target (same logic as XGBoost pipeline)
    df['next_close_change'] = df['close'].shift(-1) - df['close']
    df['target'] = 0
    df.loc[df['next_close_change'] > 0.5 * df['atr'], 'target'] = 1
    df.loc[df['next_close_change'] < -0.5 * df['atr'], 'target'] = -1
    df = df.iloc[:-1]  # drop last row (no target)

    # Scale features
    scaler = RobustScaler()
    scaled = scaler.fit_transform(df[FEATURE_COLS])
    scaled_df = pd.DataFrame(scaled, columns=FEATURE_COLS, index=df.index)
    scaled_df['target'] = df['target'].values

    # Persist
    os.makedirs('models/scalers', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    joblib.dump(scaler, 'models/scalers/robust_scaler_lgbm.pkl')
    scaled_df.to_parquet('data/processed/features_lgbm_m5.parquet')

    logger.info(f"Features saved → data/processed/features_lgbm_m5.parquet")
    logger.info(f"Scaler saved  → models/scalers/robust_scaler_lgbm.pkl")
    logger.info(f"Feature columns ({len(FEATURE_COLS)}): {FEATURE_COLS}")
    logger.info(f"Target distribution: {df['target'].value_counts().to_dict()}")
    logger.info("=" * 60)
    logger.info("LGBM FEATURE ENGINEERING — DONE")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
