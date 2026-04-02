#!/usr/bin/env python3
"""
Feature Engineering Module
- Compute classical indicators (RSI, ATR, EMA, VWAP, Bollinger, ADX)
- Compute Smart Money Concepts (Order Blocks, Fair Value Gaps, Liquidity Zones, Sweeps)
- Compute news sentiment aggregated per candle
- Add time features
- Normalize using RobustScaler
- Save to parquet and scaler
"""
import pandas as pd
import numpy as np
import ta
from sklearn.preprocessing import RobustScaler
import joblib
import yaml
import logging
from utils import get_db_connection, load_ohlc, compute_atr

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

logging.basicConfig(level=getattr(logging, config['logging']['level']))
logger = logging.getLogger(__name__)

def compute_indicators(df):
    """Add classical technical indicators."""
    # RSI
    df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
    # ATR
    df['atr'] = compute_atr(df, period=14)
    # EMA20, EMA50
    df['ema20'] = ta.trend.EMAIndicator(close=df['close'], window=20).ema_indicator()
    df['ema50'] = ta.trend.EMAIndicator(close=df['close'], window=50).ema_indicator()
    # VWAP (simple cumulative)
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    df['vwap'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_middle'] = bb.bollinger_mavg()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    # ADX
    adx = ta.trend.ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['adx'] = adx.adx()
    return df

def compute_smart_money(df):
    """
    Smart Money Concepts (vectorized):
    - Order Blocks: zones where price reversed after a large candle
    - Fair Value Gaps (FVG): three-candle pattern with a gap
    - Liquidity Zones: areas of high volume
    - Sweeps: price breaking a recent high/low and reversing
    """
    # Order Blocks: simplified vectorized version
    df['range'] = df['high'] - df['low']
    rolling_mean_range = df['range'].rolling(50, min_periods=1).mean()
    rolling_mean_vol = df['volume'].rolling(50, min_periods=1).mean()
    df['large_candle'] = (df['range'] > rolling_mean_range * 1.5) & (df['volume'] > rolling_mean_vol * 1.2)
    df['order_block'] = df['large_candle'].astype(int)
    df.drop(columns=['large_candle'], inplace=True)

    # Fair Value Gaps: vectorized
    fvg_bullish = (df['high'] < df['low'].shift(2)) & (df['low'].shift(1) > df['high'].shift(1))
    fvg_bearish = (df['low'] > df['high'].shift(2)) & (df['high'].shift(1) < df['low'].shift(1))
    df['fvg_distance'] = np.where(
        fvg_bullish,
        df['low'].shift(2) - df['high'],
        np.where(fvg_bearish, df['low'] - df['high'].shift(2), 0)
    )

    # Liquidity Zones: vectorized
    vol_avg = df['volume'].rolling(50, min_periods=1).mean()
    df['liquidity_zone'] = (df['volume'] > 2 * vol_avg).astype(int)

    # Sweeps: vectorized
    window = 10
    rolling_high = df['high'].shift(1).rolling(window).max()
    rolling_low = df['low'].shift(1).rolling(window).min()
    sweep_high = (df['high'] > rolling_high) & (df['close'] < rolling_high)
    sweep_low = (df['low'] < rolling_low) & (df['close'] > rolling_low)
    df['sweep'] = (sweep_high | sweep_low).astype(int)

    return df

def add_time_features(df):
    """Add hour, day_of_week, session."""
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    # Session: Asian (0-7), London (8-15), NY (16-23) UTC
    df['session'] = pd.cut(df['hour'], bins=[0,8,16,24], labels=['Asian','London','NY'], right=False)
    return df

def aggregate_news_per_candle(df, conn):
    """
    Aggregate news sentiment per M5 candle.
    Combines real news sentiment with synthetic sentiment based on price action.
    """
    logger.info("Aggregating news sentiment per M5 candle...")
    
    # First, generate synthetic sentiment based on price action (baseline)
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(20).std().clip(1e-6)
    df['atr_norm'] = df['returns'].abs() / df['volatility']
    
    np.random.seed(42)
    synthetic_sentiment = np.where(
        df['atr_norm'] > 1.5,
        np.clip(df['returns'] * 50 + np.random.normal(0, 0.1, len(df)), -1, 1),
        0.0
    )
    synthetic_sentiment = pd.Series(synthetic_sentiment).rolling(5, min_periods=1).mean().values
    df['sentiment_score'] = synthetic_sentiment
    
    df.drop(columns=['returns', 'volatility', 'atr_norm'], inplace=True, errors='ignore')
    
    # Then, overlay real news sentiment if available
    try:
        news_df = pd.read_sql("SELECT * FROM news_processed", conn, parse_dates=['timestamp'])
        
        if not news_df.empty:
            news_df['timestamp'] = pd.to_datetime(news_df['timestamp'])
            news_df['candle_time'] = news_df['timestamp'].dt.floor('5min')
            sentiment_agg = news_df.groupby('candle_time')['sentiment_score'].mean()
            
            # Update sentiment for candles with real news
            for idx, row in sentiment_agg.items():
                if idx in df.index:
                    df.loc[idx, 'sentiment_score'] = row
            
            logger.info(f"Real news sentiment overlaid. Real news count: {len(news_df)}")
        else:
            logger.info("No real news found; using synthetic sentiment only")
            
    except Exception as e:
        logger.warning(f"Error loading real news: {e}. Using synthetic sentiment only.")
    
    return df


def main():
    # Load OHLC M5 data
    conn = get_db_connection()
    df = load_ohlc('m5', conn=conn)
    
    # Aggregate news sentiment per M5 candle
    df = aggregate_news_per_candle(df, conn)

    # Compute indicators
    df = compute_indicators(df)
    df = compute_smart_money(df)
    df = add_time_features(df)

    # Drop rows with NaN (first few rows)
    df.dropna(inplace=True)

    # Prepare feature columns
    feature_cols = ['rsi', 'atr', 'ema20', 'ema50', 'vwap', 'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'adx',
                    'order_block', 'fvg_distance', 'liquidity_zone', 'sweep', 'sentiment_score',
                    'hour', 'day_of_week']
    # Encode session categorical
    df = pd.get_dummies(df, columns=['session'], prefix='session')
    session_cols = [c for c in df.columns if c.startswith('session_')]
    feature_cols.extend(session_cols)

    # Normalize
    scaler = RobustScaler()
    scaled = scaler.fit_transform(df[feature_cols])
    scaled_df = pd.DataFrame(scaled, columns=feature_cols, index=df.index)

    # Save scaler
    joblib.dump(scaler, 'models/scalers/robust_scaler.pkl')

    # Save features to parquet
    scaled_df.to_parquet('data/processed/features_m5.parquet')

    # Compute target: direction of next candle based on 0.5 * ATR
    # Use ATR from the current candle (already computed)
    df['next_close_change'] = df['close'].shift(-1) - df['close']
    # Target: 1 if change > 0.5 * ATR, -1 if change < -0.5 * ATR, else 0
    df['target'] = 0
    df.loc[df['next_close_change'] > 0.5 * df['atr'], 'target'] = 1
    df.loc[df['next_close_change'] < -0.5 * df['atr'], 'target'] = -1
    # Drop last row (no target)
    df_target = df[['target']].iloc[:-1]
    scaled_df = scaled_df.iloc[:-1]  # align
    scaled_df['target'] = df_target['target'].values
    scaled_df.to_parquet('data/processed/features_target_m5.parquet')

    logger.info(f"Feature engineering completed. Features: {feature_cols}")

if __name__ == '__main__':
    main()