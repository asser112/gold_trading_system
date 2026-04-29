#!/usr/bin/env python3
"""
LightGBM Backtesting Module (Separate Pipeline — Approach 1)
- Loads data/processed/features_lgbm_m5.parquet
- Applies London + NY session filter before backtesting
- Uses backtesting.py framework (identical mechanics to 08_backtester.py)
- Saves report to backtest_reports/lgbm_last_year_results.txt
"""
import os
import sys
import sqlite3
import pandas as pd
import numpy as np
import yaml
import logging
import joblib
from datetime import timedelta
from backtesting import Backtest, Strategy

np.random.seed(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
os.chdir(PROJECT_ROOT)

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

logging.basicConfig(
    level=getattr(logging, config['logging']['level']),
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

LGBM_CFG    = config.get('lightgbm', {})
SESSION_CFG = LGBM_CFG.get('session_filter', {})
LONDON_START = SESSION_CFG.get('london_start_utc', 8)
LONDON_END   = SESSION_CFG.get('london_end_utc', 17)
NY_START     = SESSION_CFG.get('ny_start_utc', 13)
NY_END       = SESSION_CFG.get('ny_end_utc', 21)


class LightGBMStrategy(Strategy):
    def init(self):
        self.lgbm_model = joblib.load('models/lightgbm/lgbm_best.pkl')
        if hasattr(self.lgbm_model, '_feature_cols'):
            self.feature_cols = self.lgbm_model._feature_cols
        elif hasattr(self.lgbm_model, 'feature_name_'):
            self.feature_cols = list(self.lgbm_model.feature_name_)
        else:
            raise RuntimeError(
                "Cannot determine feature columns from model. "
                "Re-train with scripts/10_train_lightgbm.py."
            )
        self.oz_size  = max(1, int(config['trading']['lot_size'] * 100))
        self.threshold = LGBM_CFG.get('confidence_threshold', 0.65)
        self.min_hold_bars = config['models']['ensemble']['min_hold_bars']
        self.max_bars = self.min_hold_bars + 60
        self.bar_count = 0

    def next(self):
        row = self.data
        hour = row.index[-1].hour
        in_session = (
            (LONDON_START <= hour < LONDON_END) or
            (NY_START <= hour < NY_END)
        )
        if not in_session:
            return

        X = np.array([[float(row[col]) for col in self.feature_cols]], dtype=np.float32)
        probs = self.lgbm_model.predict_proba(X)[0]
        predicted_class = int(np.argmax(probs))
        confidence = float(probs[predicted_class])

        if self.position:
            self.bar_count += 1
            if self.bar_count >= self.max_bars:
                self.position.close()
                self.bar_count = 0
        else:
            self.bar_count = 0
            if predicted_class == 2 and confidence >= self.threshold:   # buy
                self.buy(size=self.oz_size)
            elif predicted_class == 0 and confidence >= self.threshold: # sell
                self.sell(size=self.oz_size)


def main():
    logger.info("=" * 60)
    logger.info("LIGHTGBM BACKTESTER — STARTUP")
    logger.info("=" * 60)

    db_path       = 'data/gold_trading.db'
    features_path = 'data/processed/features_lgbm_m5.parquet'
    model_path    = 'models/lightgbm/lgbm_best.pkl'

    for label, path in [("Database", db_path), ("Features", features_path), ("Model", model_path)]:
        exists  = os.path.exists(path)
        size_mb = os.path.getsize(path) / 1_048_576 if exists else 0
        logger.info(f"  {label:10s}: {'OK' if exists else 'MISSING':7s}  {path}  ({size_mb:.2f} MB)")
        if not exists:
            logger.error(f"{label} not found — aborting.")
            sys.exit(1)

    _model = joblib.load(model_path)
    logger.info(f"\nModel: {type(_model).__name__}")
    if hasattr(_model, '_feature_cols'):
        feature_cols = _model._feature_cols
        logger.info(f"Features ({len(feature_cols)}): {feature_cols}")
    elif hasattr(_model, 'feature_name_'):
        feature_cols = list(_model.feature_name_)
    else:
        logger.error("Cannot determine feature list from model.")
        sys.exit(1)

    threshold  = LGBM_CFG.get('confidence_threshold', 0.65)
    min_hold   = config['models']['ensemble']['min_hold_bars']
    commission = config['backtest']['commission']
    spread     = config['backtest']['spread']
    init_bal   = config['backtest']['initial_balance']
    lot_size   = config['trading']['lot_size']

    logger.info(f"\nBacktest config:")
    logger.info(f"  Initial balance      : ${init_bal:,.2f}")
    logger.info(f"  Lot size             : {lot_size}")
    logger.info(f"  Confidence threshold : {threshold}")
    logger.info(f"  Min hold bars        : {min_hold}")
    logger.info(f"  Commission           : {commission}")
    logger.info(f"  Spread               : {spread}")
    logger.info(f"  Session filter       : London {LONDON_START}–{LONDON_END} UTC + NY {NY_START}–{NY_END} UTC")

    conn = sqlite3.connect(db_path)
    end_date   = pd.Timestamp.now()
    start_date = end_date - timedelta(days=365)
    logger.info(f"\nData period: {start_date.date()} → {end_date.date()}")

    ohlc_df = pd.read_sql(
        f"SELECT * FROM ohlc_m5 WHERE timestamp >= '{start_date}' ORDER BY timestamp",
        conn,
        index_col='timestamp',
        parse_dates=['timestamp']
    )
    conn.close()
    ohlc_df = ohlc_df[~ohlc_df.index.duplicated(keep='first')].dropna()
    ohlc_df.columns = [c.capitalize() for c in ohlc_df.columns]
    logger.info(f"OHLC bars loaded     : {len(ohlc_df):,}")

    precomputed = pd.read_parquet(features_path)
    precomputed = precomputed.reset_index()
    precomputed['timestamp'] = pd.to_datetime(precomputed['timestamp'])
    precomputed = precomputed[~precomputed['timestamp'].duplicated(keep='first')].set_index('timestamp')
    precomputed = precomputed[(precomputed.index >= start_date) & (precomputed.index <= end_date)]
    logger.info(f"Feature rows loaded  : {len(precomputed):,}")

    common_idx = ohlc_df.index.intersection(precomputed.index)
    logger.info(f"Matched bars         : {len(common_idx):,}")

    if len(common_idx) == 0:
        logger.error("No matching bars between OHLC and features — aborting.")
        sys.exit(1)

    df = ohlc_df.loc[common_idx].copy()
    for col in precomputed.columns:
        if col != 'target':
            df[col] = precomputed.loc[common_idx, col].values

    ohlc_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df[ohlc_cols + feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    logger.info(f"Running backtest on {len(df):,} bars (session guard active inside strategy)...")

    bt = Backtest(
        df,
        LightGBMStrategy,
        cash=init_bal,
        commission=commission,
        spread=spread,
        exclusive_orders=True,
        finalize_trades=True
    )
    stats = bt.run()
    trades = stats['_trades']

    os.makedirs('backtest_reports', exist_ok=True)
    report_path = 'backtest_reports/lgbm_last_year_results.txt'

    if len(trades) > 0:
        total_trades      = len(trades)
        weeks             = (end_date - start_date).days / 7
        avg_weekly_trades = total_trades / weeks if weeks > 0 else 0
        returns           = trades['ReturnPct'].values / 100
        winning_trades    = sum(1 for r in returns if r > 0)
        win_rate          = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        sharpe_ratio      = stats.get('Sharpe Ratio', 0)
        max_dd            = stats.get('Max. Drawdown', 0)
        total_pnl         = trades['PnL'].sum()
        final_equity      = init_bal + total_pnl

        lines = [
            "LIGHTGBM BACKTEST RESULTS (Last 12 Months)",
            "=" * 60,
            f"Period: {start_date.date()} to {end_date.date()}",
            f"Session: London {LONDON_START}–{LONDON_END} UTC + NY {NY_START}–{NY_END} UTC",
            f"Initial Balance: ${init_bal:,.2f}",
            "-" * 60,
            f"Total Trades: {total_trades}",
            f"Average Trades per Week: {avg_weekly_trades:.2f}",
            f"Win Rate: {win_rate:.2f}%",
            f"Average Profit per Trade: ${total_pnl / total_trades:.2f}",
            f"Sharpe Ratio: {sharpe_ratio:.4f}",
            f"Max Drawdown: {max_dd:.2f}%",
            f"Final Equity: ${final_equity:,.2f}",
            f"Total Net Profit: ${total_pnl:,.2f}",
            "=" * 60,
        ]
        for line in lines:
            logger.info(line)
        with open(report_path, 'w') as f:
            f.write('\n'.join(lines) + '\n')
    else:
        logger.warning("No trades generated — try lowering lightgbm.confidence_threshold in config.yaml")
        with open(report_path, 'w') as f:
            f.write("No trades generated during backtest period.\n")

    logger.info(f"Report saved → {report_path}")
    logger.info("=" * 60)
    logger.info("LIGHTGBM BACKTEST — DONE")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
