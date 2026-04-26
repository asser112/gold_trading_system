#!/usr/bin/env python3
"""
Backtesting Module - XGBoost Strategy with News Sentiment
Uses direct XGBoost predictions for simpler, faster backtesting
"""
import os
import sys
import pandas as pd
import numpy as np
import yaml
import logging
import sqlite3
import joblib
from datetime import timedelta
from backtesting import Backtest, Strategy
import glob

np.random.seed(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
os.chdir(PROJECT_ROOT)

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

logging.basicConfig(level=getattr(logging, config['logging']['level']))
logger = logging.getLogger(__name__)


class XGBoostStrategy(Strategy):
    def init(self):
        self.xgb_model = joblib.load('models/xgboost/xgboost_best.pkl')
        # Use the feature list the model was actually trained on
        if hasattr(self.xgb_model, 'feature_names_in_'):
            self.feature_cols = list(self.xgb_model.feature_names_in_)
        else:
            self.feature_cols = ['rsi', 'ema20', 'ema50', 'vwap', 'bb_upper', 'bb_middle', 'bb_lower',
                                'bb_width', 'adx', 'order_block', 'fvg_distance', 'liquidity_zone',
                                'sweep', 'sentiment_score', 'hour', 'day_of_week',
                                'session_Asian', 'session_London', 'session_NY']
        self.oz_size = max(1, int(config['trading']['lot_size'] * 100))
        self.threshold = config['models']['ensemble']['confidence_threshold']
        self.min_hold_bars = config['models']['ensemble']['min_hold_bars']
        self.max_bars = self.min_hold_bars + 60
        
    def next(self):
        row = self.data
        
        X = np.array([[float(row[col]) for col in self.feature_cols]], dtype=np.float32)
        
        probs = self.xgb_model.predict_proba(X)[0]
        predicted_class = np.argmax(probs)
        confidence = probs[predicted_class]
        
        if self.position:
            self.bar_count = getattr(self, 'bar_count', 0) + 1
            if self.bar_count >= self.max_bars:
                self.position.close()
        else:
            self.bar_count = 0
            if predicted_class == 2 and confidence >= self.threshold:
                self.buy(size=self.oz_size)
            elif predicted_class == 0 and confidence >= self.threshold:
                self.sell(size=self.oz_size)


def main():
    logger.info("=" * 60)
    logger.info("XGBOOST BACKTESTER — STARTUP")
    logger.info("=" * 60)

    db_path = 'data/gold_trading.db'
    features_path = 'data/processed/features_target_m5.parquet'
    model_path = 'models/xgboost/xgboost_best.pkl'

    # ── File checks ──────────────────────────────────────────────
    for label, path in [("Database", db_path), ("Features", features_path), ("Model", model_path)]:
        exists = os.path.exists(path)
        size_mb = os.path.getsize(path) / 1_048_576 if exists else 0
        logger.info(f"  {label:10s}: {'OK' if exists else 'MISSING':7s}  {path}  ({size_mb:.2f} MB)")
        if not exists:
            logger.error(f"{label} not found — aborting.")
            return

    # ── Load and inspect model ────────────────────────────────────
    _model = joblib.load(model_path)
    model_type = type(_model).__name__
    n_features = len(_model.feature_names_in_) if hasattr(_model, 'feature_names_in_') else '?'
    n_estimators = getattr(_model, 'n_estimators', '?')
    logger.info(f"\nModel loaded:")
    logger.info(f"  Type        : {model_type}")
    logger.info(f"  Estimators  : {n_estimators}")
    logger.info(f"  Features    : {n_features}")
    if hasattr(_model, 'feature_names_in_'):
        logger.info(f"  Feature list: {list(_model.feature_names_in_)}")

    if hasattr(_model, 'feature_names_in_'):
        feature_cols = list(_model.feature_names_in_)
    else:
        feature_cols = ['rsi', 'ema20', 'ema50', 'vwap', 'bb_upper', 'bb_middle', 'bb_lower',
                       'bb_width', 'adx', 'order_block', 'fvg_distance', 'liquidity_zone',
                       'sweep', 'sentiment_score', 'hour', 'day_of_week',
                       'session_Asian', 'session_London', 'session_NY']

    # ── Config summary ────────────────────────────────────────────
    threshold   = config['models']['ensemble']['confidence_threshold']
    min_hold    = config['models']['ensemble']['min_hold_bars']
    commission  = config['backtest']['commission']
    spread      = config['backtest']['spread']
    init_bal    = config['backtest']['initial_balance']
    lot_size    = config['trading']['lot_size']
    logger.info(f"\nBacktest config:")
    logger.info(f"  Initial balance      : ${init_bal:,.2f}")
    logger.info(f"  Lot size             : {lot_size}")
    logger.info(f"  Confidence threshold : {threshold}")
    logger.info(f"  Min hold bars        : {min_hold}")
    logger.info(f"  Commission           : {commission}")
    logger.info(f"  Spread               : {spread}")

    # ── Load OHLC data ────────────────────────────────────────────
    conn = sqlite3.connect(db_path)
    end_date   = pd.Timestamp.now()
    start_date = end_date - timedelta(days=365)
    logger.info(f"\nData period: {start_date.date()} → {end_date.date()} (last 12 months)")

    ohlc_df = pd.read_sql(
        f"SELECT * FROM ohlc_m5 WHERE timestamp >= '{start_date}' ORDER BY timestamp",
        conn,
        index_col='timestamp',
        parse_dates=['timestamp']
    )
    conn.close()

    ohlc_df = ohlc_df[~ohlc_df.index.duplicated(keep='first')]
    ohlc_df = ohlc_df.dropna()
    ohlc_df.columns = [c.capitalize() for c in ohlc_df.columns]
    logger.info(f"OHLC bars loaded     : {len(ohlc_df):,}  ({len(ohlc_df)/288:.0f} trading days approx)")

    # ── Load features ─────────────────────────────────────────────
    precomputed = pd.read_parquet(features_path)
    precomputed = precomputed.reset_index()
    precomputed['timestamp'] = pd.to_datetime(precomputed['timestamp'])
    precomputed = precomputed[~precomputed['timestamp'].duplicated(keep='first')]
    precomputed = precomputed.set_index('timestamp')
    precomputed = precomputed[precomputed.index >= start_date]
    precomputed = precomputed[precomputed.index <= end_date]
    logger.info(f"Feature rows loaded  : {len(precomputed):,}")

    common_idx = ohlc_df.index.intersection(precomputed.index)
    logger.info(f"Matched bars         : {len(common_idx):,}")

    if len(common_idx) == 0:
        logger.error("No matching data between OHLC and features — aborting.")
        return

    df = ohlc_df.loc[common_idx].copy()
    for col in precomputed.columns:
        if col not in ['target']:
            df[col] = precomputed.loc[common_idx, col].values
    
    ohlc_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df[ohlc_cols + feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    logger.info(f"Running backtest on {len(df)} bars...")
    
    bt = Backtest(
        df,
        XGBoostStrategy,
        cash=config['backtest']['initial_balance'],
        commission=config['backtest']['commission'],
        spread=config['backtest']['spread'],
        exclusive_orders=True,
        finalize_trades=True
    )
    
    stats = bt.run()
    
    trades = stats['_trades']
    
    if len(trades) > 0:
        total_trades = len(trades)
        
        weeks = (end_date - start_date).days / 7
        avg_trades_per_week = total_trades / weeks if weeks > 0 else 0
        
        returns = trades['ReturnPct'].values / 100
        winning_trades = sum(1 for r in returns if r > 0)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        avg_profit_pct = np.mean(returns) * 100 if len(returns) > 0 else 0
        avg_profit_pips = avg_profit_pct * 10 if avg_profit_pct != 0 else 0
        
        sharpe_ratio = stats.get('Sharpe Ratio', 0)
        max_dd = stats.get('Max. Drawdown', 0)
        
        total_pnl = trades['PnL'].sum()
        final_equity = config['backtest']['initial_balance'] + total_pnl
        total_net_profit = total_pnl
        
        logger.info("\n" + "="*60)
        logger.info("XGBOOST BACKTEST RESULTS (Last 12 Months)")
        logger.info("="*60)
        logger.info(f"Period: {start_date.date()} to {end_date.date()}")
        logger.info(f"Initial Balance: ${config['backtest']['initial_balance']:,.2f}")
        logger.info("-"*60)
        logger.info(f"Total Trades: {total_trades}")
        logger.info(f"Average Trades per Week: {avg_trades_per_week:.2f}")
        logger.info(f"Win Rate: {win_rate:.2f}%")
        logger.info(f"Average Profit per Trade: ${total_pnl / total_trades:.2f}")
        logger.info(f"Sharpe Ratio: {sharpe_ratio:.4f}")
        logger.info(f"Max Drawdown: {max_dd:.2f}%")
        logger.info(f"Final Equity: ${final_equity:,.2f}")
        logger.info(f"Total Net Profit: ${total_net_profit:,.2f}")
        logger.info("="*60)
        
        os.makedirs('backtest_reports', exist_ok=True)
        
        with open('backtest_reports/last_year_results.txt', 'w') as f:
            f.write("XGBOOST BACKTEST RESULTS (Last 12 Months)\n")
            f.write("="*60 + "\n")
            f.write(f"Period: {start_date.date()} to {end_date.date()}\n")
            f.write(f"Initial Balance: ${config['backtest']['initial_balance']:,.2f}\n")
            f.write("-"*60 + "\n")
            f.write(f"Total Trades: {total_trades}\n")
            f.write(f"Average Trades per Week: {avg_trades_per_week:.2f}\n")
            f.write(f"Win Rate: {win_rate:.2f}%\n")
            f.write(f"Average Profit per Trade: ${total_pnl / total_trades:.2f}\n")
            f.write(f"Sharpe Ratio: {sharpe_ratio:.4f}\n")
            f.write(f"Max Drawdown: {max_dd:.2f}%\n")
            f.write(f"Final Equity: ${final_equity:,.2f}\n")
            f.write(f"Total Net Profit: ${total_net_profit:,.2f}\n")
            f.write("="*60 + "\n")
        
        logger.info("Results saved to backtest_reports/last_year_results.txt")
    else:
        logger.warning("No trades generated — check confidence threshold or data range.")
        os.makedirs('backtest_reports', exist_ok=True)
        with open('backtest_reports/last_year_results.txt', 'w') as f:
            f.write("No trades generated during backtest period.\n")

    logger.info("\n" + "=" * 60)
    logger.info("BACKTEST COMPLETE")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
