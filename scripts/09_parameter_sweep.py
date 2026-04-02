#!/usr/bin/env python3
"""
Parameter sweep for Gold Trading System
Tests different confidence thresholds and min hold bars
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
import itertools

np.random.seed(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
os.chdir(PROJECT_ROOT)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FEATURE_COLS = ['rsi', 'atr', 'ema20', 'ema50', 'vwap', 'bb_upper', 'bb_middle', 'bb_lower',
                'bb_width', 'adx', 'order_block', 'fvg_distance', 'liquidity_zone',
                'sweep', 'sentiment_score', 'hour', 'day_of_week',
                'session_Asian', 'session_London', 'session_NY']

def load_data():
    db_path = 'data/gold_trading.db'
    features_path = 'data/processed/features_target_m5.parquet'
    
    conn = sqlite3.connect(db_path)
    
    end_date = pd.Timestamp.now()
    start_date = end_date - timedelta(days=365)
    
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
    
    precomputed = pd.read_parquet(features_path)
    precomputed = precomputed.reset_index()
    precomputed['timestamp'] = pd.to_datetime(precomputed['timestamp'])
    precomputed = precomputed[~precomputed['timestamp'].duplicated(keep='first')]
    precomputed = precomputed.set_index('timestamp')
    precomputed = precomputed[precomputed.index >= start_date]
    precomputed = precomputed[precomputed.index <= end_date]
    
    common_idx = ohlc_df.index.intersection(precomputed.index)
    
    df = ohlc_df.loc[common_idx].copy()
    
    for col in precomputed.columns:
        if col not in ['target']:
            df[col] = precomputed.loc[common_idx, col].values
    
    ohlc_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df[ohlc_cols + FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return df, start_date, end_date


def run_backtest(df, threshold, min_hold_bars, initial_balance, spread, lot_size):
    class SweepStrategy(Strategy):
        def init(self):
            self.xgb_model = joblib.load('models/xgboost/xgboost_best.pkl')
            self.threshold = threshold
            self.min_hold_bars = min_hold_bars
            self.max_bars = min_hold_bars + 60
            self.oz_size = max(1, int(lot_size * 100))
            
        def next(self):
            row = self.data
            
            X = np.array([[float(row[col]) for col in FEATURE_COLS]], dtype=np.float32)
            
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
    
    bt = Backtest(
        df,
        SweepStrategy,
        cash=initial_balance,
        commission=0,
        spread=spread,
        exclusive_orders=True,
        finalize_trades=True
    )
    
    stats = bt.run()
    return stats


def main():
    logger.info("Loading data...")
    df, start_date, end_date = load_data()
    logger.info(f"Data loaded: {len(df)} bars")
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    initial_balance = 10000
    spread = 0.0030
    lot_size = 0.01
    
    thresholds = [0.95, 0.97, 0.98, 0.99, 0.995]
    min_hold_bars_list = [180, 240, 300, 360]
    
    results = []
    
    total_runs = len(thresholds) * len(min_hold_bars_list)
    run_num = 0
    
    for threshold, min_hold in itertools.product(thresholds, min_hold_bars_list):
        run_num += 1
        logger.info(f"\n[{run_num}/{total_runs}] Testing threshold={threshold}, min_hold={min_hold}")
        
        try:
            stats = run_backtest(df, threshold, min_hold, initial_balance, spread, lot_size)
            
            trades = stats['_trades']
            
            if len(trades) > 0:
                total_trades = len(trades)
                returns = trades['ReturnPct'].values / 100
                winning_trades = sum(1 for r in returns if r > 0)
                win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
                
                avg_profit_pct = np.mean(returns) * 100 if len(returns) > 0 else 0
                sharpe_ratio = stats.get('Sharpe Ratio', 0)
                max_dd = stats.get('Max. Drawdown', 0)
                
                final_equity = stats.get('End Value', initial_balance)
                total_net_profit = final_equity - initial_balance
                
                logger.info(f"  Trades: {total_trades}, Win Rate: {win_rate:.1f}%, "
                          f"Net Profit: ${total_net_profit:.2f}, Sharpe: {sharpe_ratio:.2f}, "
                          f"Max DD: {max_dd:.1f}%")
                
                results.append({
                    'threshold': threshold,
                    'min_hold_bars': min_hold,
                    'total_trades': total_trades,
                    'win_rate': win_rate,
                    'avg_profit_pct': avg_profit_pct,
                    'net_profit': total_net_profit,
                    'sharpe': sharpe_ratio,
                    'max_dd': max_dd,
                    'final_equity': final_equity
                })
            else:
                logger.info(f"  No trades generated")
                results.append({
                    'threshold': threshold,
                    'min_hold_bars': min_hold,
                    'total_trades': 0,
                    'win_rate': 0,
                    'avg_profit_pct': 0,
                    'net_profit': 0,
                    'sharpe': 0,
                    'max_dd': 0,
                    'final_equity': initial_balance
                })
        except Exception as e:
            logger.error(f"  Error: {e}")
            results.append({
                'threshold': threshold,
                'min_hold_bars': min_hold,
                'total_trades': 0,
                'win_rate': 0,
                'avg_profit_pct': 0,
                'net_profit': -999999,
                'sharpe': 0,
                'max_dd': 0,
                'final_equity': initial_balance
            })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('net_profit', ascending=False)
    
    logger.info("\n" + "="*80)
    logger.info("PARAMETER SWEEP RESULTS (Sorted by Net Profit)")
    logger.info("="*80)
    
    for _, row in results_df.iterrows():
        logger.info(f"Threshold={row['threshold']:.3f}, MinHold={int(row['min_hold_bars'])}, "
                   f"Trades={int(row['total_trades'])}, WinRate={row['win_rate']:.1f}%, "
                   f"Profit=${row['net_profit']:.2f}, Sharpe={row['sharpe']:.2f}, "
                   f"MaxDD={row['max_dd']:.1f}%")
    
    best = results_df.iloc[0]
    logger.info("\n" + "="*80)
    logger.info("BEST CONFIGURATION")
    logger.info("="*80)
    logger.info(f"Confidence Threshold: {best['threshold']}")
    logger.info(f"Min Hold Bars: {int(best['min_hold_bars'])}")
    logger.info(f"Total Trades: {int(best['total_trades'])}")
    logger.info(f"Win Rate: {best['win_rate']:.2f}%")
    logger.info(f"Net Profit: ${best['net_profit']:.2f}")
    logger.info(f"Sharpe Ratio: {best['sharpe']:.4f}")
    logger.info(f"Max Drawdown: {best['max_dd']:.2f}%")
    logger.info(f"Final Equity: ${best['final_equity']:.2f}")
    
    os.makedirs('backtest_reports', exist_ok=True)
    
    with open('backtest_reports/parameter_sweep_results.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("GOLD TRADING SYSTEM - PARAMETER SWEEP RESULTS\n")
        f.write("="*80 + "\n")
        f.write(f"Period: {start_date.date()} to {end_date.date()}\n")
        f.write(f"Initial Balance: ${initial_balance}\n")
        f.write(f"Spread: 30 pips (0.0030)\n")
        f.write(f"Lot Size: {lot_size} (1 oz)\n")
        f.write("="*80 + "\n\n")
        
        f.write("PARAMETER COMBINATIONS (Sorted by Net Profit):\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Threshold':>10} {'MinHold':>8} {'Trades':>7} {'WinRate':>8} {'NetProfit':>12} {'Sharpe':>8} {'MaxDD':>8}\n")
        f.write("-"*80 + "\n")
        
        for _, row in results_df.iterrows():
            f.write(f"{row['threshold']:>10.3f} {int(row['min_hold_bars']):>8} {int(row['total_trades']):>7} "
                   f"{row['win_rate']:>7.1f}% ${row['net_profit']:>10.2f} {row['sharpe']:>8.2f} {row['max_dd']:>7.1f}%\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("BEST CONFIGURATION\n")
        f.write("="*80 + "\n")
        f.write(f"Confidence Threshold: {best['threshold']}\n")
        f.write(f"Min Hold Bars: {int(best['min_hold_bars'])}\n")
        f.write(f"Total Trades: {int(best['total_trades'])}\n")
        f.write(f"Win Rate: {best['win_rate']:.2f}%\n")
        f.write(f"Net Profit: ${best['net_profit']:.2f}\n")
        f.write(f"Sharpe Ratio: {best['sharpe']:.4f}\n")
        f.write(f"Max Drawdown: {best['max_dd']:.2f}%\n")
        f.write(f"Final Equity: ${best['final_equity']:.2f}\n")
        
        if best['net_profit'] > 0:
            monthly_profit = best['net_profit'] / 12
            monthly_return = (monthly_profit / initial_balance) * 100
            annual_return = (best['net_profit'] / initial_balance) * 100
            
            f.write("\n" + "="*80 + "\n")
            f.write("PROFITABILITY ANALYSIS\n")
            f.write("="*80 + "\n")
            f.write(f"Monthly Profit (0.01 lot): ${monthly_profit:.2f}\n")
            f.write(f"Monthly Return (0.01 lot): {monthly_return:.2f}%\n")
            f.write(f"Annual Return (0.01 lot): {annual_return:.2f}%\n")
            
            target_monthly = initial_balance * 0.125
            required_profit = target_monthly * 12
            if best['net_profit'] > 0:
                scale_factor = required_profit / best['net_profit']
                f.write(f"\nTo achieve 12.5% monthly return (${target_monthly:.2f}):\n")
                f.write(f"Required annual profit: ${required_profit:.2f}\n")
                f.write(f"Scale factor: {scale_factor:.1f}x\n")
                f.write(f"Recommended lot size: {lot_size * scale_factor:.2f} ({int(lot_size * scale_factor * 100)} oz)\n")
        else:
            f.write("\n" + "="*80 + "\n")
            f.write("PROFITABILITY ANALYSIS\n")
            f.write("="*80 + "\n")
            f.write("Strategy is NOT profitable with 30-pip spread.\n")
            f.write("Consider: Lower spread broker or RL retraining.\n")
    
    logger.info("\nResults saved to backtest_reports/parameter_sweep_results.txt")


if __name__ == '__main__':
    main()
