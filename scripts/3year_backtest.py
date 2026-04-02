#!/usr/bin/env python3
"""
3-Year Backtest for Exness Standard Account
With year-by-year breakdown
"""
import os
import sys
import pandas as pd
import numpy as np
import joblib
import yaml
import sqlite3

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

INITIAL_BALANCE = config['backtest']['initial_balance']
COMMISSION = config['backtest']['commission']
SPREAD_PIPS = config['backtest']['spread']
SLIPPAGE_PIPS = config['backtest']['slippage']

CONFIDENCE_THRESHOLD = config['models']['ensemble']['confidence_threshold']
RISK_PER_TRADE = config['models']['ensemble'].get('risk_per_trade', None)
MIN_HOLD_BARS = config['models'].get('ensemble', {}).get('min_hold_bars', 180)
FIXED_LOT_SIZE = config['trading'].get('lot_size', 0.022)

PIP_SIZE = 0.01
LOT_SIZE_OZ = 100
PIP_VALUE_PER_LOT = 1.0  # $1 per pip for XAUUSD


def run_backtest(period_start, period_end, period_name):
    """Run backtest for a specific period."""
    print(f"\n{'='*60}")
    print(f"Running backtest: {period_name}")
    print(f"Period: {period_start} to {period_end}")
    print(f"{'='*60}")
    
    df = pd.read_parquet('data/processed/features_target_m5.parquet')
    df = df.dropna()
    df = df.loc[period_start:period_end]
    
    if len(df) == 0:
        print(f"No data for period {period_name}")
        return None
    
    print(f"Bars: {len(df)}")
    
    xgb_model = joblib.load('models/xgboost/xgboost_best.pkl')
    feature_cols = [c for c in df.columns if c != 'target']
    
    conn = sqlite3.connect('data/gold_trading.db')
    close_df = pd.read_sql(
        f"SELECT timestamp, close FROM ohlc_m5 WHERE timestamp >= '{period_start}' AND timestamp <= '{period_end}' ORDER BY timestamp",
        conn,
        index_col='timestamp',
        parse_dates=['timestamp']
    )
    conn.close()
    close_df = close_df[~close_df.index.duplicated(keep='first')]
    df_aligned = df.join(close_df[['close']], how='inner')
    close_prices = df_aligned['close'].values
    
    if len(df_aligned) == 0:
        print("No aligned data")
        return None
    
    X = df_aligned[feature_cols].values
    probs = xgb_model.predict_proba(X)
    preds = xgb_model.predict(X)
    
    raw_signal = np.where(preds == 0, -1, np.where(preds == 2, 1, 0))
    confidence = np.max(probs, axis=1)
    final_signals = np.where(confidence >= CONFIDENCE_THRESHOLD, raw_signal, 0)
    
    equity = INITIAL_BALANCE
    equity_curve = [equity]
    
    trades = []
    wins = 0
    losses = 0
    pnl_list = []
    
    TOTAL_COST_PIPS = SPREAD_PIPS + SLIPPAGE_PIPS + (COMMISSION / PIP_SIZE)
    
    position = 0
    entry_price = 0
    entry_idx = 0
    position_lot_size = 0
    
    for i in range(len(df_aligned)):
        signal = final_signals[i]
        price = close_prices[i]
        
        if RISK_PER_TRADE is not None:
            risk_amount = equity * (RISK_PER_TRADE / 100)
            lot_size = risk_amount / 50  # Assume 50 pip SL
            lot_size = max(0.01, min(lot_size, 1.0))
        else:
            lot_size = FIXED_LOT_SIZE
        
        if position == 0 and signal != 0:
            position = signal
            entry_price = price
            entry_idx = i
            position_lot_size = lot_size
        elif position != 0 and signal != 0 and signal != position:
            if (i - entry_idx) >= MIN_HOLD_BARS:
                direction = 1 if position == 1 else -1
                price_diff = (price - entry_price) * direction
                pnl_pips = price_diff * 100 - TOTAL_COST_PIPS
                pnl = pnl_pips * PIP_VALUE_PER_LOT * position_lot_size
                
                trades.append({'pnl': pnl, 'pips': pnl_pips})
                equity += pnl
                pnl_list.append(pnl)
                
                if pnl > 0:
                    wins += 1
                else:
                    losses += 1
                
                position = signal
                entry_price = price
                entry_idx = i
                position_lot_size = lot_size
        else:
            equity_curve.append(equity)
    
    if len(equity_curve) < len(df_aligned) + 1:
        equity_curve.append(equity)
    
    if position != 0:
        price = close_prices[-1]
        direction = 1 if position == 1 else -1
        price_diff = (price - entry_price) * direction
        pnl_pips = price_diff * 100 - TOTAL_COST_PIPS
        pnl = pnl_pips * PIP_VALUE_PER_LOT * position_lot_size
        trades.append({'pnl': pnl, 'pips': pnl_pips})
        equity += pnl
        pnl_list.append(pnl)
        if pnl > 0:
            wins += 1
        else:
            losses += 1
    
    total_trades = len(trades)
    if total_trades > 0:
        win_rate = 100 * wins / total_trades
        avg_pnl = np.mean(pnl_list)
        avg_pips = np.mean([t['pips'] for t in trades])
        
        returns = np.diff(equity_curve) / np.maximum(equity_curve[:-1], 1)
        returns = returns[np.isfinite(returns)]
        sharpe = np.sqrt(252 * 288) * np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        
        peak = INITIAL_BALANCE
        max_dd = 0
        for e in equity_curve:
            if e > peak:
                peak = e
            dd = (peak - e) / peak * 100 if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
        
        total_return = (equity - INITIAL_BALANCE) / INITIAL_BALANCE * 100
        
        start = pd.Timestamp(period_start)
        end = pd.Timestamp(period_end)
        weeks = max((end - start).days / 7, 1)
        trades_per_week = total_trades / weeks
    else:
        win_rate = avg_pnl = avg_pips = sharpe = max_dd = total_return = trades_per_week = 0
    
    return {
        'period': period_name,
        'start': period_start,
        'end': period_end,
        'total_trades': total_trades,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'avg_profit_pips': avg_pips,
        'avg_profit_dollars': avg_pnl,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'total_return_pct': total_return,
        'final_equity': equity,
        'net_profit': equity - INITIAL_BALANCE,
        'trades_per_week': trades_per_week
    }


def main():
    print("="*70)
    print("3-YEAR BACKTEST - EXNESS STANDARD (30 pip spread)")
    print("="*70)
    print(f"Configuration:")
    print(f"  Lot Size: {FIXED_LOT_SIZE}")
    print(f"  Confidence: {CONFIDENCE_THRESHOLD}")
    print(f"  Min Hold: {MIN_HOLD_BARS} bars")
    print(f"  Spread: {SPREAD_PIPS} pips")
    print(f"  Commission: {COMMISSION}")
    print(f"  Initial Balance: ${INITIAL_BALANCE:,.2f}")
    
    # Define periods
    periods = [
        ('2023-04-01', '2024-03-31', 'Year 1 (2023-2024)'),
        ('2024-04-01', '2025-03-31', 'Year 2 (2024-2025)'),
        ('2025-04-01', '2026-03-31', 'Year 3 (2025-2026)'),
    ]
    
    results = []
    total_profit = 0
    total_trades = 0
    
    for start, end, name in periods:
        result = run_backtest(start, end, name)
        if result:
            results.append(result)
            total_profit += result['net_profit']
            total_trades += result['total_trades']
    
    # Print summary
    print("\n" + "="*70)
    print("YEAR-BY-YEAR RESULTS")
    print("="*70)
    print(f"{'Period':<20} | {'Trades':>7} | {'Win%':>6} | {'Profit $':>10} | {'DD%':>6} | {'Sharpe':>7}")
    print("-"*70)
    
    for r in results:
        print(f"{r['period']:<20} | {r['total_trades']:>7} | {r['win_rate']:>5.1f}% | ${r['net_profit']:>9,.2f} | {r['max_dd']:>5.1f}% | {r['sharpe']:>7.4f}")
    
    print("-"*70)
    print(f"{'TOTAL 3 YEARS':<20} | {total_trades:>7} |        | ${total_profit:>9,.2f}")
    print("="*70)
    
    # Calculate overall metrics
    if results:
        avg_sharpe = np.mean([r['sharpe'] for r in results])
        max_dd_overall = max([r['max_dd'] for r in results])
        avg_win_rate = np.mean([r['win_rate'] for r in results])
        
        print("\n3-YEAR SUMMARY:")
        print(f"  Total Trades: {total_trades}")
        print(f"  Average Win Rate: {avg_win_rate:.2f}%")
        print(f"  Total Net Profit: ${total_profit:,.2f}")
        print(f"  Average Sharpe Ratio: {avg_sharpe:.4f}")
        print(f"  Maximum Drawdown (any year): {max_dd_overall:.2f}%")
        
        # Annualized return
        years = 3
        annualized_return = ((INITIAL_BALANCE + total_profit) / INITIAL_BALANCE) ** (1/years) - 1
        print(f"  Annualized Return: {annualized_return*100:.2f}%")
    
    # Save report
    os.makedirs('backtest_reports', exist_ok=True)
    with open('backtest_reports/3year_results_lot0022.txt', 'w') as f:
        f.write("="*70 + "\n")
        f.write("3-YEAR BACKTEST RESULTS - EXNESS STANDARD\n")
        f.write("="*70 + "\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  Lot Size: {FIXED_LOT_SIZE}\n")
        f.write(f"  Confidence Threshold: {CONFIDENCE_THRESHOLD}\n")
        f.write(f"  Min Hold Bars: {MIN_HOLD_BARS}\n")
        f.write(f"  Spread: {SPREAD_PIPS} pips\n")
        f.write(f"  Commission: {COMMISSION}\n")
        f.write(f"  Initial Balance: ${INITIAL_BALANCE:,.2f}\n\n")
        
        f.write("-"*70 + "\n")
        f.write(f"{'Period':<20} | {'Trades':>7} | {'Win%':>6} | {'Profit $':>10} | {'DD%':>6} | {'Sharpe':>7}\n")
        f.write("-"*70 + "\n")
        
        for r in results:
            f.write(f"{r['period']:<20} | {r['total_trades']:>7} | {r['win_rate']:>5.1f}% | ${r['net_profit']:>9,.2f} | {r['max_dd']:>5.1f}% | {r['sharpe']:>7.4f}\n")
        
        f.write("-"*70 + "\n")
        f.write(f"{'TOTAL 3 YEARS':<20} | {total_trades:>7} |        | ${total_profit:>9,.2f}\n")
        f.write("="*70 + "\n\n")
        
        if results:
            f.write(f"Summary:\n")
            f.write(f"  Total Trades: {total_trades}\n")
            f.write(f"  Average Win Rate: {avg_win_rate:.2f}%\n")
            f.write(f"  Total Net Profit: ${total_profit:,.2f}\n")
            f.write(f"  Average Sharpe Ratio: {avg_sharpe:.4f}\n")
            f.write(f"  Maximum Drawdown (any year): {max_dd_overall:.2f}%\n")
            f.write(f"  Annualized Return: {annualized_return*100:.2f}%\n")
    
    print("\nReport saved to backtest_reports/3year_results_lot0022.txt")
    
    # Recommendation
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    if max_dd_overall <= 40 and avg_sharpe > 0.4:
        print("✓ Configuration APPROVED for live trading")
        print(f"  - Sharpe ratio > 0.4: {avg_sharpe:.4f}")
        print(f"  - Max drawdown <= 40%: {max_dd_overall:.2f}%")
    else:
        print("✗ Configuration needs adjustment")
        print(f"  - Sharpe: {avg_sharpe:.4f} (target > 0.4)")
        print(f"  - Max DD: {max_dd_overall:.2f}% (target <= 40%)")


if __name__ == '__main__':
    main()