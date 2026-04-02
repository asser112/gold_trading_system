#!/usr/bin/env python3
"""
Ensemble Backtesting Module
- Uses trained XGBoost model with ensemble strategy
- Computes trading metrics for the last year with realistic costs
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
STOP_LOSS_PIPS = config['models'].get('ensemble', {}).get('stop_loss_pips', 50)
MIN_HOLD_BARS = config['models'].get('ensemble', {}).get('min_hold_bars', 60)

PIP_SIZE = 0.01
LOT_SIZE_OZ = 100
PIP_VALUE_PER_LOT = 1.0  # $1 per pip for XAUUSD (1 lot = 100 oz)
FIXED_LOT_SIZE = 0.01  # Fallback if risk_per_trade not set


def backtest_ensemble(period_start='2023-04-01', period_end='2026-04-01', verbose=True):
    """Run backtest for specified period."""
    if verbose:
        print("Loading data...")
    
    df = pd.read_parquet('data/processed/features_target_m5.parquet')
    df = df.dropna()
    
    if verbose:
        print(f"Total data: {len(df)} bars")
    
    df = df.loc[period_start:period_end]
    if verbose:
        print(f"Backtest period: {df.index[0]} to {df.index[-1]}")
        print(f"Bars: {len(df)}")
    
    print("Loading XGBoost model...")
    xgb_model = joblib.load('models/xgboost/xgboost_best.pkl')
    feature_cols = [c for c in df.columns if c != 'target']
    
    print("Loading close prices...")
    conn = sqlite3.connect('data/gold_trading.db')
    close_df = pd.read_sql(
        f"SELECT timestamp, close FROM ohlc_m5 WHERE timestamp >= '{start_date}' AND timestamp <= '{end_date}' ORDER BY timestamp",
        conn,
        index_col='timestamp',
        parse_dates=['timestamp']
    )
    conn.close()
    close_df = close_df[~close_df.index.duplicated(keep='first')]
    df_aligned = df.join(close_df[['close']], how='inner')
    close_prices = df_aligned['close'].values
    
    print(f"Aligned data: {len(df_aligned)} bars")
    
    print("Running backtest...")
    
    X = df_aligned[feature_cols].values
    probs = xgb_model.predict_proba(X)
    preds = xgb_model.predict(X)
    
    raw_signal = np.where(preds == 0, -1, np.where(preds == 2, 1, 0))
    confidence = np.max(probs, axis=1)
    
    signal_threshold = config['models']['ensemble']['confidence_threshold']
    final_signals = np.where(confidence >= signal_threshold, raw_signal, 0)
    
    equity = INITIAL_BALANCE
    equity_curve = [equity]
    
    trades = []
    wins = 0
    losses = 0
    pnl_list = []
    
    TOTAL_COST_PIPS = SPREAD_PIPS + SLIPPAGE_PIPS + (COMMISSION / PIP_SIZE)
    MIN_HOLD_BARS = config['models'].get('ensemble', {}).get('min_hold_bars', 60)
    RISK_PERCENT = config['trading']['risk_percent'] / 100
    
    position = 0
    entry_price = 0
    entry_idx = 0
    position_lot_size = 0
    entry_equity = INITIAL_BALANCE
    
    for i in range(len(df_aligned)):
        signal = final_signals[i]
        price = close_prices[i]
        
        if position == 0 and signal != 0:
            entry_equity = equity
            
            if RISK_PER_TRADE is not None:
                # XAUUSD: 1 pip = $1 per 1 lot (100 oz)
                # Lot size = risk_amount / stop_loss_pips
                # This gives lot in standard lots (where 1 lot = $1/pip)
                risk_amount = equity * (RISK_PER_TRADE / 100)
                lot_size = risk_amount / STOP_LOSS_PIPS
                lot_size = max(0.01, min(lot_size, 1.0))  # Clamp between 0.01 and 1.0  # Clamp between 0.01 and 1.0
            else:
                lot_size = FIXED_LOT_SIZE
            
            if i < 5 or i % 500000 == 0:
                print(f"  Open position at {i}: equity={equity:.2f}, lot={lot_size:.4f}")
            
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
                
                if len(trades) < 3 or (len(trades) == 5000):
                    print(f"  Close trade {len(trades)}: pnl_pips={pnl_pips:.2f}, pnl=${pnl:.2f}, lot={position_lot_size:.4f}")
                
                trades.append({
                    'entry_idx': entry_idx,
                    'exit_idx': i,
                    'type': 'long' if position == 1 else 'short',
                    'entry': entry_price,
                    'exit': price,
                    'pnl': pnl,
                    'pips': pnl_pips
                })
                
                equity += pnl
                pnl_list.append(pnl)
                
                if pnl > 0:
                    wins += 1
                else:
                    losses += 1
                
                position = signal
                entry_price = price
                entry_idx = i
        else:
            equity_curve.append(equity)
        
        if i % 100000 == 0:
            print(f"  Progress: {i}/{len(df_aligned)} ({100*i/len(df_aligned):.1f}%)")
    
    if len(equity_curve) < len(df_aligned) + 1:
        equity_curve.append(equity)
    
    if position != 0:
        price = close_prices[-1]
        direction = 1 if position == 1 else -1
        price_diff = (price - entry_price) * direction
        pnl_pips = price_diff * 100 - TOTAL_COST_PIPS
        pnl = pnl_pips * PIP_VALUE_PER_LOT * position_lot_size
        
        trades.append({
            'entry_idx': entry_idx,
            'exit_idx': len(df_aligned) - 1,
            'type': 'long' if position == 1 else 'short',
            'entry': entry_price,
            'exit': price,
            'pnl': pnl,
            'pips': pnl_pips
        })
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
        
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        weeks = (end - start).days / 7
        trades_per_week = total_trades / weeks if weeks > 0 else 0
    else:
        win_rate = 0
        avg_pnl = 0
        avg_pips = 0
        sharpe = 0
        max_dd = 0
        total_return = (equity - INITIAL_BALANCE) / INITIAL_BALANCE * 100
        trades_per_week = 0
    
    metrics = {
        'total_trades': total_trades,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'total_return_pct': total_return,
        'avg_profit_per_trade': avg_pnl,
        'avg_pips_per_trade': avg_pips,
        'sharpe_ratio': sharpe,
        'max_drawdown_pct': max_dd,
        'final_equity': equity,
        'trades_per_week': trades_per_week,
        'start_date': str(df_aligned.index[0]),
        'end_date': str(df_aligned.index[-1]),
    }
    
    print("="*50)
    print("ENSEMBLE BACKTEST - Exness Standard (30 pip spread)")
    print("="*50)
    print(f"Period: {metrics['start_date']} to {metrics['end_date']}")
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"  Wins: {metrics['wins']}, Losses: {metrics['losses']}")
    print(f"Win Rate: {metrics['win_rate']:.2f}%")
    print(f"Total Return: {metrics['total_return_pct']:.2f}%")
    print(f"Avg Profit/Trade: ${metrics['avg_profit_per_trade']:.2f}")
    print(f"Avg Pips/Trade: {metrics['avg_pips_per_trade']:.2f}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
    print(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
    print(f"Final Equity: ${metrics['final_equity']:.2f}")
    print("="*50)
    
    os.makedirs('backtest_reports', exist_ok=True)
    output_file = 'backtest_reports/last_year_results_exness_30pip.txt'
    with open(output_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("ENSEMBLE BACKTEST RESULTS (Last 12 Months)\n")
        f.write("="*60 + "\n\n")
        f.write(f"Period: {metrics['start_date']} to {metrics['end_date']}\n")
        f.write(f"Initial Balance: ${INITIAL_BALANCE:,.2f}\n")
        f.write(f"Spread: {SPREAD_PIPS} pips\n")
        f.write(f"Commission: {COMMISSION*100:.2f}%\n")
        f.write(f"Slippage: {SLIPPAGE_PIPS} pips\n")
        f.write(f"Risk per Trade: {RISK_PER_TRADE}%\n")
        f.write(f"Stop Loss Pips: {STOP_LOSS_PIPS}\n")
        f.write(f"Confidence Threshold: {CONFIDENCE_THRESHOLD}\n")
        f.write("-"*60 + "\n\n")
        f.write(f"Total Trades: {metrics['total_trades']}\n")
        f.write(f"  Wins: {metrics['wins']}\n")
        f.write(f"  Losses: {metrics['losses']}\n")
        f.write(f"Average Trades per Week: {metrics['trades_per_week']:.2f}\n")
        f.write(f"Win Rate: {metrics['win_rate']:.2f}%\n")
        f.write(f"Total Return: {metrics['total_return_pct']:.2f}%\n")
        f.write(f"Average Profit per Trade: ${metrics['avg_profit_per_trade']:.2f}\n")
        f.write(f"Average Pips per Trade: {metrics['avg_pips_per_trade']:.2f}\n")
        f.write(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}\n")
        f.write(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%\n")
        f.write(f"Final Equity: ${metrics['final_equity']:.2f}\n")
        f.write(f"Total Net Profit: ${equity - INITIAL_BALANCE:,.2f}\n")
        f.write("\n" + "="*60 + "\n")
        f.write("Model: Ensemble Strategy (XGBoost + News Sentiment)\n")
        f.write(f"Features: {len(feature_cols)} (with news sentiment)\n")
        f.write(f"Confidence Threshold: {signal_threshold}\n")
        f.write("Note: Full year data (2025-04-01 to 2026-04-01)\n")
        f.write("="*60 + "\n")
    
    pd.DataFrame({'equity': equity_curve}).to_csv('backtest_reports/equity_curve.csv', index=False)
    
    print(f"\nMetrics saved to {output_file}")
    return metrics


if __name__ == '__main__':
    backtest_ensemble()
