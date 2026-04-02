#!/usr/bin/env python3
"""
Parameter Sweep for Exness Standard Account (30 pip spread)
Tests combinations of confidence threshold and min hold time
"""
import os
import sys
import subprocess
import yaml

# Configuration for sweep
CONFIDENCE_THRESHOLDS = [0.95, 0.96, 0.97, 0.98, 0.99]
MIN_HOLD_BARS_OPTIONS = [60, 120, 180, 240]

# Fixed settings
SPREAD = 30
COMMISSION = 0

results = []

print("="*70)
print("PARAMETER SWEEP - Exness Standard (30 pip spread)")
print("="*70)
print(f"Testing {len(CONFIDENCE_THRESHOLDS)} confidence thresholds x {len(MIN_HOLD_BARS_OPTIONS)} hold times = {len(CONFIDENCE_THRESHOLDS) * len(MIN_HOLD_BARS_OPTIONS)} combinations")
print("="*70)

for conf in CONFIDENCE_THRESHOLDS:
    for hold in MIN_HOLD_BARS_OPTIONS:
        print(f"\n--- Testing: confidence={conf}, min_hold={hold} ---")
        
        # Update config
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        config['models']['ensemble']['confidence_threshold'] = conf
        config['models']['ensemble']['min_hold_bars'] = hold
        
        with open('config.yaml', 'w') as f:
            yaml.dump(config, f)
        
        # Run backtest
        result = subprocess.run(
            ['python', 'scripts/backtest_ensemble.py'],
            capture_output=True,
            text=True
        )
        
        # Parse results from output
        output = result.stdout + result.stderr
        
        # Extract metrics
        trades = wins = losses = net_profit = sharpe = max_dd = 0
        
        for line in output.split('\n'):
            if 'Total Trades:' in line:
                trades = int(line.split(':')[1].strip())
            elif 'Wins:' in line and ',' in line:
                parts = line.split(',')
                wins = int(parts[0].split(':')[1].strip())
                losses = int(parts[1].split(':')[1].strip())
            elif 'Sharpe Ratio:' in line:
                sharpe = float(line.split(':')[1].strip())
            elif 'Max Drawdown:' in line:
                max_dd = float(line.split(':')[1].strip().replace('%', ''))
            elif 'Total Net Profit:' in line:
                net_profit = float(line.split('$')[1].replace(',', ''))
        
        results.append({
            'confidence': conf,
            'min_hold': hold,
            'trades': trades,
            'wins': wins,
            'losses': losses,
            'net_profit': net_profit,
            'sharpe': sharpe,
            'max_dd': max_dd
        })
        
        print(f"  Trades: {trades}, Net Profit: ${net_profit:.2f}, Sharpe: {sharpe:.4f}, Max DD: {max_dd:.2f}%")

print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)
print(f"{'Conf':>6} | {'Hold':>5} | {'Trades':>6} | {'Profit':>10} | {'Sharpe':>7} | {'Max DD':>8}")
print("-"*70)

for r in results:
    print(f"{r['confidence']:>6.2f} | {r['min_hold']:>5} | {r['trades']:>6} | ${r['net_profit']:>9.2f} | {r['sharpe']:>7.4f} | {r['max_dd']:>7.2f}%")

# Find best combination
print("\n" + "="*70)
print("BEST CONFIGURATION")
print("="*70)

# Priority: Sharpe > 0.4, Max DD < 40%, then highest Sharpe
best = None
for r in results:
    if r['max_dd'] < 40 and r['sharpe'] > 0.4:
        if best is None or r['sharpe'] > best['sharpe']:
            best = r

# If no combination meets all criteria, pick best by Sharpe
if best is None:
    best = max(results, key=lambda x: x['sharpe'])

print(f"Confidence Threshold: {best['confidence']}")
print(f"Min Hold Bars: {best['min_hold']}")
print(f"Total Trades: {best['trades']}")
print(f"Net Profit: ${best['net_profit']:.2f}")
print(f"Sharpe Ratio: {best['sharpe']:.4f}")
print(f"Max Drawdown: {best['max_dd']:.2f}%")

# Update config with best settings
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
config['models']['ensemble']['confidence_threshold'] = best['confidence']
config['models']['ensemble']['min_hold_bars'] = best['min_hold']

with open('config.yaml', 'w') as f:
    yaml.dump(config, f)

print("\nConfig updated with best settings.")