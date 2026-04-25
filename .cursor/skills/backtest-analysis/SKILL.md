---
name: backtest-analysis
description: Guides running and interpreting backtests for the gold trading system. Use when running backtests, reading backtest reports, comparing strategies, tuning risk parameters, or analyzing equity curves.
---

# Backtest Analysis

## Running a backtest

```bash
# Standalone (uses config.yaml settings)
python scripts/08_backtester.py

# Via full pipeline (runs as step 7)
python run_pipeline.py
```

Output lands in `backtest_reports/`:
- `trading_metrics*.txt` — human-readable performance summary
- `equity_curve.html` — interactive Plotly chart
- `equity_curve.csv` — raw equity series
- `summary_stats.csv` — aggregated stats

## Key metrics to evaluate

| Metric | Acceptable range | Notes |
|--------|-----------------|-------|
| Win Rate | > 45% (directional) | Lower is ok with good R:R |
| Sharpe Ratio | > 0.5 | Current demo: -0.34 (ensemble uses EMA stub in backtest) |
| Max Drawdown | < 20% | Current reports show -83% — indicates lot sizing issue |
| Trades/week | > 0.5 | Too few = signal too conservative |

**Important:** `backtest_reports/trading_metrics.txt` is labeled "Simple EMA Crossover — for demonstration only." The live system uses the ensemble ML model. Treat these numbers as a baseline, not ML performance.

## Config parameters that affect backtest

```yaml
backtest:
  initial_balance: 10000
  commission: 0
  slippage: 0.0002
  spread: 0.0030         # 30 pip spread (Exness-style)

trading:
  lot_size: 0.03
  risk_percent: 0        # 0 = fixed lot; > 0 = % of balance per trade
  atr_multiplier_sl: 1.5
  atr_multiplier_tp: 2.5

models:
  ensemble:
    confidence_threshold: 0.60  # raise to filter low-confidence signals
    min_hold_bars: 180          # minimum bars between trades
```

## Common issues

**Too few trades:** `confidence_threshold` too high or `min_hold_bars` too large. Lower threshold or reduce hold bars.

**Large drawdown with small lot:** Usually means the strategy holds losing trades too long — check `atr_multiplier_sl`.

**Backtest vs live discrepancy:** The backtester runs `08_backtester.py` (simple strategy); `07_trading_logic.py` uses the ensemble. They are not the same. For ML-based backtest, apply the ensemble model's predictions to historical data manually.

## Reading existing reports

```bash
# Quick summary
cat backtest_reports/trading_metrics.txt

# Compare different lot sizes
ls backtest_reports/last_year_results*.txt
# Files: lot_0.01, lot_0.02, lot_0.05, dynamic_risk, fixed_lot, risk_1pct_conf90

# View equity curve
open backtest_reports/equity_curve.html   # macOS
```

## Comparing runs

```bash
python -c "
import os, glob
for f in sorted(glob.glob('backtest_reports/last_year*.txt')):
    lines = open(f).readlines()
    ret = next((l for l in lines if 'Total Return' in l), '')
    dd  = next((l for l in lines if 'Max Drawdown' in l), '')
    print(os.path.basename(f))
    print(' ', ret.strip(), '|', dd.strip())
    print()
"
```
