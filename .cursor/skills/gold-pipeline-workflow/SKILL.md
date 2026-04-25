---
name: gold-pipeline-workflow
description: Guides running, debugging, and extending the XAUUSD gold trading ML pipeline. Use when working with pipeline scripts (01-09), debugging training failures, re-running individual steps, or adding new features to the pipeline.
---

# Gold Trading ML Pipeline

## Pipeline overview

Orchestrated by `run_pipeline.py`. Steps run sequentially:

| Script | Purpose | Key outputs |
|--------|---------|-------------|
| `scripts/01_data_collection.py` | Fetch OHLCV + news from yfinance, Dukascopy, Alpha Vantage, GNews | `data/gold_trading.db` |
| `scripts/02_feature_engineering.py` | TA indicators, news sentiment, lag features | processed tables in DB |
| `scripts/03_train_xgboost.py` | XGBoost with Optuna tuning | `models/xgboost/` |
| `scripts/04_train_transformer.py` | PyTorch Transformer (seq_len=60) | `models/transformer/` |
| `scripts/05_train_rl_agent.py` | PPO agent via stable-baselines3 | `models/rl_agent/` |
| `scripts/06_ensemble.py` | Meta-model combining all three | `models/ensemble/` |
| `scripts/08_backtester.py` | Backtesting library simulation | `backtest_reports/` |

`scripts/07_trading_logic.py` is the **live signal generator** — not part of the training pipeline.

## Running

```bash
# Full pipeline
python run_pipeline.py

# Single step (from project root)
python scripts/03_train_xgboost.py

# Live signal generator (separate process)
python scripts/07_trading_logic.py
```

## Key config (`config.yaml`)

- `data.db_path` — **Windows absolute path by default**; update to local path on macOS/Linux
- `data.start_date` — controls training window
- `models.transformer.epochs` / `models.rl.episodes` — tune for speed vs quality
- `models.ensemble.confidence_threshold` — 0.60 default; raise to reduce false signals
- `logging.file` — also has hardcoded Windows path; update if needed

## Common failure modes

**`db_path` not found:** Config has a Windows path. Update `data.db_path` and `logging.file` to local paths.

**Missing `src` module:** `scripts/07_trading_logic.py` has a legacy `from src.trading_bot import TradingBot` reference that no longer exists. The file has a standalone fallback — check that fallback is active.

**Transformer OOM:** Reduce `models.transformer.batch_size` (64 → 32) or `d_model` (128 → 64).

**RL agent not learning:** Increase `models.rl.episodes` (10 is very low for production). Check gym env reward function in `05_train_rl_agent.py`.

**Pipeline timeout:** Each script has a 2-hour timeout. Long RL training may hit it; run steps individually.

## Debugging tips

```bash
# Watch logs in real time
tail -f logs/system.log
tail -f logs/signal_generator.log

# Check DB tables
python -c "import sqlite3; c=sqlite3.connect('data/gold_trading.db'); print([r[0] for r in c.execute(\"SELECT name FROM sqlite_master WHERE type='table'\")])"

# Check feature count before/after engineering
python -c "import sqlite3,pandas as pd; c=sqlite3.connect('data/gold_trading.db'); print(pd.read_sql('SELECT * FROM features LIMIT 1',c).shape)"
```

## Extending the pipeline

- Add a new script `scripts/10_my_step.py` and register it in `run_pipeline.py`'s `pipeline_steps` list.
- New features belong in `02_feature_engineering.py`; add column to the DB features table.
- New models: add to `06_ensemble.py` as an additional base learner.
