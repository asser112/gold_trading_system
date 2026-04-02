# Gold Trading System

Fully automated trading system for XAUUSD using market data, news sentiment, and ensemble of ML models.

## Installation

1. Clone this repository.
2. Install Python 3.12 and dependencies: `pip install -r requirements.txt`
3. Install MetaTrader 5 (build 4200+).
4. Copy `mt5_ea/gold_trading_ea.mq5` to `MQL5/Experts/` and compile.
5. Edit `config.yaml` with your API keys and paths.

## Running the System

### 1. Data Collection & Training
Run the full pipeline:
```bash
python run_pipeline.py