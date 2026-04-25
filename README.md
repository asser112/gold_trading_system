# Gold Trading System

Fully automated XAUUSD trading system using an ensemble of ML models (XGBoost, Transformer, RL agent) combined with news sentiment. Generates live signals consumed by a MetaTrader 5 Expert Advisor.

---

## Table of Contents

1. [Requirements](#1-requirements)
2. [Installation](#2-installation)
3. [Configuration](#3-configuration)
4. [Adding Your MT5 Account](#4-adding-your-mt5-account)
5. [Finding Your MT5 Terminal ID](#5-finding-your-mt5-terminal-id)
6. [Deploying the EA](#6-deploying-the-ea)
7. [Running the Pipeline](#7-running-the-pipeline)
8. [Running Live Trading](#8-running-live-trading)
9. [Dashboard](#9-dashboard)
10. [Auto-Start on Boot](#10-auto-start-on-boot)
11. [Project Structure](#11-project-structure)

---

## 1. Requirements

| Requirement | Version |
|---|---|
| Windows | 10 / 11 / Server 2019+ |
| Python | 3.12 |
| MetaTrader 5 | Build 4200+ |
| Broker account | Any MT5-compatible broker (Exness, IC Markets, Pepperstone, etc.) |

> **VPS recommendation:** 4 vCPU / 6 GB RAM / 150 GB SSD Windows Server. ForexVPS Edge plan or equivalent.

---

## 2. Installation

```bat
:: 1. Clone the repository
git clone https://github.com/your-repo/gold_trading_system.git
cd gold_trading_system

:: 2. Install Python dependencies
pip install -r requirements.txt
```

---

## 3. Configuration

All settings live in **`config.yaml`** in the project root. Open it and update the following:

### API Keys

```yaml
data:
  alphavantage_key: YOUR_ALPHAVANTAGE_KEY   # https://www.alphavantage.co/support/#api-key
  gnews_api_key: YOUR_GNEWS_KEY             # https://gnews.io
  news_api_key: YOUR_NEWSAPI_KEY            # https://newsapi.org
```

### Trading Parameters

```yaml
trading:
  symbol: XAUUSD          # or XAUUSDr depending on your broker
  lot_size: 0.03          # fixed lot size (used when risk_percent is 0)
  risk_percent: 0         # set > 0 to use % risk per trade instead of fixed lot
  max_spread: 30          # in points — EA skips entry if spread is wider
  magic_number: 123456    # unique ID for this EA's trades
  mt5_terminal_id: YOUR_TERMINAL_ID   # see Section 5 below
```

### Telegram Alerts (optional)

```yaml
telegram:
  enabled: true
  api_id: YOUR_API_ID
  api_hash: YOUR_API_HASH
  chat_id: "@your_channel"
```

Leave `enabled: false` to disable.

---

## 4. Adding Your MT5 Account

### Step 1 — Install MetaTrader 5

Download MT5 from your broker's website (not the generic MetaQuotes version, as brokers may have custom builds).

### Step 2 — Log in to your trading account

1. Open MetaTrader 5
2. Go to **File > Open an Account** (or press Ctrl+N)
3. Search for your broker by name
4. Select **"Connect to an existing account"**
5. Enter your **Login**, **Password**, and select the correct **server**
6. Click **Finish**

Your account balance should appear in the bottom toolbar once connected.

### Step 3 — Verify the symbol name

The default symbol in `config.yaml` is `XAUUSD`. Some brokers use:
- `XAUUSDr` (Exness raw spread)
- `XAUUSD.` (with a dot suffix)
- `GOLD`

Check your broker's **Market Watch** (Ctrl+M) and update `trading.symbol` in `config.yaml` to match exactly.

---

## 5. Finding Your MT5 Terminal ID

The terminal ID is a long hex string that identifies your MT5 installation's data folder. It is needed to mirror signals directly into MT5's file sandbox.

### Method 1 — Browse the folder

Open **File Explorer** and navigate to:
```
%APPDATA%\MetaQuotes\Terminal\
```
You will see one or more folders with hex names like `D0E8209F77C8CF37AD8BF550E51FF075`. If you have one MT5 installation, there will be exactly one. Copy that folder name.

### Method 2 — From inside MT5

1. In MT5 go to **File > Open Data Folder**
2. The folder that opens is your terminal data folder
3. Copy the last segment of the path (the hex ID)

### Method 3 — From MetaEditor

1. Press **F4** in MT5 to open MetaEditor
2. Go to **Tools > Options > Compiler**
3. The path shown includes the terminal ID

### Set the ID

Once you have it, set it in two places:

**`config.yaml`:**
```yaml
trading:
  mt5_terminal_id: D0E8209F77C8CF37AD8BF550E51FF075   # replace with yours
```

**`deploy.bat`** (line 14):
```bat
set "MT5_TERMINAL_ID=D0E8209F77C8CF37AD8BF550E51FF075"
```

---

## 6. Deploying the EA

After setting the terminal ID, run the deployment script:

```bat
deploy.bat
```

This will:
1. Copy `gold_trading_ea.mq5` to MT5's Experts folder
2. Copy `signal_reader.mqh` to MT5's Include folder
3. Place an initial `signal.txt` in MT5's Files folder

### Compile the EA in MetaEditor

1. In MT5 press **F4** to open MetaEditor
2. In the Navigator panel open **Experts > gold_trading_ea**
3. Press **F7** to compile — you should see `0 errors, 0 warnings`

### Attach the EA to a chart

1. In MT5, open an **XAUUSD chart** (match the symbol from `config.yaml`)
2. Set the timeframe to **M5**
3. Drag **gold_trading_ea** from the Navigator panel onto the chart
4. In the dialog that appears:
   - **Common tab:** check `Allow live trading` and `Allow DLL imports`
   - **Inputs tab:** verify these values match your `config.yaml`:

| Parameter | Default | Notes |
|---|---|---|
| `SignalFileName` | `signal.txt` | Do not change |
| `MagicNumber` | `123456` | Must match `config.yaml` |
| `LotSize` | `0.03` | Fixed lot when `RiskPercent` is 0 |
| `RiskPercent` | `0` | % of balance per trade (0 = use LotSize) |
| `MaxSpread` | `30` | Points |

5. Click **OK**
6. Click the **Auto Trading** button in the toolbar (turns green when active)

The EA smiley face should appear in the top-right corner of the chart.

---

## 7. Running the Pipeline

Train all models from scratch:

```bat
python run_pipeline.py
```

This runs these steps in order:

| Step | Script | Output |
|---|---|---|
| 1 | `01_data_collection.py` | `data/gold_trading.db` |
| 2 | `02_feature_engineering.py` | Feature tables in DB |
| 3 | `03_train_xgboost.py` | `models/xgboost/` |
| 4 | `04_train_transformer.py` | `models/transformer/` |
| 5 | `05_train_rl_agent.py` | `models/rl_agent/` |
| 6 | `06_ensemble.py` | `models/ensemble/` |
| 7 | `08_backtester.py` | `backtest_reports/` |

You can also run any step individually:

```bat
python scripts\03_train_xgboost.py
```

Logs are written to `logs/system.log`.

---

## 8. Running Live Trading

Once models are trained and the EA is attached and running:

```bat
start_trading.bat
```

This starts the signal generator loop (`scripts/07_trading_logic.py`). It runs every 60 seconds, writes a signal to `mt5_ea/signal.txt`, and mirrors it to the MT5 Files folder. The EA picks it up and acts accordingly.

To stop: press **Ctrl+C** in the terminal window. The signal generator will write a final `hold` signal before exiting.

---

## 9. Dashboard

Launch the Streamlit monitoring dashboard:

```bat
streamlit run dashboard\app.py
```

Open `http://localhost:8501` in your browser. Shows:
- Equity curve from the last backtest
- Latest signal (direction, confidence, SL, TP)
- News sentiment chart
- Recent trades (if trade logging is active)
- Live price candlestick chart

---

## 10. Auto-Start on Boot

To make the signal generator start automatically when the VPS reboots:

```bat
setup_autostart.bat
```

This adds a Windows Startup shortcut. After running it, the signal generator will launch automatically on every login.

---

## 11. Project Structure

```
gold_trading_system/
├── config.yaml                  # All settings — edit this first
├── run_pipeline.py              # Full training pipeline orchestrator
│
├── scripts/
│   ├── 01_data_collection.py    # OHLCV + news data → SQLite
│   ├── 02_feature_engineering.py
│   ├── 03_train_xgboost.py
│   ├── 04_train_transformer.py
│   ├── 05_train_rl_agent.py
│   ├── 06_ensemble.py           # Meta-model combining all three
│   ├── 07_trading_logic.py      # Live signal generator (run this for live trading)
│   ├── 08_backtester.py
│   └── 09_monitoring.py
│
├── mt5_ea/
│   ├── gold_trading_ea.mq5      # MetaTrader 5 Expert Advisor
│   ├── signal_reader.mqh        # MQL5 include for reading signal.txt
│   └── signal.txt               # Signal file (written by Python, read by EA)
│
├── dashboard/
│   └── app.py                   # Streamlit monitoring dashboard
│
├── models/                      # Trained model artifacts (created by pipeline)
├── data/                        # SQLite database and raw data
├── backtest_reports/            # Backtest output files
├── logs/                        # Runtime logs
│
├── deploy.bat                   # Copies EA files into MT5 — run once after setup
├── start_trading.bat            # Starts the live signal generator
├── setup_autostart.bat          # Adds signal generator to Windows Startup
└── requirements.txt
```

---

## 12. SaaS Backend (Signal Subscription Service)

The `backend/` folder contains a subscription web portal that lets users pay with crypto and receive signals in their MT5 EA.

### Setup

```bat
cd backend
pip install -r requirements.txt
copy .env.example .env
:: Edit .env with your NOWPayments keys and secrets
```

### Running the backend

```bat
:: From project root
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### Connect the signal generator

In `config.yaml`, set:

```yaml
backend:
  url: "https://your-domain.com"
  internal_signal_secret: "same value as INTERNAL_SIGNAL_SECRET in .env"
```

### EA configuration for subscribers

In MT5 EA inputs:
- `SignalUrl` = `https://your-domain.com/api/signal`
- `ApiKey` = the key shown in the user's dashboard
- Leave `SignalFileName` blank

**Important:** In MT5, go to **Tools > Options > Expert Advisors** and add your domain to the allowed WebRequest URLs list. Otherwise MT5 will block the HTTP call.

### NOWPayments setup

1. Create a free account at [nowpayments.io](https://nowpayments.io)
2. Get your API key from the dashboard
3. Set your IPN (webhook) URL to `https://your-domain.com/webhooks/nowpayments`
4. Copy the IPN secret to `NOWPAYMENTS_IPN_SECRET` in `.env`

---

## Signal File Format

The signal file (`mt5_ea/signal.txt`) is JSON:

```json
{
  "signal": "buy",
  "confidence": 0.72,
  "sl": 3180.50,
  "tp": 3250.00,
  "reason": "ML ensemble buy signal",
  "timestamp": "2026-04-25T10:30:00.000000"
}
```

- `signal`: `"buy"`, `"sell"`, or `"hold"`
- `confidence`: 0.0–1.0. The EA ignores signals below `confidence_threshold` (default `0.60`) set in `config.yaml`
- `sl` / `tp`: absolute price levels in the instrument's quote currency
