# Gold Trading System

Automated XAUUSD trading system with **multiple ML bots** (e.g. **XGBoost**, **LightGBM + Session Filter**) registered in one backend. Each bot has its own **slug** (`xgboost-v1`, `lgbm-session-v1`, …), signal stream, and EA URL on **one domain**. Run several signal generators + MT5 accounts in parallel on the same VPS.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Requirements](#2-requirements)
3. [Installation](#3-installation)
4. [Training the Models](#4-training-the-models)
   - [XGBoost Pipeline](#41-xgboost-pipeline)
   - [LightGBM Pipeline](#42-lightgbm-pipeline)
5. [Server Configuration](#5-server-configuration)
   - [DNS](#51-dns)
   - [Caddy — Reverse Proxy & HTTPS](#52-caddy--reverse-proxy--https)
   - [Backend .env](#53-backend-env)
   - [Start the Backend](#54-start-the-backend)
   - [Multi-Bot API](#55-multi-bot-api)
   - [Auto-Start with NSSM](#56-auto-start-with-nssm)
   - [NOWPayments Webhooks](#57-nowpayments-webhooks)
6. [MT5 Account Setup](#6-mt5-account-setup)
   - [Finding the Terminal ID](#61-finding-the-terminal-id)
   - [Deploying the EA](#62-deploying-the-ea)
7. [Running Both Signal Generators](#7-running-both-signal-generators)
8. [Switching the Active Model](#8-switching-the-active-model)
9. [Backtesting](#9-backtesting)
10. [Monitoring Dashboard](#10-monitoring-dashboard)
11. [Project Structure](#11-project-structure)
12. [Signal File Format](#12-signal-file-format)
13. [Environment Variable Reference](#13-environment-variable-reference)

---

## 1. System Overview

```
Internet
   │
   └── gold.yepwoo.com ─── Caddy :443 ─── Uvicorn :8000 (single FastAPI backend)
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
   POST /internal/signal/xgboost-v1    POST /internal/signal/lgbm-session-v1
        │                     │                     │
   Signal gen A            Signal gen B         (more bots…)
   BOT_SLUG=xgboost-v1     BOT_SLUG=lgbm-session-v1
        │                     │
   MT5 #1                  MT5 #2
   GET …/api/signal/xgboost-v1    GET …/api/signal/lgbm-session-v1
```

Default bots are seeded on backend startup (`xgboost-v1`, `lgbm-session-v1`). Add new versions by inserting rows in the `bots` table (or extend [`backend/bot_defaults.py`](backend/bot_defaults.py)) and run a signal generator with `BOT_SLUG=new-slug`.

---

## 2. Requirements

| Requirement | Version |
|---|---|
| Windows | 10 / 11 / Server 2019+ |
| Python | 3.12 |
| MetaTrader 5 | Build 4200+ |
| Broker account | Any MT5-compatible broker |

> **VPS recommendation:** 4 vCPU / 8 GB RAM / 150 GB SSD Windows Server.
> Each extra signal generator + MT5 instance adds CPU/RAM; one backend process serves all bots.

---

## 3. Installation

Run these commands once on your VPS (RDP in first):

```bat
:: 1. Install Python 3.12 from https://python.org
::    Check "Add Python to PATH" during installation.

:: 2. Install Git from https://git-scm.com

:: 3. Clone the repository
git clone https://github.com/your-repo/gold_trading_system.git
cd gold_trading_system

:: 4. Install all dependencies (both pipeline and backend)
pip install -r requirements.txt
pip install -r backend\requirements.txt
```

Create the required directories if they do not exist:

```bat
mkdir backend\static
mkdir models\xgboost
mkdir models\lightgbm
mkdir models\scalers
mkdir data\processed
mkdir data\raw
mkdir backtest_reports
mkdir logs
```

---

## 4. Training the Models

Both pipelines start with the same data collection step. Run it once:

```bat
python scripts\01_data_collection.py
```

This downloads historical XAUUSD OHLCV data and stores it in `data/gold_trading.db`.

---

### 4.1 XGBoost Pipeline

```bat
:: Step 1 — Feature engineering (creates data/processed/features_target_m5.parquet)
python scripts\02_feature_engineering.py

:: Step 2 — Train XGBoost model (creates models/xgboost/xgboost_best.pkl)
python scripts\03_train_xgboost.py

:: Step 3 — Backtest (creates backtest_reports/last_year_results.txt)
python scripts\08_backtester.py
```

---

### 4.2 LightGBM Pipeline

The LightGBM pipeline is fully independent. It uses its own feature engineering script that adds session flags (`is_london`, `is_ny`, `is_overlap`) and H1 trend features (`h1_ema20`, `h1_ema50`, `h1_trend`). Training is restricted to London + NY session bars only.

```bat
:: Step 1 — LightGBM feature engineering (creates data/processed/features_lgbm_m5.parquet)
python scripts\02b_feature_engineering_lgbm.py

:: Step 2 — Train LightGBM model (creates models/lightgbm/lgbm_best.pkl)
python scripts\10_train_lightgbm.py

:: Step 3 — Backtest (creates backtest_reports/lgbm_last_year_results.txt)
python scripts\11_backtest_lightgbm.py
```

Review both backtest reports and compare:

```bat
type backtest_reports\last_year_results.txt
type backtest_reports\lgbm_last_year_results.txt
```

---

## 5. Server Configuration

### 5.1 DNS

Add one **A record** for your public hostname (subdomain) pointing to the VPS IP, e.g.:

| Type | Host | Value |
|------|------|-------|
| A | `gold` | `<VPS public IP>` |

DNS propagation takes 5–30 minutes.

---

### 5.2 Caddy — Reverse Proxy & HTTPS

Caddy handles HTTPS automatically using free Let's Encrypt certificates.

1. Download `caddy_windows_amd64.exe` from [caddyserver.com/download](https://caddyserver.com/download)
2. Place it in `C:\caddy\` and rename to `caddy.exe`
3. Open **Windows Firewall** → allow inbound TCP on ports **80** and **443**
4. Create `C:\caddy\Caddyfile`:

```
gold.yepwoo.com {
    reverse_proxy localhost:8000
}
```

5. Validate and run:

```bat
C:\caddy\caddy.exe validate --config C:\caddy\Caddyfile
C:\caddy\caddy.exe run --config C:\caddy\Caddyfile
```

---

### 5.3 Backend .env

Single `.env` for the whole service:

```bat
copy backend\.env.example backend\.env
notepad backend\.env
```

```env
DATABASE_URL=sqlite:///./backend/trading_saas.db
SECRET_KEY=<generate: python -c "import secrets; print(secrets.token_hex(32))">
NOWPAYMENTS_API_KEY=<from nowpayments.io>
NOWPAYMENTS_IPN_SECRET=<from nowpayments.io>
INTERNAL_SIGNAL_SECRET=<generate another random string>
BASE_URL=https://gold.yepwoo.com
SUBSCRIPTION_PRICE_USD=50.0
SUBSCRIPTION_DAYS=30
```

In `config.yaml`, match the same URL and secret for the signal generator:

```yaml
backend:
  url: "https://gold.yepwoo.com"
  internal_signal_secret: "<same value as INTERNAL_SIGNAL_SECRET above>"
```

Optional: `ENV_FILE` can point at an alternate env file (see [`backend/config.py`](backend/config.py)).

---

### 5.4 Start the Backend

```bat
cd C:\path\to\gold_trading_system
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Verify:

```bat
curl http://localhost:8000/
curl https://gold.yepwoo.com/
```

Create the admin/test user (once):

```bat
python backend\create_admin.py
```

On startup, the app creates tables, migrates old SQLite schemas if needed, and seeds default bots (`xgboost-v1`, `lgbm-session-v1`).

---

### 5.5 Multi-Bot API

| Endpoint | Purpose |
|----------|---------|
| `GET /api/bots` | List active bots (slug, name, description) |
| `GET /api/signal?api_key=…` | **Legacy:** latest signal for the `xgboost-v1` stream (falls back to global latest if that bot is missing) |
| `GET /api/signal/{slug}?api_key=…` | Latest signal for a specific bot, e.g. `xgboost-v1`, `lgbm-session-v1` |
| `POST /internal/signal` | **Legacy:** ingest signal (defaults to `xgboost-v1` when bot row exists) |
| `POST /internal/signal/{slug}` | Ingest signal for that bot (used by `07_trading_logic.py` with `BOT_SLUG`) |

Signal generators POST to `/internal/signal/{BOT_SLUG}` using `x-internal-secret: INTERNAL_SIGNAL_SECRET`.

---

### 5.6 Auto-Start with NSSM

```bat
nssm install GoldBackend "python" "-m uvicorn backend.main:app --host 127.0.0.1 --port 8000"
nssm set GoldBackend AppDirectory "C:\path\to\gold_trading_system"
nssm start GoldBackend

nssm install Caddy "C:\caddy\caddy.exe" "run --config C:\caddy\Caddyfile"
nssm start Caddy
```

---

### 5.7 NOWPayments Webhooks

In [NOWPayments](https://nowpayments.io) → **Settings > IPN**, set IPN URL to:

`https://gold.yepwoo.com/webhooks/nowpayments`

Copy the IPN secret into `NOWPAYMENTS_IPN_SECRET` in `backend/.env`.

---

## 6. MT5 Account Setup

You need **two MT5 accounts** — one for XGBoost, one for LightGBM. They can be on the same broker or different ones. Both can be demo accounts while testing.

### 6.1 Finding the Terminal ID

The terminal ID tells the signal generator where to write the signal file inside MT5's sandbox. Each MT5 installation has a unique ID.

**Method — Open the data folder from inside MT5:**

1. Open MT5
2. Go to **File > Open Data Folder**
3. The folder that opens will have a path like:
   `C:\Users\...\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075\`
4. Copy the last segment — that is your terminal ID

If you have two MT5 accounts open simultaneously (on the same VPS), each will have a different terminal ID. Browse:

```
%APPDATA%\MetaQuotes\Terminal\
```

You will see one folder per MT5 installation.

---

### 6.2 Deploying the EA

Run the deploy script once per MT5 account. It copies the EA and signal files into the correct MT5 folders.

**XGBoost MT5 account:**

```bat
:: In deploy.bat, set MT5_TERMINAL_ID to the XGBoost terminal ID
notepad deploy.bat
deploy.bat
```

**LightGBM MT5 account:**

Edit `deploy.bat` again, update `MT5_TERMINAL_ID` to the LightGBM terminal ID, then run it again.

**Compile and attach the EA** (do this for both MT5 accounts):

1. In MT5, press **F4** to open MetaEditor
2. Open **Experts > gold_trading_ea** in the Navigator
3. Press **F7** to compile — must show `0 errors, 0 warnings`
4. Open an **XAUUSD M5** chart
5. Drag `gold_trading_ea` from the Navigator onto the chart
6. In the EA dialog, set these inputs:

| Parameter | XGBoost Account | LightGBM Account |
|---|---|---|
| `SignalUrl` | `https://gold.yepwoo.com/api/signal/xgboost-v1` | `https://gold.yepwoo.com/api/signal/lgbm-session-v1` |
| `ApiKey` | *(same admin API key)* | *(same)* |
| `SignalFileName` | *(leave blank)* | *(leave blank)* |
| `MagicNumber` | `123456` | `123457` |
| `LotSize` | `0.03` | `0.03` |

Legacy: `https://gold.yepwoo.com/api/signal` still works and tracks the `xgboost-v1` stream when that bot exists.

7. In **Common** tab: check `Allow live trading` and `Allow DLL imports`
8. Click **OK** → click **Auto Trading** in the toolbar (turns green)

**Allow WebRequest in MT5** (required for both accounts):

Go to **Tools > Options > Expert Advisors** and add your site, e.g. `https://gold.yepwoo.com`

---

## 7. Running Both Signal Generators

Open two Command Prompt windows (or two NSSM services).

**XGBoost signal generator:**

```bat
:: Sets MODEL_TYPE=xgboost and BOT_SLUG=xgboost-v1
start_xgboost_signal.bat

:: Manual
set MODEL_TYPE=xgboost
set BOT_SLUG=xgboost-v1
python scripts\07_trading_logic.py
```

**LightGBM signal generator:**

```bat
:: Edit BACKEND_URL, INTERNAL_SIGNAL_SECRET, MT5_TERMINAL_ID first
start_lgbm_signal.bat

:: Manual (same BACKEND_URL / secret as backend\.env)
set MODEL_TYPE=lightgbm
set BOT_SLUG=lgbm-session-v1
set BACKEND_URL=https://gold.yepwoo.com
set INTERNAL_SIGNAL_SECRET=<same as backend/.env>
set MT5_TERMINAL_ID=<LightGBM terminal ID>
set SIGNAL_FILE_PATH=mt5_ea\signal_lgbm.txt
python scripts\07_trading_logic.py
```

> The LightGBM signal generator will automatically hold during Asian session hours (21:00–08:00 UTC) and only generate signals during London (08:00–17:00 UTC) and New York (13:00–21:00 UTC) sessions.

**Register signal generators as NSSM services** (run as Administrator):

```bat
nssm install GoldSignalXGB "python" "scripts\07_trading_logic.py"
nssm set GoldSignalXGB AppDirectory "C:\path\to\gold_trading_system"
nssm set GoldSignalXGB AppEnvironmentExtra "MODEL_TYPE=xgboost BOT_SLUG=xgboost-v1 BACKEND_URL=https://gold.yepwoo.com INTERNAL_SIGNAL_SECRET=<secret>"
nssm start GoldSignalXGB

nssm install GoldSignalLGBM "python" "scripts\07_trading_logic.py"
nssm set GoldSignalLGBM AppDirectory "C:\path\to\gold_trading_system"
nssm set GoldSignalLGBM AppEnvironmentExtra "MODEL_TYPE=lightgbm BOT_SLUG=lgbm-session-v1 BACKEND_URL=https://gold.yepwoo.com INTERNAL_SIGNAL_SECRET=<secret> MT5_TERMINAL_ID=<lgbm_terminal_id> SIGNAL_FILE_PATH=mt5_ea\signal_lgbm.txt"
nssm start GoldSignalLGBM
```

---

## 8. Switching Model and Bot Stream

Set both `MODEL_TYPE` (which ML code runs) and `BOT_SLUG` (which backend stream receives signals):

```bat
set MODEL_TYPE=xgboost
set BOT_SLUG=xgboost-v2
python scripts\07_trading_logic.py
```

Add `xgboost-v2` to [`backend/bot_defaults.py`](backend/bot_defaults.py) (or insert into `bots` in the DB) before posting.

`config.yaml` `model.active` still defaults `MODEL_TYPE` when the env var is unset.

---

## 9. Backtesting

Run a backtest after training to evaluate each model before going live.

**XGBoost:**

```bat
python scripts\08_backtester.py
type backtest_reports\last_year_results.txt
```

**LightGBM:**

```bat
python scripts\11_backtest_lightgbm.py
type backtest_reports\lgbm_last_year_results.txt
```

If a backtest produces no trades, the confidence threshold is too high. Lower it in `config.yaml`:

```yaml
# XGBoost threshold
models:
  ensemble:
    confidence_threshold: 0.60   # lower = more trades

# LightGBM threshold
lightgbm:
  confidence_threshold: 0.60
```

---

## 10. Monitoring Dashboard

```bat
streamlit run dashboard\app.py
```

Open `http://localhost:8501` to see equity curves, latest signals, and trade history.

---

## 11. Project Structure

```
gold_trading_system/
│
├── config.yaml                       # Central config — edit this first
├── run_pipeline.py                   # XGBoost pipeline orchestrator
├── requirements.txt
│
├── scripts/
│   ├── 01_data_collection.py         # Downloads OHLCV data → SQLite
│   ├── 02_feature_engineering.py     # XGBoost features → features_target_m5.parquet
│   ├── 02b_feature_engineering_lgbm.py  # LightGBM features → features_lgbm_m5.parquet
│   ├── 03_train_xgboost.py           # Trains XGBoost → models/xgboost/xgboost_best.pkl
│   ├── 07_trading_logic.py           # Signal generator (MODEL_TYPE + BOT_SLUG → /internal/signal/{slug})
│   ├── 08_backtester.py              # XGBoost backtest
│   ├── 10_train_lightgbm.py          # Trains LightGBM → models/lightgbm/lgbm_best.pkl
│   ├── 11_backtest_lightgbm.py       # LightGBM backtest
│   └── utils.py
│
├── backend/
│   ├── main.py                       # FastAPI app + startup migration + seed bots
│   ├── config.py                     # Reads from .env (optional ENV_FILE)
│   ├── models.py                     # User, Bot, Signal, …
│   ├── bot_defaults.py               # Default bot rows + seed_default_bots()
│   ├── database.py
│   ├── auth.py
│   ├── create_admin.py               # Admin user + seed bots
│   ├── routers/
│   │   ├── user.py
│   │   ├── payments.py
│   │   └── signals.py                # /api/signal, /api/signal/{slug}, /internal/signal/{slug}
│   ├── templates/
│   ├── static/
│   ├── requirements.txt
│   └── .env.example
│
├── mt5_ea/
│   ├── gold_trading_ea.mq5           # MetaTrader 5 Expert Advisor
│   ├── signal_reader.mqh
│   ├── signal.txt                    # XGBoost signal file
│   └── signal_lgbm.txt              # LightGBM signal file (created at runtime)
│
├── models/
│   ├── xgboost/xgboost_best.pkl
│   ├── lightgbm/lgbm_best.pkl
│   └── scalers/
│
├── data/
│   ├── gold_trading.db               # OHLCV + news data
│   └── processed/
│       ├── features_target_m5.parquet   # XGBoost features
│       └── features_lgbm_m5.parquet     # LightGBM features
│
├── backtest_reports/
│   ├── last_year_results.txt         # XGBoost backtest
│   └── lgbm_last_year_results.txt    # LightGBM backtest
│
├── dashboard/app.py                  # Streamlit monitoring dashboard
│
├── deploy.bat                        # Copies EA to MT5 folders
├── start_xgboost_signal.bat        # XGBoost stream (BOT_SLUG=xgboost-v1)
├── start_signal_generator.bat      # Deprecated alias → calls start_xgboost_signal.bat
├── start_lgbm_signal.bat             # LightGBM stream (BOT_SLUG=lgbm-session-v1)
└── setup_autostart.bat
```

---

## 12. Signal File Format

Both signal generators write the same JSON format:

```json
{
  "signal": "buy",
  "confidence": 0.81,
  "sl": 3180.50,
  "tp": 3250.00,
  "reason": "LightGBM (buy, conf=0.81)",
  "timestamp": "2026-04-28T09:00:00.000000"
}
```

| Field | Values | Notes |
|---|---|---|
| `signal` | `"buy"` / `"sell"` / `"hold"` | |
| `confidence` | 0.0 – 1.0 | EA ignores signals below its configured threshold |
| `sl` | price | Absolute stop-loss level |
| `tp` | price | Absolute take-profit level |

---

## 13. Environment Variable Reference

All env vars can be set in the shell before running any script. They always override `config.yaml`.

| Variable | Used by | Description |
|---|---|---|
| `MODEL_TYPE` | `07_trading_logic.py` | `xgboost` or `lightgbm` |
| `BOT_SLUG` | `07_trading_logic.py` | Backend bot id / URL slug (`xgboost-v1`, `lgbm-session-v1`, …); defaults from `MODEL_TYPE` |
| `BACKEND_URL` | `07_trading_logic.py` | Overrides `backend.url` in `config.yaml` |
| `INTERNAL_SIGNAL_SECRET` | `07_trading_logic.py` | Overrides `backend.internal_signal_secret` |
| `MT5_TERMINAL_ID` | `07_trading_logic.py` | Overrides `trading.mt5_terminal_id` in `config.yaml` |
| `SIGNAL_FILE_PATH` | `07_trading_logic.py` | Path to local signal file (default: `mt5_ea/signal.txt`) |
| `ENV_FILE` | `backend/config.py` | Optional alternate path to load `.env` |

---

## Quick Reference — Starting Everything

```bat
:: Backend (single instance, port 8000)
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

:: Signal generators (one window each; different BOT_SLUG / MT5_TERMINAL_ID)
start_xgboost_signal.bat
start_lgbm_signal.bat

:: Caddy
C:\caddy\caddy.exe run --config C:\caddy\Caddyfile
```
