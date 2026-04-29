@echo off
title LightGBM Signal Generator
cd /d "%~dp0"

REM ── Same backend as XGBoost — only BOT_SLUG differs ────────────────────────
REM BACKEND_URL / INTERNAL_SIGNAL_SECRET must match backend\.env (single server).

set MODEL_TYPE=lightgbm
set BOT_SLUG=lgbm-session-v1

REM Same domain as your FastAPI app (see BASE_URL in backend\.env)
set BACKEND_URL=https://gold.yepwoo.com

REM Must match INTERNAL_SIGNAL_SECRET in backend\.env
set INTERNAL_SIGNAL_SECRET=CHANGE_ME_same_as_backend_env

REM MT5 terminal ID for the LightGBM demo account.
REM Find it: MT5 → Help → About, or %APPDATA%\MetaQuotes\Terminal\
set MT5_TERMINAL_ID=PASTE_YOUR_LGBM_TERMINAL_ID_HERE

REM Local signal file (separate from mt5_ea\signal.txt)
set SIGNAL_FILE_PATH=mt5_ea\signal_lgbm.txt

python scripts\07_trading_logic.py
pause
