@echo off
title Gold Trading Signal Generator — XGBoost (xgboost-v1)
cd /d "%~dp0"

set MODEL_TYPE=xgboost
set BOT_SLUG=xgboost-v1

python scripts\07_trading_logic.py
