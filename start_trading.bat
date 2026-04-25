@echo off
title Gold Trading System - Signal Generator
echo ============================================
echo Gold Trading System - Signal Generator
echo ============================================
echo.
echo Symbol: XAUUSDr
echo Timeframe: M5
echo Signal File: mt5_ea\signal.txt
echo              (also mirrors to MQL5\Files)
echo.
echo Starting signal generator...
echo Press Ctrl+C to stop.
echo.

cd /d "%~dp0"
python scripts\07_trading_logic.py

pause
