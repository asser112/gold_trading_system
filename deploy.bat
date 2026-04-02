@echo off
setlocal enabledelayedexpansion

echo ============================================
echo Gold Trading System - Deployment Script
echo ============================================
echo.

set "SRC=%~dp0"
set "MT5_TERM=C:\Users\Ahmed\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075"
set "MT5_EA=%MT5_TERM%\MQL5\Experts"
set "MT5_FILES=%MT5_TERM%\MQL5\Files"
set "MT5_INCLUDE=%MT5_TERM%\MQL5\Include"
set "DEST=C:\Users\Ahmed\Desktop\gold_trading_system"

echo [1/5] Verifying directories...
if not exist "%MT5_EA%" (
    echo [ERROR] MT5 Experts folder not found: %MT5_EA%
    echo Make sure MetaTrader 5 is installed.
    pause
    exit /b 1
)
echo [OK] MT5 found.
echo.

echo [2/5] Copying EA to MT5 Experts folder...
copy /Y "%DEST%\mt5_ea\gold_trading_ea.mq5" "%MT5_EA%\gold_trading_ea.mq5" >nul 2>&1
if exist "%MT5_EA%\gold_trading_ea.mq5" (
    echo [OK] EA installed to: %MT5_EA%\gold_trading_ea.mq5
) else (
    echo [ERROR] Failed to copy EA
    pause
    exit /b 1
)
echo.

echo [3/5] Copying signal_reader.mqh to MT5 Include folder...
if exist "%DEST%\mt5_ea\signal_reader.mqh" (
    copy /Y "%DEST%\mt5_ea\signal_reader.mqh" "%MT5_INCLUDE%\signal_reader.mqh" >nul 2>&1
    echo [OK] signal_reader.mqh installed.
) else (
    echo [WARN] signal_reader.mqh not found, skipping.
)
echo.

echo [4/5] Setting up signal file in MQL5\Files...
if not exist "%MT5_FILES%" mkdir "%MT5_FILES%"
copy /Y "%DEST%\mt5_ea\signal.txt" "%MT5_FILES%\signal.txt" >nul 2>&1
if exist "%MT5_FILES%\signal.txt" (
    echo [OK] Signal file ready in MQL5\Files.
) else (
    echo [WARN] Could not copy signal.txt. The signal generator will create it.
)
echo.

echo [5/5] Verifying signal generator...
if exist "%DEST%\scripts\07_trading_logic.py" (
    echo [OK] Signal generator found: %DEST%\scripts\07_trading_logic.py
) else (
    echo [ERROR] Signal generator not found!
    pause
    exit /b 1
)
echo.

echo ============================================
echo Deployment Complete!
echo ============================================
echo.
echo NEXT STEPS:
echo.
echo 1. COMPILE THE EA:
echo    - Open MT5, press F4 for MetaEditor
echo    - Open gold_trading_ea.mq5 from Experts
echo    - Press F7 to compile
echo.
echo 2. ATTACH EA TO CHART:
echo    - Open XAUUSDr chart in MT5
echo    - Set timeframe to M5
echo    - Drag gold_trading_ea onto the chart
echo    - In Inputs tab, verify:
echo      SignalFileName = signal.txt
echo      MagicNumber = 123456
echo      LotSize = 0.01 (or 0.03)
echo      RiskPercent = 0
echo      MaxSpread = 30
echo.
echo 3. ENABLE AUTO-TRADING:
echo    - Click the green button in MT5 toolbar
echo.
echo 4. START SIGNAL GENERATOR:
echo    - Double-click: %DEST%\start_trading.bat
echo.
pause
