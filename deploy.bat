@echo off
setlocal enabledelayedexpansion

echo ============================================
echo Gold Trading System - Deployment Script
echo ============================================
echo.

:: Project root is the folder containing this script
set "SRC=%~dp0"

:: ---------------------------------------------------------------
:: CONFIGURE: Paste your MT5 terminal data folder ID below.
:: Find it: open MT5 > Help > About, or browse:
::   %APPDATA%\MetaQuotes\Terminal\   <- look for a long hex folder
:: ---------------------------------------------------------------
set "MT5_TERMINAL_ID=YOUR_TERMINAL_ID"

if "%MT5_TERMINAL_ID%"=="YOUR_TERMINAL_ID" (
    echo [ERROR] MT5_TERMINAL_ID is not configured.
    echo.
    echo Open this file ^(deploy.bat^) and set MT5_TERMINAL_ID to your
    echo terminal folder ID. You can find it by browsing:
    echo   %APPDATA%\MetaQuotes\Terminal\
    echo The ID is the long hex-named folder inside.
    echo.
    pause
    exit /b 1
)

set "MT5_TERM=%APPDATA%\MetaQuotes\Terminal\%MT5_TERMINAL_ID%"
set "MT5_EA=%MT5_TERM%\MQL5\Experts"
set "MT5_FILES=%MT5_TERM%\MQL5\Files"
set "MT5_INCLUDE=%MT5_TERM%\MQL5\Include"

echo [1/5] Verifying MT5 directories...
if not exist "%MT5_EA%" (
    echo [ERROR] MT5 Experts folder not found: %MT5_EA%
    echo Make sure MetaTrader 5 is installed and has been opened at least once.
    pause
    exit /b 1
)
echo [OK] MT5 found at: %MT5_TERM%
echo.

echo [2/5] Copying EA to MT5 Experts folder...
copy /Y "%SRC%mt5_ea\gold_trading_ea.mq5" "%MT5_EA%\gold_trading_ea.mq5" >nul 2>&1
if exist "%MT5_EA%\gold_trading_ea.mq5" (
    echo [OK] EA installed to: %MT5_EA%\gold_trading_ea.mq5
) else (
    echo [ERROR] Failed to copy EA. Check folder permissions.
    pause
    exit /b 1
)
echo.

echo [3/5] Copying signal_reader.mqh to MT5 Include folder...
if exist "%SRC%mt5_ea\signal_reader.mqh" (
    copy /Y "%SRC%mt5_ea\signal_reader.mqh" "%MT5_INCLUDE%\signal_reader.mqh" >nul 2>&1
    echo [OK] signal_reader.mqh installed.
) else (
    echo [WARN] signal_reader.mqh not found, skipping.
)
echo.

echo [4/5] Setting up signal file in MQL5\Files...
if not exist "%MT5_FILES%" mkdir "%MT5_FILES%"
copy /Y "%SRC%mt5_ea\signal.txt" "%MT5_FILES%\signal.txt" >nul 2>&1
if exist "%MT5_FILES%\signal.txt" (
    echo [OK] Signal file ready in MQL5\Files.
) else (
    echo [WARN] Could not copy signal.txt — the signal generator will create it.
)
echo.

echo [5/5] Verifying signal generator...
if exist "%SRC%scripts\07_trading_logic.py" (
    echo [OK] Signal generator found.
) else (
    echo [ERROR] Signal generator not found: scripts\07_trading_logic.py
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
echo    - Open MT5, press F4 to open MetaEditor
echo    - Open Experts\gold_trading_ea.mq5
echo    - Press F7 to compile (no errors expected)
echo.
echo 2. ATTACH EA TO CHART:
echo    - Open an XAUUSDr chart in MT5 (or XAUUSD)
echo    - Set timeframe to M5
echo    - Drag gold_trading_ea onto the chart
echo    - In the Inputs tab verify:
echo        SignalFileName = signal.txt
echo        MagicNumber   = 123456
echo        LotSize       = 0.03
echo        MaxSpread     = 30
echo    - Check "Allow live trading" and "Allow DLL imports"
echo    - Click OK
echo.
echo 3. ENABLE AUTO-TRADING:
echo    - Click the green Auto Trading button in the MT5 toolbar
echo.
echo 4. START SIGNAL GENERATOR:
echo    - Double-click: start_trading.bat
echo.
echo 5. (OPTIONAL) AUTO-START ON BOOT:
echo    - Double-click: setup_autostart.bat
echo.
pause
