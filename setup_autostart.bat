@echo off
echo ============================================
echo Gold Trading System - Auto-Start Setup
echo ============================================
echo.
echo This will add the signal generator to Windows Startup
echo so it runs automatically when you log in.
echo.

set "STARTUP=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup"
set "TARGET=C:\Users\Ahmed\Desktop\gold_trading_system\start_signal_generator.bat"

if not exist "%TARGET%" (
    echo ERROR: start_signal_generator.bat not found at:
    echo %TARGET%
    echo.
    echo Please run deploy.bat first.
    pause
    exit /b 1
)

echo Creating startup shortcut...
powershell -Command "$WshShell = New-Object -ComObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('%STARTUP%\GoldTrading.lnk'); $Shortcut.TargetPath = '%TARGET%'; $Shortcut.WorkingDirectory = 'C:\Users\Ahmed\Desktop\gold_trading_system'; $Shortcut.Description = 'Gold Trading Signal Generator'; $Shortcut.Save(); Write-Output 'Shortcut created'"

if %ERRORLEVEL% EQU 0 (
    echo [OK] Auto-start configured successfully
    echo The signal generator will start when Windows boots.
) else (
    echo [ERROR] Failed to create startup shortcut
    echo You can manually create one by placing a shortcut to:
    echo %TARGET%
    echo in: %STARTUP%
)

echo.
pause
