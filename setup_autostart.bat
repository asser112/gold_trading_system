@echo off
echo ============================================
echo Gold Trading System - Auto-Start Setup
echo ============================================
echo.
echo This will add the signal generator to Windows Startup
echo so it runs automatically when you log in.
echo.

set "PROJECT_DIR=%~dp0"
set "TARGET=%PROJECT_DIR%start_signal_generator.bat"
set "STARTUP=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup"

if not exist "%TARGET%" (
    echo ERROR: start_signal_generator.bat not found at:
    echo %TARGET%
    pause
    exit /b 1
)

echo Creating startup shortcut...
powershell -Command "$WshShell = New-Object -ComObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('%STARTUP%\GoldTrading.lnk'); $Shortcut.TargetPath = '%TARGET%'; $Shortcut.WorkingDirectory = '%PROJECT_DIR%'; $Shortcut.Description = 'Gold Trading Signal Generator'; $Shortcut.Save(); Write-Output 'Shortcut created'"

if %ERRORLEVEL% EQU 0 (
    echo [OK] Auto-start configured. Signal generator will start on next login.
) else (
    echo [ERROR] Failed to create startup shortcut.
    echo Manually place a shortcut to %TARGET% in:
    echo %STARTUP%
)

echo.
pause
