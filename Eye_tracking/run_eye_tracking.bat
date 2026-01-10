@echo off
echo ========================================
echo   Eye Tracking Application
echo ========================================
echo.
echo Running System Check...
py -3.11 check_setup.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ========================================
    echo SYSTEM CHECK FAILED!
    echo ========================================
    echo Please run: install_dependencies.bat
    pause
    exit /b
)

echo.
echo Starting eye tracking with Python 3.11...
echo.
echo Instructions:
echo - Calibration will start in fullscreen
echo - Look at each white point and press SPACE
echo - Press START TRACKING button after calibration
echo - Tracking runs for 30 seconds
echo - Press ESC to abort anytime
echo.
echo Starting in 3 seconds...
timeout /t 3 /nobreak >nul

py -3.11 main.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ========================================
    echo Error occurred during execution!
    echo ========================================
    echo Please check the error message above.
    echo.
)

echo.
echo Closing window in 10 minutes... (Close manually if done)
timeout /t 600 >nul
