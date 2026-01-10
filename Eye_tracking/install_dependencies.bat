@echo off
echo ========================================
echo Eye Tracking App - Installer (Python 3.11)
echo ========================================
echo.
echo Detected: Python 3.11
echo Installing standard packages...
echo.

REM Upgrade pip
py -3.11 -m pip install --upgrade pip

REM Install dependencies
echo.
echo Installing opencv, mediapipe, pandas, openpyxl, numpy...
py -3.11 -m pip install opencv-python mediapipe pandas openpyxl numpy

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo Installation successful!
    echo ========================================
    echo.
    echo You can now run: run_eye_tracking.bat
    echo.
) else (
    echo.
    echo ========================================
    echo Installation failed!
    echo ========================================
    echo.
    echo Please make sure you have internet connection.
    echo.
)

pause
