@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

echo ================================================================
echo CURRENCY PREDICTION SYSTEM INSTALLATION
echo ================================================================
echo.

echo [1/8] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found!
    echo Download Python from https://python.org/downloads/
    echo Make sure Python is added to PATH
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Python version: %PYTHON_VERSION%

echo [2/8] Checking pip...
python -m pip --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: pip not found!
    echo Install pip: python -m ensurepip --upgrade
    pause
    exit /b 1
)

echo [3/8] Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    echo Virtual environment created
) else (
    echo Virtual environment already exists
)

echo [4/8] Activating virtual environment...
call venv\Scripts\activate.bat

echo [5/8] Upgrading pip...
python -m pip install --upgrade pip

echo [6/8] Installing dependencies...
echo This may take several minutes...
python -m pip install -r requirements.txt

if errorlevel 1 (
    echo ERROR: Failed to install dependencies!
    echo Try installing manually:
    echo pip install pandas numpy matplotlib seaborn scikit-learn
    pause
    exit /b 1
)

echo [7/8] Verifying installation...
python -c "import pandas, numpy, sklearn, matplotlib, seaborn, yfinance, plotly, tensorflow, torch; print('Core libraries installed successfully')" 2>nul
if errorlevel 1 (
    echo WARNING: Some libraries may not work correctly
    echo Run system test for detailed diagnostics
) else (
    echo All libraries installed successfully!
)

echo [8/8] Creating directories...
if not exist "data" mkdir data
if not exist "models" mkdir models
if not exist "plots" mkdir plots
if not exist "reports" mkdir reports
if not exist "predictions" mkdir predictions

echo.
echo ================================================================
echo INSTALLATION COMPLETED!
echo ================================================================
echo.
echo To run the system:
echo   1. Activate virtual environment: venv\Scripts\activate.bat
echo   2. Launch system: python launch.py
echo.
echo Quick start (with activated venv):
echo   python launch.py
echo.
echo System test:
echo   python test_system.py
echo.
echo Deactivate virtual environment:
echo   deactivate
echo.
echo ================================================================
pause 