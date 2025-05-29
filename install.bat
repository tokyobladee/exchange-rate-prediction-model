@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

echo ================================================================
echo CURRENCY PREDICTION SYSTEM INSTALLATION
echo ================================================================
echo.

echo [1/6] Checking Python...
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

echo [2/6] Checking pip...
python -m pip --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: pip not found!
    echo Install pip: python -m ensurepip --upgrade
    pause
    exit /b 1
)

echo [3/6] Upgrading pip...
python -m pip install --upgrade pip

echo [4/6] Installing dependencies...
echo This may take several minutes...
python -m pip install -r requirements.txt

if errorlevel 1 (
    echo ERROR: Failed to install dependencies!
    echo Try installing manually:
    echo pip install pandas numpy matplotlib seaborn scikit-learn
    pause
    exit /b 1
)

echo [5/6] Verifying installation...
python -c "import pandas, numpy, sklearn, matplotlib, seaborn, yfinance, plotly, tensorflow, torch; print('Core libraries installed successfully')" 2>nul
if errorlevel 1 (
    echo WARNING: Some libraries may not work correctly
    echo Run system test for detailed diagnostics
) else (
    echo All libraries installed successfully!
)

echo [6/6] Creating directories...
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
echo To run the system use:
echo   python launch.py
echo.
echo Quick start:
echo   python launch.py (menu selection)
echo.
echo System test:
echo   python test_system.py
echo.
echo ================================================================
pause 