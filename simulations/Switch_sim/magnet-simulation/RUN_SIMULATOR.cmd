@echo off
setlocal
cd /d "%~dp0"
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not available. Install Python 3.10 or newer and enable its PATH option.
    pause
    exit /b 1
)
python -c "import numpy, scipy, matplotlib" >nul 2>&1
if errorlevel 1 (
    echo Install the required libraries first:
    echo python -m pip install -r requirements.txt
    pause
    exit /b 1
)
python "%~dp0magnet_pulse_sim.py"
if errorlevel 1 (
    echo The simulator reported an error. See the message above.
    pause
    exit /b 1
)
endlocal
