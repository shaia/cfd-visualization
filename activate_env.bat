@echo off
REM Windows batch script to activate the CFD visualization conda environment

echo Activating CFD Visualization conda environment...

REM Check if conda is available
conda --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Conda is not installed or not in PATH
    echo Please install Anaconda or Miniconda first
    pause
    exit /b 1
)

REM Activate the environment
conda activate cfd-visualization

if %errorlevel% neq 0 (
    echo Error: Failed to activate environment 'cfd-visualization'
    echo Make sure the environment exists. Run setup_env.bat first if needed.
    pause
    exit /b 1
)

echo Environment activated successfully!
echo You can now run Python scripts with all dependencies available.
echo.

REM Keep the command prompt open with the environment activated
cmd /k