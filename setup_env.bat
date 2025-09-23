@echo off
REM Windows batch script to set up the CFD visualization conda environment

echo Setting up CFD Visualization conda environment...

REM Check if conda is available
conda --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Conda is not installed or not in PATH
    echo Please install Anaconda or Miniconda first
    pause
    exit /b 1
)

REM Create conda environment from environment.yml
echo Creating conda environment from environment.yml...
conda env create -f environment.yml

if %errorlevel% neq 0 (
    echo Error: Failed to create conda environment
    pause
    exit /b 1
)

echo.
echo Environment created successfully!
echo.
echo To activate the environment, run:
echo   conda activate cfd-visualization
echo.
echo To install this package in development mode, run:
echo   conda activate cfd-visualization
echo   pip install -e .
echo.
pause