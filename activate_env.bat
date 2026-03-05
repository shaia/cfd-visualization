@echo off
REM Windows batch script to activate the CFD visualization virtual environment

echo Activating CFD Visualization environment...

REM Check if .venv exists
if not exist ".venv\Scripts\activate.bat" (
    echo Error: Virtual environment not found at .venv\
    echo Run 'uv venv .venv' first to create it.
    pause
    exit /b 1
)

REM Activate the environment
call .venv\Scripts\activate.bat

echo Environment activated successfully!
echo You can now run Python scripts with all dependencies available.
echo.

REM Keep the command prompt open with the environment activated
cmd /k
