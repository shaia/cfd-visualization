#!/bin/bash
# Shell script to set up the CFD visualization virtual environment using uv

echo "Setting up CFD Visualization virtual environment..."

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed or not in PATH"
    echo "Please install uv first: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment with uv..."
uv venv .venv

if [ $? -ne 0 ]; then
    echo "Error: Failed to create virtual environment"
    exit 1
fi

echo ""
echo "Virtual environment created successfully!"
echo ""
echo "To activate the environment, run:"
echo "  source .venv/bin/activate"
echo ""
echo "To install this package in development mode, run:"
echo "  source .venv/bin/activate"
echo "  uv pip install -e ."
echo ""
