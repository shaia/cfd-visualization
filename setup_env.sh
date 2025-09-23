#!/bin/bash
# Shell script to set up the CFD visualization conda environment

echo "Setting up CFD Visualization conda environment..."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: Conda is not installed or not in PATH"
    echo "Please install Anaconda or Miniconda first"
    exit 1
fi

# Create conda environment from environment.yml
echo "Creating conda environment from environment.yml..."
conda env create -f environment.yml

if [ $? -ne 0 ]; then
    echo "Error: Failed to create conda environment"
    exit 1
fi

echo ""
echo "Environment created successfully!"
echo ""
echo "To activate the environment, run:"
echo "  conda activate cfd-visualization"
echo ""
echo "To install this package in development mode, run:"
echo "  conda activate cfd-visualization"
echo "  pip install -e ."
echo ""