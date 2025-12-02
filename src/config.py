"""
CFD Visualization Configuration

Centralized configuration for all visualization scripts.
All scripts should import paths from this module.
"""

import os
from pathlib import Path

# Project root directory (cfd-visualization/)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Default data directory for VTK input files
# Can be overridden via CFD_VIZ_DATA_DIR environment variable
DATA_DIR = Path(os.environ.get('CFD_VIZ_DATA_DIR', PROJECT_ROOT / 'data' / 'vtk_files'))

# Output directory for visualizations
# Can be overridden via CFD_VIZ_OUTPUT_DIR environment variable
OUTPUT_DIR = Path(os.environ.get('CFD_VIZ_OUTPUT_DIR', PROJECT_ROOT / 'output'))

# Subdirectories for different output types
ANIMATIONS_DIR = OUTPUT_DIR / 'animations'
PLOTS_DIR = OUTPUT_DIR / 'plots'
HTML_DIR = OUTPUT_DIR / 'html'

# Default VTK file pattern
DEFAULT_VTK_PATTERN = '*.vtk'


def ensure_dirs():
    """Create output directories if they don't exist."""
    for d in [DATA_DIR, OUTPUT_DIR, ANIMATIONS_DIR, PLOTS_DIR, HTML_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def find_vtk_files(pattern=None, directory=None):
    """
    Find VTK files matching the given pattern.

    Args:
        pattern: Glob pattern for VTK files (default: *.vtk)
        directory: Directory to search (default: DATA_DIR)

    Returns:
        List of Path objects for matching files, sorted by name
    """
    search_dir = Path(directory) if directory else DATA_DIR
    search_pattern = pattern or DEFAULT_VTK_PATTERN

    if not search_dir.exists():
        return []

    return sorted(search_dir.glob(search_pattern))


def get_data_dir():
    """Get the data directory, creating it if needed."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def get_output_dir():
    """Get the output directory, creating it if needed."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR
