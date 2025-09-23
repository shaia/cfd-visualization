"""
CFD Visualization Framework
============================

A comprehensive visualization suite for Computational Fluid Dynamics analysis.

Package Structure:
    analysis/     - High-priority analysis tools for CFD flow analysis
    animation/    - Animation and flow visualization tools
    interactive/  - Interactive and advanced visualization tools
    utilities/    - Core utilities and supporting tools

Usage:
    from src.analysis import vorticity_visualizer
    from src.animation import create_cfd_animation
    from src.interactive import interactive_dashboard
    from src.utilities import visualize_cfd
"""

# Import all submodules for convenience
from . import analysis
from . import animation
from . import interactive
from . import utilities

__version__ = "0.1.0"
__author__ = "CFD Team"

__all__ = ['analysis', 'animation', 'interactive', 'utilities']