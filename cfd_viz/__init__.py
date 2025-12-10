"""
CFD Visualization Library
=========================

A Python library for visualizing and analyzing CFD simulation results.

Modules:
    common: Configuration and VTK file I/O
    fields: Pure computation functions for flow field quantities
    analysis: Analysis tools (vorticity, line profiles, comparisons)
    animation: Animation creation utilities
    interactive: Interactive dashboard tools

Quick Start:
    >>> from cfd_viz.common import read_vtk_file
    >>> from cfd_viz.fields import magnitude, vorticity
    >>>
    >>> # Load data
    >>> data = read_vtk_file("simulation.vtk")
    >>>
    >>> # Compute derived fields
    >>> speed = magnitude(data.u, data.v)
    >>> omega = vorticity(data.u, data.v, data.dx, data.dy)

For more examples, see the examples/ directory.
"""

__version__ = "0.1.0"

# Re-export commonly used items for convenience
# Import fields module for easy access
from . import fields
from .common import VTKData, ensure_dirs, find_vtk_files, read_vtk_file

__all__ = [
    "__version__",
    # I/O
    "VTKData",
    "read_vtk_file",
    "find_vtk_files",
    "ensure_dirs",
    # Submodules
    "fields",
]
