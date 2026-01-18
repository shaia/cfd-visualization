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
    cfd_python_integration: Optional cfd-python integration utilities

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

cfd-python Integration:
    When cfd-python is installed, additional features are available:

    >>> from cfd_viz.cfd_python_integration import has_cfd_python, get_cfd_python
    >>> if has_cfd_python():
    ...     cfd = get_cfd_python()
    ...     result = cfd.run_simulation_with_params(nx=50, ny=50, steps=100)

    Install with: pip install cfd-visualization[simulation]

For more examples, see the examples/ directory.
"""

__version__ = "0.1.0"

# Re-export commonly used items for convenience
# Import fields module for easy access
from . import fields
from .cfd_python_integration import (
    check_cfd_python_version,
    get_cfd_python,
    get_cfd_python_version,
    has_cfd_python,
    require_cfd_python,
    require_cfd_python_version,
)
from .common import VTKData, ensure_dirs, find_vtk_files, read_vtk_file
from .convert import from_cfd_python, from_simulation_result, to_cfd_python
from .info import get_recommended_settings, get_system_info, print_system_info
from .quick import quick_plot, quick_plot_data, quick_plot_result
from .stats import (
    calculate_field_stats,
    compute_flow_statistics,
    compute_velocity_magnitude,
)

__all__ = [
    "__version__",
    # I/O
    "VTKData",
    "read_vtk_file",
    "find_vtk_files",
    "ensure_dirs",
    # Submodules
    "fields",
    # cfd-python integration
    "has_cfd_python",
    "get_cfd_python",
    "get_cfd_python_version",
    "require_cfd_python",
    "check_cfd_python_version",
    "require_cfd_python_version",
    # Conversion utilities
    "from_cfd_python",
    "from_simulation_result",
    "to_cfd_python",
    # Statistics
    "calculate_field_stats",
    "compute_flow_statistics",
    "compute_velocity_magnitude",
    # Quick plotting
    "quick_plot",
    "quick_plot_result",
    "quick_plot_data",
    # System info
    "get_system_info",
    "get_recommended_settings",
    "print_system_info",
]
