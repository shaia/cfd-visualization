"""Plotting subpackage for CFD visualization.

This package provides reusable plotting functions that take dataclasses
from the analysis module and produce matplotlib visualizations.

All functions follow a consistent pattern:
- Take analysis dataclasses as input
- Accept an optional axes parameter (create new if not provided)
- Return the axes object for further customization

Submodules:
    fields: Functions for plotting scalar and vector fields.
    line_plots: Functions for plotting line profiles and cross-sections.
    analysis: Functions for plotting analysis results (comparison, BL, etc.).
    time_series: Functions for plotting time series and convergence data.

Example:
    >>> from cfd_viz.analysis import extract_line_profile, detect_wake_regions
    >>> from cfd_viz.plotting import plot_line_profile, plot_wake_region
    >>>
    >>> # Extract data using analysis functions
    >>> profile = extract_line_profile(u, v, x, y, (0, 0.5), (1, 0.5))
    >>> wake = detect_wake_regions(u, v)
    >>>
    >>> # Plot using plotting functions
    >>> fig, (ax1, ax2) = plt.subplots(1, 2)
    >>> plot_line_profile(profile, ax=ax1)
    >>> plot_wake_region(wake, X, Y, velocity_mag, ax=ax2)
    >>> plt.show()
"""

from .analysis import (
    plot_case_comparison,
    plot_field_difference,
    plot_flow_statistics,
    plot_spatial_fluctuations,
    plot_wake_region,
)
from .fields import (
    plot_contour_field,
    plot_pressure_field,
    plot_streamlines,
    plot_vector_field,
    plot_velocity_field,
    plot_vorticity_field,
)
from .line_plots import (
    plot_boundary_layer_profile,
    plot_boundary_layer_profiles,
    plot_cross_sectional_averages,
    plot_line_profile,
    plot_multiple_profiles,
    plot_velocity_profiles,
)
from .time_series import (
    plot_convergence_history,
    plot_metric_time_series,
    plot_monitoring_dashboard,
    plot_statistics_panel,
)

__all__ = [
    # Fields
    "plot_contour_field",
    "plot_velocity_field",
    "plot_pressure_field",
    "plot_vorticity_field",
    "plot_vector_field",
    "plot_streamlines",
    # Line Plots
    "plot_line_profile",
    "plot_multiple_profiles",
    "plot_velocity_profiles",
    "plot_boundary_layer_profile",
    "plot_boundary_layer_profiles",
    "plot_cross_sectional_averages",
    # Analysis
    "plot_field_difference",
    "plot_case_comparison",
    "plot_wake_region",
    "plot_spatial_fluctuations",
    "plot_flow_statistics",
    # Time Series
    "plot_metric_time_series",
    "plot_convergence_history",
    "plot_monitoring_dashboard",
    "plot_statistics_panel",
]
