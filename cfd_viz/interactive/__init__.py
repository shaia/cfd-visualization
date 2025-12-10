"""Interactive visualization subpackage for CFD data.

This package provides pure functions for creating interactive Plotly
visualizations from CFD data. Functions accept numpy arrays and return
Plotly Figure objects.

Submodules:
    plotly: Functions for creating Plotly figures (heatmaps, contours, etc.)

Example:
    >>> from cfd_viz.interactive import (
    ...     create_interactive_frame,
    ...     create_heatmap_figure,
    ...     create_dashboard_figure,
    ... )
    >>>
    >>> # Create frame from numpy arrays
    >>> frame = create_interactive_frame(x, y, u, v, p=p, time_index=0)
    >>>
    >>> # Create interactive figures
    >>> fig = create_heatmap_figure(x, y, velocity_mag, title="Velocity")
    >>> fig.show()
    >>>
    >>> # Create multi-panel dashboard
    >>> dashboard = create_dashboard_figure(frame)
    >>> dashboard.write_html("dashboard.html")
"""

from .plotly import (
    InteractiveFrameCollection,
    InteractiveFrameData,
    create_animated_dashboard,
    create_contour_figure,
    create_convergence_figure,
    create_dashboard_figure,
    create_heatmap_figure,
    create_interactive_frame,
    create_interactive_frame_collection,
    create_surface_figure,
    create_time_series_figure,
    create_vector_figure,
)

__all__ = [
    # Dataclasses
    "InteractiveFrameData",
    "InteractiveFrameCollection",
    # Frame creation
    "create_interactive_frame",
    "create_interactive_frame_collection",
    # Single-panel figures
    "create_heatmap_figure",
    "create_contour_figure",
    "create_vector_figure",
    "create_surface_figure",
    # Time series
    "create_time_series_figure",
    "create_convergence_figure",
    # Dashboards
    "create_dashboard_figure",
    "create_animated_dashboard",
]
