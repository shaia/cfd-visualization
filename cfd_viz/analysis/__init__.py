"""Analysis subpackage for CFD flow field analysis.

This package provides pure functions for analyzing CFD simulation results.
All functions return data structures (dataclasses) rather than producing
plots directly, enabling flexible integration with different visualization
backends.

Submodules:
    case_comparison: Functions for comparing CFD cases and parameter sweeps.
    line_extraction: Functions for extracting line profiles and cross-sections.
    boundary_layer: Functions for boundary layer analysis.
    time_series: Functions for temporal analysis and convergence monitoring.
    flow_features: Functions for detecting wake regions and flow features.

Example:
    >>> from cfd_viz.analysis import compare_fields, extract_line_profile
    >>> from cfd_viz.analysis import analyze_boundary_layer
    >>>
    >>> # Compare two simulation cases
    >>> comparison = compare_fields(u1, v1, p1, u2, v2, p2, dx, dy)
    >>> print(f"Velocity RMS difference: {comparison.velocity_diff.rms_diff}")
    >>>
    >>> # Extract centerline profile
    >>> profile = extract_line_profile(u, v, x, y, (0, 0.5), (1, 0.5))
    >>> print(f"Max velocity: {profile.max_velocity}")
    >>>
    >>> # Analyze boundary layer
    >>> bl = analyze_boundary_layer(u, v, x, y, wall_y=0, x_location=0.5)
    >>> print(f"BL thickness: {bl.delta_99}")
"""

# Case comparison functions and classes
# Boundary layer analysis functions and classes
from .boundary_layer import (
    BoundaryLayerDevelopment,
    BoundaryLayerProfile,
    WallShearStress,
    analyze_boundary_layer,
    analyze_boundary_layer_development,
    blasius_solution,
    compute_displacement_thickness,
    compute_momentum_thickness,
    compute_wall_shear,
    compute_wall_shear_distribution,
    find_boundary_layer_edge,
)
from .case_comparison import (
    CaseComparison,
    FieldDifference,
    ParameterSweepResult,
    compare_fields,
    compute_convergence_metrics,
    compute_error_norms,
    compute_field_difference,
    parameter_sweep_analysis,
)

# Flow features detection functions and classes
from .flow_features import (
    CrossSectionalAverages,
    RecirculationZone,
    SpatialFluctuations,
    WakeRegion,
    compute_adverse_pressure_gradient,
    compute_cross_sectional_averages,
    compute_pressure_gradient,
    compute_spatial_fluctuations,
    detect_recirculation_zones,
    detect_wake_regions,
)

# Line extraction functions and classes
from .line_extraction import (
    CrossSection,
    LineProfile,
    MultipleProfiles,
    compute_centerline_profiles,
    compute_mass_flow_rate,
    compute_momentum_flux,
    compute_profile_statistics,
    extract_horizontal_profile,
    extract_line_profile,
    extract_multiple_profiles,
    extract_vertical_profile,
)

# Time series analysis functions and classes
from .time_series import (
    ConvergenceMetrics,
    FlowHistoryStatistics,
    MonitoringHistory,
    MonitoringSnapshot,
    ProbeTimeSeries,
    TemporalStatistics,
    analyze_convergence,
    analyze_flow_history,
    compute_monitoring_snapshot,
    compute_rms_fluctuations,
    compute_running_average,
    compute_temporal_statistics,
    compute_time_averaged_field,
    create_monitoring_history,
    detect_periodicity,
    extract_probe_time_series,
)

__all__ = [
    # Case Comparison
    "CaseComparison",
    "FieldDifference",
    "ParameterSweepResult",
    "compare_fields",
    "compute_field_difference",
    "parameter_sweep_analysis",
    "compute_convergence_metrics",
    "compute_error_norms",
    # Line Extraction
    "LineProfile",
    "CrossSection",
    "MultipleProfiles",
    "extract_line_profile",
    "extract_vertical_profile",
    "extract_horizontal_profile",
    "extract_multiple_profiles",
    "compute_centerline_profiles",
    "compute_profile_statistics",
    "compute_mass_flow_rate",
    "compute_momentum_flux",
    # Boundary Layer
    "BoundaryLayerProfile",
    "BoundaryLayerDevelopment",
    "WallShearStress",
    "analyze_boundary_layer",
    "analyze_boundary_layer_development",
    "compute_wall_shear_distribution",
    "find_boundary_layer_edge",
    "compute_displacement_thickness",
    "compute_momentum_thickness",
    "compute_wall_shear",
    "blasius_solution",
    # Time Series
    "TemporalStatistics",
    "FlowHistoryStatistics",
    "ConvergenceMetrics",
    "ProbeTimeSeries",
    "MonitoringSnapshot",
    "MonitoringHistory",
    "compute_temporal_statistics",
    "analyze_flow_history",
    "compute_time_averaged_field",
    "compute_rms_fluctuations",
    "extract_probe_time_series",
    "analyze_convergence",
    "compute_running_average",
    "detect_periodicity",
    "compute_monitoring_snapshot",
    "create_monitoring_history",
    # Flow Features
    "WakeRegion",
    "SpatialFluctuations",
    "CrossSectionalAverages",
    "RecirculationZone",
    "detect_wake_regions",
    "compute_spatial_fluctuations",
    "compute_cross_sectional_averages",
    "detect_recirculation_zones",
    "compute_pressure_gradient",
    "compute_adverse_pressure_gradient",
]
