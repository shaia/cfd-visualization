"""Pure functions for comparing CFD simulation cases.

This module provides functions for comparing multiple CFD simulation results,
calculating differences, and generating comparison metrics. All functions
return data structures rather than producing plots directly.

Example:
    >>> from cfd_viz.analysis.comparison import compare_fields, CaseComparison
    >>> comparison = compare_fields(u1, v1, p1, u2, v2, p2, dx, dy)
    >>> print(f"Max velocity difference: {comparison.velocity_diff_max}")
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from cfd_viz.fields import velocity
from cfd_viz.fields.derived import FlowStatistics, calculate_flow_statistics
from cfd_viz.fields.vorticity import vorticity as compute_vorticity


@dataclass
class FieldDifference:
    """Difference between two scalar fields.

    Attributes:
        diff: Point-wise difference (field2 - field1).
        abs_diff: Absolute difference.
        max_diff: Maximum difference.
        min_diff: Minimum difference.
        mean_diff: Mean difference.
        rms_diff: Root mean square difference.
        relative_diff: Relative difference where field1 > threshold.
        max_relative_diff: Maximum relative difference.
    """

    diff: NDArray
    abs_diff: NDArray
    max_diff: float
    min_diff: float
    mean_diff: float
    rms_diff: float
    relative_diff: NDArray
    max_relative_diff: float

    def to_dict(self) -> Dict:
        """Convert to dictionary (excluding array fields for summary)."""
        return {
            "max_diff": self.max_diff,
            "min_diff": self.min_diff,
            "mean_diff": self.mean_diff,
            "rms_diff": self.rms_diff,
            "max_relative_diff": self.max_relative_diff,
        }


@dataclass
class CaseComparison:
    """Comprehensive comparison between two CFD cases.

    Attributes:
        stats1: Flow statistics for case 1.
        stats2: Flow statistics for case 2.
        velocity_diff: Velocity magnitude difference analysis.
        pressure_diff: Pressure field difference analysis (if available).
        vorticity_diff: Vorticity field difference analysis.
        u_diff: U-velocity component difference.
        v_diff: V-velocity component difference.
        metrics_comparison: Dict mapping metric names to (value1, value2, diff, pct_change).
    """

    stats1: FlowStatistics
    stats2: FlowStatistics
    velocity_diff: FieldDifference
    pressure_diff: Optional[FieldDifference]
    vorticity_diff: FieldDifference
    u_diff: FieldDifference
    v_diff: FieldDifference
    metrics_comparison: Dict[str, Tuple[float, float, float, float]]

    def summary(self) -> Dict:
        """Return summary of key comparison metrics."""
        return {
            "velocity_rms_diff": self.velocity_diff.rms_diff,
            "velocity_max_diff": self.velocity_diff.max_diff,
            "vorticity_rms_diff": self.vorticity_diff.rms_diff,
            "pressure_rms_diff": (
                self.pressure_diff.rms_diff if self.pressure_diff else None
            ),
            "metrics": self.metrics_comparison,
        }


@dataclass
class ParameterSweepResult:
    """Results from a parameter sweep analysis.

    Attributes:
        parameter_name: Name of the swept parameter.
        parameter_values: List of parameter values.
        statistics: List of FlowStatistics for each case.
        metrics: Dict mapping metric names to lists of values.
        trends: Dict mapping metric names to (slope, intercept) from linear fit.
    """

    parameter_name: str
    parameter_values: List[float]
    statistics: List[FlowStatistics]
    metrics: Dict[str, List[float]]
    trends: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    def get_metric_trend(self, metric_name: str) -> Optional[Tuple[float, float]]:
        """Get linear trend for a specific metric."""
        return self.trends.get(metric_name)


def compute_field_difference(
    field1: NDArray,
    field2: NDArray,
    threshold: float = 1e-10,
) -> FieldDifference:
    """Compute comprehensive difference between two fields.

    Args:
        field1: First field (reference).
        field2: Second field (comparison).
        threshold: Minimum value for relative difference calculation.

    Returns:
        FieldDifference with all difference metrics.
    """
    diff = field2 - field1
    abs_diff = np.abs(diff)

    # Relative difference (avoid division by zero)
    denominator = np.where(np.abs(field1) > threshold, np.abs(field1), threshold)
    relative_diff = abs_diff / denominator

    return FieldDifference(
        diff=diff,
        abs_diff=abs_diff,
        max_diff=float(np.max(diff)),
        min_diff=float(np.min(diff)),
        mean_diff=float(np.mean(diff)),
        rms_diff=float(np.sqrt(np.mean(diff**2))),
        relative_diff=relative_diff,
        max_relative_diff=float(np.max(relative_diff)),
    )


def compare_fields(
    u1: NDArray,
    v1: NDArray,
    p1: Optional[NDArray],
    u2: NDArray,
    v2: NDArray,
    p2: Optional[NDArray],
    dx: float,
    dy: float,
) -> CaseComparison:
    """Compare two CFD flow fields comprehensively.

    Args:
        u1, v1: Velocity components for case 1.
        p1: Pressure field for case 1 (optional).
        u2, v2: Velocity components for case 2.
        p2: Pressure field for case 2 (optional).
        dx, dy: Grid spacing.

    Returns:
        CaseComparison with all comparison data.

    Raises:
        ValueError: If field shapes don't match.
    """
    if u1.shape != u2.shape or v1.shape != v2.shape:
        raise ValueError(
            f"Field shapes must match: {u1.shape} vs {u2.shape}, {v1.shape} vs {v2.shape}"
        )

    # Calculate statistics for both cases
    stats1 = calculate_flow_statistics(u1, v1, p1, dx, dy)
    stats2 = calculate_flow_statistics(u2, v2, p2, dx, dy)

    # Velocity magnitude difference
    vel_mag1 = velocity.magnitude(u1, v1)
    vel_mag2 = velocity.magnitude(u2, v2)
    velocity_diff = compute_field_difference(vel_mag1, vel_mag2)

    # Component differences
    u_diff = compute_field_difference(u1, u2)
    v_diff = compute_field_difference(v1, v2)

    # Pressure difference
    pressure_diff = None
    if p1 is not None and p2 is not None:
        pressure_diff = compute_field_difference(p1, p2)

    # Vorticity difference
    omega1 = compute_vorticity(u1, v1, dx, dy)
    omega2 = compute_vorticity(u2, v2, dx, dy)
    vorticity_diff = compute_field_difference(omega1, omega2)

    # Metrics comparison
    d1 = stats1.to_dict()
    d2 = stats2.to_dict()
    metrics_comparison = {}

    for key in d1:
        val1 = d1[key]
        val2 = d2[key]
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            diff = val2 - val1
            pct_change = (diff / val1 * 100) if val1 != 0 else 0.0
            metrics_comparison[key] = (val1, val2, diff, pct_change)

    return CaseComparison(
        stats1=stats1,
        stats2=stats2,
        velocity_diff=velocity_diff,
        pressure_diff=pressure_diff,
        vorticity_diff=vorticity_diff,
        u_diff=u_diff,
        v_diff=v_diff,
        metrics_comparison=metrics_comparison,
    )


def parameter_sweep_analysis(
    cases: List[Tuple[NDArray, NDArray, Optional[NDArray]]],
    parameter_values: List[float],
    parameter_name: str,
    dx: float,
    dy: float,
) -> ParameterSweepResult:
    """Analyze results from a parameter sweep.

    Args:
        cases: List of (u, v, p) tuples for each case.
        parameter_values: Parameter value for each case.
        parameter_name: Name of the swept parameter.
        dx, dy: Grid spacing.

    Returns:
        ParameterSweepResult with statistics and trends.
    """
    if len(cases) != len(parameter_values):
        raise ValueError("Number of cases must match number of parameter values")

    # Calculate statistics for each case
    statistics = []
    for u, v, p in cases:
        stats = calculate_flow_statistics(u, v, p, dx, dy)
        statistics.append(stats)

    # Sort by parameter value
    sorted_indices = np.argsort(parameter_values)
    parameter_values = [parameter_values[i] for i in sorted_indices]
    statistics = [statistics[i] for i in sorted_indices]

    # Extract metrics
    metrics: Dict[str, List[float]] = {
        "max_velocity": [],
        "mean_velocity": [],
        "total_kinetic_energy": [],
        "max_vorticity": [],
        "mean_vorticity": [],
        "velocity_uniformity": [],
    }

    for stats in statistics:
        d = stats.to_dict()
        for key, values in metrics.items():
            if key in d:
                values.append(d[key])

    # Calculate trends (linear fit)
    trends: Dict[str, Tuple[float, float]] = {}
    if len(parameter_values) >= 2:
        param_array = np.array(parameter_values)
        for key, values in metrics.items():
            if len(values) == len(parameter_values):
                coeffs = np.polyfit(param_array, values, 1)
                trends[key] = (float(coeffs[0]), float(coeffs[1]))

    return ParameterSweepResult(
        parameter_name=parameter_name,
        parameter_values=parameter_values,
        statistics=statistics,
        metrics=metrics,
        trends=trends,
    )


def compute_convergence_metrics(
    fields_sequence: List[Tuple[NDArray, NDArray]],
) -> Dict[str, NDArray]:
    """Compute convergence metrics from a sequence of flow fields.

    Useful for analyzing iterative solver convergence or time evolution.

    Args:
        fields_sequence: List of (u, v) tuples representing time/iteration steps.

    Returns:
        Dict with:
            - 'velocity_change': Array of RMS velocity changes between steps.
            - 'max_velocity_change': Array of max velocity changes.
            - 'is_converged': Boolean indicating if last change is below threshold.
    """
    if len(fields_sequence) < 2:
        return {
            "velocity_change": np.array([]),
            "max_velocity_change": np.array([]),
            "is_converged": True,
        }

    velocity_changes = []
    max_changes = []

    for i in range(1, len(fields_sequence)):
        u_prev, v_prev = fields_sequence[i - 1]
        u_curr, v_curr = fields_sequence[i]

        vel_mag_prev = velocity.magnitude(u_prev, v_prev)
        vel_mag_curr = velocity.magnitude(u_curr, v_curr)

        diff = vel_mag_curr - vel_mag_prev
        rms_change = np.sqrt(np.mean(diff**2))
        max_change = np.max(np.abs(diff))

        velocity_changes.append(rms_change)
        max_changes.append(max_change)

    velocity_changes = np.array(velocity_changes)
    max_changes = np.array(max_changes)

    # Consider converged if last RMS change is below 1e-6
    is_converged = velocity_changes[-1] < 1e-6 if len(velocity_changes) > 0 else True

    return {
        "velocity_change": velocity_changes,
        "max_velocity_change": max_changes,
        "is_converged": is_converged,
    }


def compute_error_norms(
    u_computed: NDArray,
    v_computed: NDArray,
    u_reference: NDArray,
    v_reference: NDArray,
) -> Dict[str, float]:
    """Compute various error norms between computed and reference solutions.

    Args:
        u_computed, v_computed: Computed velocity components.
        u_reference, v_reference: Reference/analytical velocity components.

    Returns:
        Dict with L1, L2, and L_inf norms for u, v, and velocity magnitude.
    """
    # U-velocity errors
    u_error = u_computed - u_reference
    u_l1 = float(np.mean(np.abs(u_error)))
    u_l2 = float(np.sqrt(np.mean(u_error**2)))
    u_linf = float(np.max(np.abs(u_error)))

    # V-velocity errors
    v_error = v_computed - v_reference
    v_l1 = float(np.mean(np.abs(v_error)))
    v_l2 = float(np.sqrt(np.mean(v_error**2)))
    v_linf = float(np.max(np.abs(v_error)))

    # Velocity magnitude errors
    vel_computed = velocity.magnitude(u_computed, v_computed)
    vel_reference = velocity.magnitude(u_reference, v_reference)
    vel_error = vel_computed - vel_reference
    vel_l1 = float(np.mean(np.abs(vel_error)))
    vel_l2 = float(np.sqrt(np.mean(vel_error**2)))
    vel_linf = float(np.max(np.abs(vel_error)))

    return {
        "u_l1": u_l1,
        "u_l2": u_l2,
        "u_linf": u_linf,
        "v_l1": v_l1,
        "v_l2": v_l2,
        "v_linf": v_linf,
        "velocity_l1": vel_l1,
        "velocity_l2": vel_l2,
        "velocity_linf": vel_linf,
    }
