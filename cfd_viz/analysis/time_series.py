"""Pure functions for time series analysis of CFD simulations.

This module provides functions for analyzing temporal evolution of flow fields,
computing time-averaged statistics, and monitoring simulation convergence.
All functions return data structures rather than producing plots directly.

Example:
    >>> from cfd_viz.analysis.time_series import compute_temporal_statistics
    >>> stats = compute_temporal_statistics(velocity_history, time_values)
    >>> print(f"Mean velocity: {stats.mean}")
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from cfd_viz.fields import velocity


@dataclass
class TemporalStatistics:
    """Temporal statistics for a scalar quantity.

    Attributes:
        time: Time array.
        values: Values at each time.
        mean: Time-averaged value.
        std: Standard deviation.
        min_value: Minimum value.
        max_value: Maximum value.
        trend: Linear trend (slope, intercept) if computed.
        is_stationary: True if signal appears stationary.
    """

    time: NDArray
    values: NDArray
    mean: float
    std: float
    min_value: float
    max_value: float
    trend: Optional[Tuple[float, float]]
    is_stationary: bool

    @property
    def coefficient_of_variation(self) -> float:
        """Coefficient of variation (std/mean)."""
        return self.std / self.mean if self.mean != 0 else 0.0


@dataclass
class FlowHistoryStatistics:
    """Time statistics for multiple flow quantities.

    Attributes:
        time: Time array.
        max_velocity: TemporalStatistics for max velocity.
        mean_velocity: TemporalStatistics for mean velocity.
        kinetic_energy: TemporalStatistics for total kinetic energy.
        max_vorticity: TemporalStatistics for max vorticity magnitude.
        mean_pressure: TemporalStatistics for mean pressure (if available).
    """

    time: NDArray
    max_velocity: TemporalStatistics
    mean_velocity: TemporalStatistics
    kinetic_energy: TemporalStatistics
    max_vorticity: TemporalStatistics
    mean_pressure: Optional[TemporalStatistics]


@dataclass
class ConvergenceMetrics:
    """Metrics for assessing simulation convergence.

    Attributes:
        iteration: Iteration/time step numbers.
        residuals: Dict mapping residual names to arrays.
        convergence_rate: Estimated convergence rate for each residual.
        is_converged: True if all residuals below tolerance.
        iterations_to_convergence: Estimated iterations to reach tolerance.
    """

    iteration: NDArray
    residuals: Dict[str, NDArray]
    convergence_rate: Dict[str, float]
    is_converged: bool
    iterations_to_convergence: Optional[int]


@dataclass
class ProbeTimeSeries:
    """Time series data at a specific probe location.

    Attributes:
        time: Time array.
        location: (x, y) coordinates of probe.
        u: U-velocity time series.
        v: V-velocity time series.
        velocity_mag: Velocity magnitude time series.
        pressure: Pressure time series (if available).
        mean_u: Time-averaged u-velocity.
        mean_v: Time-averaged v-velocity.
        rms_u: RMS of u-fluctuations.
        rms_v: RMS of v-fluctuations.
        turbulent_intensity: Local turbulent intensity.
    """

    time: NDArray
    location: Tuple[float, float]
    u: NDArray
    v: NDArray
    velocity_mag: NDArray
    pressure: Optional[NDArray]
    mean_u: float
    mean_v: float
    rms_u: float
    rms_v: float
    turbulent_intensity: float


def compute_temporal_statistics(
    time: NDArray,
    values: NDArray,
    compute_trend: bool = True,
    stationarity_threshold: float = 0.1,
) -> TemporalStatistics:
    """Compute temporal statistics for a time series.

    Args:
        time: Time array.
        values: Values at each time step.
        compute_trend: Whether to compute linear trend.
        stationarity_threshold: Max allowed CV for stationarity.

    Returns:
        TemporalStatistics with all computed metrics.
    """
    mean = float(np.mean(values))
    std = float(np.std(values))
    min_val = float(np.min(values))
    max_val = float(np.max(values))

    trend = None
    if compute_trend and len(time) >= 2:
        coeffs = np.polyfit(time, values, 1)
        trend = (float(coeffs[0]), float(coeffs[1]))

    # Check stationarity based on coefficient of variation
    cv = std / abs(mean) if mean != 0 else 0.0
    is_stationary = cv < stationarity_threshold

    return TemporalStatistics(
        time=time,
        values=values,
        mean=mean,
        std=std,
        min_value=min_val,
        max_value=max_val,
        trend=trend,
        is_stationary=is_stationary,
    )


def analyze_flow_history(
    time: NDArray,
    u_history: List[NDArray],
    v_history: List[NDArray],
    p_history: Optional[List[NDArray]] = None,
    dx: float = 1.0,
    dy: float = 1.0,
) -> FlowHistoryStatistics:
    """Analyze temporal evolution of flow field statistics.

    Args:
        time: Time array.
        u_history: List of u-velocity fields at each time.
        v_history: List of v-velocity fields at each time.
        p_history: List of pressure fields at each time (optional).
        dx, dy: Grid spacing for energy integration.

    Returns:
        FlowHistoryStatistics with statistics for all quantities.
    """
    n_steps = len(u_history)

    # Compute quantities at each time step
    max_vel = np.zeros(n_steps)
    mean_vel = np.zeros(n_steps)
    kinetic_energy = np.zeros(n_steps)
    max_vort = np.zeros(n_steps)
    mean_p = np.zeros(n_steps) if p_history else None

    for i in range(n_steps):
        u, v = u_history[i], v_history[i]
        vel_mag = velocity.magnitude(u, v)

        max_vel[i] = np.max(vel_mag)
        mean_vel[i] = np.mean(vel_mag)

        # Kinetic energy
        ke = 0.5 * (u**2 + v**2)
        kinetic_energy[i] = np.sum(ke) * dx * dy

        # Vorticity
        du_dy = np.gradient(u, dy, axis=0)
        dv_dx = np.gradient(v, dx, axis=1)
        vort = dv_dx - du_dy
        max_vort[i] = np.max(np.abs(vort))

        if p_history:
            mean_p[i] = np.mean(p_history[i])

    # Compute statistics for each quantity
    max_vel_stats = compute_temporal_statistics(time, max_vel)
    mean_vel_stats = compute_temporal_statistics(time, mean_vel)
    ke_stats = compute_temporal_statistics(time, kinetic_energy)
    vort_stats = compute_temporal_statistics(time, max_vort)

    p_stats = None
    if mean_p is not None:
        p_stats = compute_temporal_statistics(time, mean_p)

    return FlowHistoryStatistics(
        time=time,
        max_velocity=max_vel_stats,
        mean_velocity=mean_vel_stats,
        kinetic_energy=ke_stats,
        max_vorticity=vort_stats,
        mean_pressure=p_stats,
    )


def compute_time_averaged_field(
    field_history: List[NDArray],
    start_index: int = 0,
) -> NDArray:
    """Compute time-averaged field.

    Args:
        field_history: List of field snapshots.
        start_index: Start averaging from this index (to skip transient).

    Returns:
        Time-averaged field.
    """
    if not field_history:
        raise ValueError("Empty field history")

    fields = field_history[start_index:]
    return np.mean(np.stack(fields, axis=0), axis=0)


def compute_rms_fluctuations(
    field_history: List[NDArray],
    mean_field: Optional[NDArray] = None,
    start_index: int = 0,
) -> NDArray:
    """Compute RMS of field fluctuations.

    Args:
        field_history: List of field snapshots.
        mean_field: Pre-computed mean field (optional).
        start_index: Start from this index.

    Returns:
        RMS fluctuation field.
    """
    fields = field_history[start_index:]

    if mean_field is None:
        mean_field = np.mean(np.stack(fields, axis=0), axis=0)

    # Compute fluctuations squared
    fluct_squared = np.zeros_like(mean_field)
    for field in fields:
        fluct_squared += (field - mean_field) ** 2

    fluct_squared /= len(fields)
    return np.sqrt(fluct_squared)


def extract_probe_time_series(
    time: NDArray,
    u_history: List[NDArray],
    v_history: List[NDArray],
    x: NDArray,
    y: NDArray,
    probe_location: Tuple[float, float],
    p_history: Optional[List[NDArray]] = None,
) -> ProbeTimeSeries:
    """Extract time series at a specific probe location.

    Args:
        time: Time array.
        u_history, v_history: Velocity field histories.
        x, y: 1D coordinate arrays.
        probe_location: (x, y) coordinates for probe.
        p_history: Pressure field history (optional).

    Returns:
        ProbeTimeSeries with all data at probe location.
    """
    # Find nearest grid point
    i_x = np.argmin(np.abs(x - probe_location[0]))
    i_y = np.argmin(np.abs(y - probe_location[1]))
    actual_location = (x[i_x], y[i_y])

    # Extract time series
    u_probe = np.array([u[i_y, i_x] for u in u_history])
    v_probe = np.array([v[i_y, i_x] for v in v_history])
    vel_mag = np.sqrt(u_probe**2 + v_probe**2)

    p_probe = None
    if p_history:
        p_probe = np.array([p[i_y, i_x] for p in p_history])

    # Compute statistics
    mean_u = float(np.mean(u_probe))
    mean_v = float(np.mean(v_probe))

    u_fluct = u_probe - mean_u
    v_fluct = v_probe - mean_v

    rms_u = float(np.sqrt(np.mean(u_fluct**2)))
    rms_v = float(np.sqrt(np.mean(v_fluct**2)))

    # Turbulent intensity
    mean_vel = np.sqrt(mean_u**2 + mean_v**2)
    ti = np.sqrt(0.5 * (rms_u**2 + rms_v**2)) / mean_vel if mean_vel > 0 else 0.0

    return ProbeTimeSeries(
        time=time,
        location=actual_location,
        u=u_probe,
        v=v_probe,
        velocity_mag=vel_mag,
        pressure=p_probe,
        mean_u=mean_u,
        mean_v=mean_v,
        rms_u=rms_u,
        rms_v=rms_v,
        turbulent_intensity=float(ti),
    )


def analyze_convergence(
    iterations: NDArray,
    residuals: Dict[str, NDArray],
    tolerance: float = 1e-6,
) -> ConvergenceMetrics:
    """Analyze convergence behavior from residual history.

    Args:
        iterations: Iteration numbers.
        residuals: Dict mapping residual names to value arrays.
        tolerance: Convergence tolerance.

    Returns:
        ConvergenceMetrics with convergence analysis.
    """
    convergence_rates: Dict[str, float] = {}
    all_converged = True

    for name, values in residuals.items():
        # Check if converged
        if values[-1] > tolerance:
            all_converged = False

        # Estimate convergence rate from recent iterations
        if len(values) >= 10:
            recent = values[-10:]
            log_recent = np.log10(np.maximum(recent, 1e-15))
            coeffs = np.polyfit(range(10), log_recent, 1)
            convergence_rates[name] = float(coeffs[0])  # Slope in log scale
        else:
            convergence_rates[name] = 0.0

    # Estimate iterations to convergence
    iters_to_conv = None
    if not all_converged:
        # Use primary residual for estimate
        primary_name = next(iter(residuals.keys()))
        primary_vals = residuals[primary_name]
        rate = convergence_rates[primary_name]

        if rate < 0:  # Converging
            log_current = np.log10(max(primary_vals[-1], 1e-15))
            log_target = np.log10(tolerance)
            iters_needed = (log_target - log_current) / rate
            iters_to_conv = max(0, int(np.ceil(iters_needed)))

    return ConvergenceMetrics(
        iteration=iterations,
        residuals=residuals,
        convergence_rate=convergence_rates,
        is_converged=all_converged,
        iterations_to_convergence=iters_to_conv,
    )


def compute_running_average(
    values: NDArray,
    window_size: int = 10,
) -> NDArray:
    """Compute running average of a time series.

    Args:
        values: Input time series.
        window_size: Averaging window size.

    Returns:
        Running average array (same length, with edge effects).
    """
    if len(values) < window_size:
        return np.full_like(values, np.mean(values))

    # Use convolution for efficient computation
    kernel = np.ones(window_size) / window_size
    running_avg = np.convolve(values, kernel, mode="same")

    # Fix edge effects
    for i in range(window_size // 2):
        running_avg[i] = np.mean(values[: i + window_size // 2 + 1])
        running_avg[-(i + 1)] = np.mean(values[-(i + window_size // 2 + 1) :])

    return running_avg


def detect_periodicity(
    values: NDArray,
    sampling_rate: float = 1.0,
) -> Optional[float]:
    """Detect dominant period in time series using FFT.

    Args:
        values: Time series values.
        sampling_rate: Sampling frequency.

    Returns:
        Dominant period (time units), or None if no clear periodicity.
    """
    if len(values) < 4:
        return None

    # Remove mean
    centered = values - np.mean(values)

    # Check if signal has any variation
    if np.std(centered) < 1e-10:
        return None  # Constant signal has no periodicity

    # FFT
    fft = np.fft.rfft(centered)
    freqs = np.fft.rfftfreq(len(centered), d=1.0 / sampling_rate)

    # Find peak (excluding DC component)
    magnitudes = np.abs(fft[1:])
    if len(magnitudes) == 0:
        return None

    peak_idx = np.argmax(magnitudes) + 1  # +1 because we excluded DC

    # Check if peak is significant
    mean_mag = np.mean(magnitudes)
    if mean_mag < 1e-10 or magnitudes[peak_idx - 1] < 3 * mean_mag:
        return None  # No clear peak

    # Convert to period
    dominant_freq = freqs[peak_idx]
    if dominant_freq > 0:
        return float(1.0 / dominant_freq)
    return None


@dataclass
class FlowMetrics:
    """Flow field metrics at a single time instant.

    Attributes:
        timestamp: Time of measurement.
        max_velocity: Maximum velocity magnitude.
        mean_velocity: Mean velocity magnitude.
        max_pressure: Maximum pressure.
        mean_pressure: Mean pressure.
        total_kinetic_energy: Total kinetic energy.
        max_vorticity: Maximum vorticity magnitude.
    """

    timestamp: float
    max_velocity: float
    mean_velocity: float
    max_pressure: float
    mean_pressure: float
    total_kinetic_energy: float
    max_vorticity: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "max_velocity": self.max_velocity,
            "mean_velocity": self.mean_velocity,
            "max_pressure": self.max_pressure,
            "mean_pressure": self.mean_pressure,
            "total_kinetic_energy": self.total_kinetic_energy,
            "max_vorticity": self.max_vorticity,
        }


@dataclass
class FlowMetricsTimeSeries:
    """Time series of flow metrics for convergence tracking.

    Attributes:
        snapshots: List of FlowMetrics objects.
        max_length: Maximum number of entries to keep.
    """

    snapshots: List[FlowMetrics]
    max_length: int = 100

    def add(self, metrics: FlowMetrics) -> None:
        """Add metrics, trimming old entries if needed."""
        self.snapshots.append(metrics)
        if len(self.snapshots) > self.max_length:
            self.snapshots = self.snapshots[-self.max_length :]

    def get_metric_array(self, metric_name: str) -> NDArray:
        """Get array of values for a specific metric."""
        return np.array([getattr(s, metric_name) for s in self.snapshots])

    def get_timestamps(self) -> NDArray:
        """Get array of timestamps."""
        return np.array([s.timestamp for s in self.snapshots])

    def estimate_convergence_trend(
        self, metric_name: str, window: int = 5
    ) -> Optional[float]:
        """Estimate trend for a metric over recent entries.

        Args:
            metric_name: Name of the metric to analyze.
            window: Number of recent entries to use.

        Returns:
            Slope of linear fit, or None if insufficient data.
        """
        if len(self.snapshots) < window:
            return None

        recent = self.snapshots[-window:]
        values = np.array([getattr(s, metric_name) for s in recent])
        x = np.arange(window)
        slope = np.polyfit(x, values, 1)[0]
        return float(slope)

    def is_converged(self, threshold: float = 1e-5) -> bool:
        """Check if simulation appears converged based on velocity trend."""
        trend = self.estimate_convergence_trend("max_velocity")
        if trend is None:
            return False
        return abs(trend) < threshold


def compute_flow_metrics(
    u: NDArray,
    v: NDArray,
    p: Optional[NDArray],
    dx: float,
    dy: float,
    timestamp: float = 0.0,
) -> FlowMetrics:
    """Compute flow metrics for a single time instant.

    Args:
        u, v: Velocity component fields.
        p: Pressure field (optional).
        dx, dy: Grid spacing.
        timestamp: Time of this measurement.

    Returns:
        FlowMetrics with all computed values.
    """
    velocity_mag = velocity.magnitude(u, v)

    # Vorticity
    du_dy = np.gradient(u, dy, axis=0)
    dv_dx = np.gradient(v, dx, axis=1)
    vorticity = dv_dx - du_dy

    # Kinetic energy
    kinetic_energy = 0.5 * (u**2 + v**2)
    total_ke = float(np.sum(kinetic_energy) * dx * dy)

    # Pressure (default to zeros if not provided)
    if p is None:
        p = np.zeros_like(u)

    return FlowMetrics(
        timestamp=timestamp,
        max_velocity=float(np.max(velocity_mag)),
        mean_velocity=float(np.mean(velocity_mag)),
        max_pressure=float(np.max(p)),
        mean_pressure=float(np.mean(p)),
        total_kinetic_energy=total_ke,
        max_vorticity=float(np.max(np.abs(vorticity))),
    )


def create_flow_metrics_time_series(max_length: int = 100) -> FlowMetricsTimeSeries:
    """Create an empty flow metrics time series.

    Args:
        max_length: Maximum number of entries to retain.

    Returns:
        Empty FlowMetricsTimeSeries object.
    """
    return FlowMetricsTimeSeries(snapshots=[], max_length=max_length)
