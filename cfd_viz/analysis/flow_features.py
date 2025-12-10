"""Pure functions for detecting and analyzing flow features.

This module provides functions for identifying flow features such as
wake regions, recirculation zones, and velocity fluctuations. All
functions return data structures rather than producing plots directly.

Example:
    >>> from cfd_viz.analysis.flow_features import detect_wake_regions
    >>> wake = detect_wake_regions(u, v)
    >>> print(f"Wake area fraction: {wake.area_fraction:.2%}")
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class WakeRegion:
    """Wake region detection results.

    Attributes:
        mask: Boolean array where True indicates wake region.
        threshold: Velocity threshold used for detection.
        area_fraction: Fraction of domain identified as wake.
        centroid: (x_idx, y_idx) of wake centroid, or None if no wake.
        min_velocity: Minimum velocity magnitude in the wake.
        mean_velocity: Mean velocity magnitude in the wake.
    """

    mask: NDArray
    threshold: float
    area_fraction: float
    centroid: Optional[Tuple[float, float]]
    min_velocity: float
    mean_velocity: float

    @property
    def num_cells(self) -> int:
        """Number of cells in wake region."""
        return int(np.sum(self.mask))


@dataclass
class SpatialFluctuations:
    """Spatial velocity fluctuations from a reference profile.

    Attributes:
        u_fluct: U-velocity fluctuation field.
        v_fluct: V-velocity fluctuation field.
        fluct_magnitude: Fluctuation magnitude field.
        u_mean_profile: Mean u-velocity profile (averaged along reference axis).
        v_mean_profile: Mean v-velocity profile (averaged along reference axis).
        rms_u: RMS of u-velocity fluctuations.
        rms_v: RMS of v-velocity fluctuations.
        turbulence_intensity: Turbulence intensity based on fluctuations.
    """

    u_fluct: NDArray
    v_fluct: NDArray
    fluct_magnitude: NDArray
    u_mean_profile: NDArray
    v_mean_profile: NDArray
    rms_u: float
    rms_v: float
    turbulence_intensity: float


@dataclass
class CrossSectionalAverages:
    """Cross-sectional averaged profiles.

    Attributes:
        coordinate: Coordinate along the averaging direction.
        u_avg: Averaged u-velocity profile.
        v_avg: Averaged v-velocity profile.
        velocity_mag_avg: Averaged velocity magnitude profile.
        p_avg: Averaged pressure profile (if available).
        averaging_axis: Axis along which averaging was performed ('x' or 'y').
    """

    coordinate: NDArray
    u_avg: NDArray
    v_avg: NDArray
    velocity_mag_avg: NDArray
    p_avg: Optional[NDArray]
    averaging_axis: str

    @property
    def bulk_velocity(self) -> float:
        """Overall bulk velocity."""
        return float(np.mean(self.velocity_mag_avg))


@dataclass
class RecirculationZone:
    """Recirculation zone detection results.

    Attributes:
        mask: Boolean array where True indicates recirculation.
        area_fraction: Fraction of domain with recirculation.
        centroid: (x_idx, y_idx) of recirculation centroid.
        mean_reverse_velocity: Mean reverse velocity in recirculation.
    """

    mask: NDArray
    area_fraction: float
    centroid: Optional[Tuple[float, float]]
    mean_reverse_velocity: float


def detect_wake_regions(
    u: NDArray,
    v: NDArray,
    threshold_fraction: float = 0.1,
    reference_velocity: Optional[float] = None,
) -> WakeRegion:
    """Detect wake regions based on low velocity magnitude.

    Wake regions are identified where velocity magnitude falls below
    a threshold fraction of the reference velocity.

    Args:
        u, v: Velocity component fields.
        threshold_fraction: Fraction of reference velocity below which
            flow is considered wake (default 0.1 = 10%).
        reference_velocity: Reference velocity for threshold. If None,
            uses maximum velocity magnitude in the field.

    Returns:
        WakeRegion with mask and statistics.
    """
    velocity_mag = np.sqrt(u**2 + v**2)

    if reference_velocity is None:
        reference_velocity = float(np.max(velocity_mag))

    threshold = threshold_fraction * reference_velocity
    wake_mask = velocity_mag < threshold

    area_fraction = float(np.mean(wake_mask))

    # Calculate centroid if wake exists
    centroid = None
    if np.any(wake_mask):
        y_indices, x_indices = np.where(wake_mask)
        centroid = (float(np.mean(x_indices)), float(np.mean(y_indices)))

    # Wake statistics
    if np.any(wake_mask):
        min_velocity = float(np.min(velocity_mag[wake_mask]))
        mean_velocity = float(np.mean(velocity_mag[wake_mask]))
    else:
        min_velocity = float(np.min(velocity_mag))
        mean_velocity = 0.0

    return WakeRegion(
        mask=wake_mask,
        threshold=threshold,
        area_fraction=area_fraction,
        centroid=centroid,
        min_velocity=min_velocity,
        mean_velocity=mean_velocity,
    )


def compute_spatial_fluctuations(
    u: NDArray,
    v: NDArray,
    averaging_axis: int = 1,
) -> SpatialFluctuations:
    """Compute spatial velocity fluctuations from mean profile.

    Calculates fluctuations by subtracting the mean profile (averaged
    along the specified axis) from each location.

    Args:
        u, v: Velocity component fields.
        averaging_axis: Axis along which to compute mean (0=y, 1=x).
            Default 1 averages along x, giving y-varying mean profile.

    Returns:
        SpatialFluctuations with fluctuation fields and statistics.
    """
    # Compute mean profiles
    u_mean = np.mean(u, axis=averaging_axis, keepdims=True)
    v_mean = np.mean(v, axis=averaging_axis, keepdims=True)

    # Compute fluctuations
    u_fluct = u - u_mean
    v_fluct = v - v_mean
    fluct_magnitude = np.sqrt(u_fluct**2 + v_fluct**2)

    # RMS values
    rms_u = float(np.sqrt(np.mean(u_fluct**2)))
    rms_v = float(np.sqrt(np.mean(v_fluct**2)))

    # Turbulence intensity (fluctuation RMS / mean velocity)
    mean_vel = np.sqrt(np.mean(u) ** 2 + np.mean(v) ** 2)
    if mean_vel > 1e-10:
        turbulence_intensity = np.sqrt(rms_u**2 + rms_v**2) / mean_vel
    else:
        turbulence_intensity = 0.0

    return SpatialFluctuations(
        u_fluct=u_fluct,
        v_fluct=v_fluct,
        fluct_magnitude=fluct_magnitude,
        u_mean_profile=u_mean.squeeze(),
        v_mean_profile=v_mean.squeeze(),
        rms_u=rms_u,
        rms_v=rms_v,
        turbulence_intensity=float(turbulence_intensity),
    )


def compute_cross_sectional_averages(
    u: NDArray,
    v: NDArray,
    x: NDArray,
    y: NDArray,
    averaging_axis: str = "x",
    p: Optional[NDArray] = None,
) -> CrossSectionalAverages:
    """Compute cross-sectional averaged profiles.

    Args:
        u, v: Velocity component fields.
        x, y: 1D coordinate arrays.
        averaging_axis: Axis to average along ('x' or 'y').
        p: Optional pressure field.

    Returns:
        CrossSectionalAverages with averaged profiles.
    """
    velocity_mag = np.sqrt(u**2 + v**2)

    if averaging_axis == "x":
        # Average along x (axis=1), result varies with y
        u_avg = np.mean(u, axis=1)
        v_avg = np.mean(v, axis=1)
        vel_avg = np.mean(velocity_mag, axis=1)
        coordinate = y.copy()
        p_avg = np.mean(p, axis=1) if p is not None else None
    else:
        # Average along y (axis=0), result varies with x
        u_avg = np.mean(u, axis=0)
        v_avg = np.mean(v, axis=0)
        vel_avg = np.mean(velocity_mag, axis=0)
        coordinate = x.copy()
        p_avg = np.mean(p, axis=0) if p is not None else None

    return CrossSectionalAverages(
        coordinate=coordinate,
        u_avg=u_avg,
        v_avg=v_avg,
        velocity_mag_avg=vel_avg,
        p_avg=p_avg,
        averaging_axis=averaging_axis,
    )


def detect_recirculation_zones(
    u: NDArray,
    v: NDArray,
    main_flow_direction: str = "x",
) -> RecirculationZone:
    """Detect recirculation zones where flow reverses direction.

    Args:
        u, v: Velocity component fields.
        main_flow_direction: Expected main flow direction ('x' or 'y').

    Returns:
        RecirculationZone with mask and statistics.
    """
    if main_flow_direction == "x":
        # Recirculation where u < 0 (reverse flow in x)
        recirc_mask = u < 0
        reverse_velocity = -u[recirc_mask] if np.any(recirc_mask) else np.array([0])
    else:
        # Recirculation where v < 0 (reverse flow in y)
        recirc_mask = v < 0
        reverse_velocity = -v[recirc_mask] if np.any(recirc_mask) else np.array([0])

    area_fraction = float(np.mean(recirc_mask))

    # Centroid
    centroid = None
    if np.any(recirc_mask):
        y_indices, x_indices = np.where(recirc_mask)
        centroid = (float(np.mean(x_indices)), float(np.mean(y_indices)))

    mean_reverse = (
        float(np.mean(reverse_velocity)) if len(reverse_velocity) > 0 else 0.0
    )

    return RecirculationZone(
        mask=recirc_mask,
        area_fraction=area_fraction,
        centroid=centroid,
        mean_reverse_velocity=mean_reverse,
    )


def compute_pressure_gradient(
    p: NDArray,
    dx: float,
    dy: float,
) -> Tuple[NDArray, NDArray]:
    """Compute pressure gradient field.

    Args:
        p: Pressure field.
        dx, dy: Grid spacing.

    Returns:
        Tuple of (dp_dx, dp_dy) gradient arrays.
    """
    dp_dy = np.gradient(p, dy, axis=0)
    dp_dx = np.gradient(p, dx, axis=1)

    return dp_dx, dp_dy


def compute_adverse_pressure_gradient(
    p: NDArray,
    u: NDArray,
    v: NDArray,
    dx: float,
    dy: float,
) -> NDArray:
    """Compute adverse pressure gradient indicator.

    Adverse pressure gradient occurs when pressure increases in the
    flow direction.

    Args:
        p: Pressure field.
        u, v: Velocity component fields.
        dx, dy: Grid spacing.

    Returns:
        Scalar field where positive values indicate adverse gradient.
    """
    dp_dx, dp_dy = compute_pressure_gradient(p, dx, dy)

    # Velocity magnitude for normalization
    vel_mag = np.sqrt(u**2 + v**2)
    vel_mag = np.maximum(vel_mag, 1e-10)  # Avoid division by zero

    # Unit velocity direction
    u_hat = u / vel_mag
    v_hat = v / vel_mag

    # Pressure gradient in flow direction
    dp_ds = dp_dx * u_hat + dp_dy * v_hat

    return dp_ds
