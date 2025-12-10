"""Pure functions for extracting and analyzing flow profiles.

This module provides functions for extracting data along lines, paths, and
cross-sections in CFD flow fields. All functions return data structures
rather than producing plots directly.

Example:
    >>> from cfd_viz.analysis.profiles import extract_line_profile, LineProfile
    >>> profile = extract_line_profile(u, v, x, y, (0, 0.5), (1, 0.5))
    >>> print(f"Max velocity along line: {profile.velocity_mag.max()}")
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator


@dataclass
class LineProfile:
    """Data along a line through a flow field.

    Attributes:
        distance: Distance along the line from start point.
        x_coords: X coordinates along the line.
        y_coords: Y coordinates along the line.
        u: U-velocity component along the line.
        v: V-velocity component along the line.
        velocity_mag: Velocity magnitude along the line.
        pressure: Pressure along the line (if available).
        start_point: Starting point (x, y).
        end_point: Ending point (x, y).
    """

    distance: NDArray
    x_coords: NDArray
    y_coords: NDArray
    u: NDArray
    v: NDArray
    velocity_mag: NDArray
    pressure: Optional[NDArray]
    start_point: Tuple[float, float]
    end_point: Tuple[float, float]

    @property
    def length(self) -> float:
        """Total length of the line."""
        return float(self.distance[-1]) if len(self.distance) > 0 else 0.0

    @property
    def max_velocity(self) -> float:
        """Maximum velocity magnitude along the line."""
        return float(np.max(self.velocity_mag))

    @property
    def mean_velocity(self) -> float:
        """Mean velocity magnitude along the line."""
        return float(np.mean(self.velocity_mag))


@dataclass
class CrossSection:
    """Cross-sectional data at a fixed location.

    Attributes:
        position: Position along the primary axis (x for vertical, y for horizontal).
        coordinate: Coordinate values along the cross-section.
        u: U-velocity profile.
        v: V-velocity profile.
        velocity_mag: Velocity magnitude profile.
        pressure: Pressure profile (if available).
        is_vertical: True if cross-section is vertical (constant x).
    """

    position: float
    coordinate: NDArray
    u: NDArray
    v: NDArray
    velocity_mag: NDArray
    pressure: Optional[NDArray]
    is_vertical: bool

    @property
    def bulk_velocity(self) -> float:
        """Bulk (area-averaged) velocity magnitude."""
        return float(np.mean(self.velocity_mag))

    @property
    def max_velocity(self) -> float:
        """Maximum velocity in cross-section."""
        return float(np.max(self.velocity_mag))


@dataclass
class MultipleProfiles:
    """Collection of profiles at multiple locations.

    Attributes:
        profiles: List of CrossSection objects.
        positions: Positions where profiles were extracted.
        is_vertical: True if profiles are vertical cross-sections.
    """

    profiles: List[CrossSection]
    positions: List[float]
    is_vertical: bool

    def get_profile_at(self, position: float) -> Optional[CrossSection]:
        """Get profile closest to specified position."""
        if not self.profiles:
            return None
        idx = np.argmin(np.abs(np.array(self.positions) - position))
        return self.profiles[idx]

    def get_bulk_velocities(self) -> NDArray:
        """Get bulk velocities at all profile locations."""
        return np.array([p.bulk_velocity for p in self.profiles])


def extract_line_profile(
    u: NDArray,
    v: NDArray,
    x: NDArray,
    y: NDArray,
    start_point: Tuple[float, float],
    end_point: Tuple[float, float],
    p: Optional[NDArray] = None,
    num_points: int = 100,
) -> LineProfile:
    """Extract flow data along a line between two points.

    Args:
        u, v: Velocity component fields.
        x, y: 1D coordinate arrays.
        start_point: (x, y) coordinates of line start.
        end_point: (x, y) coordinates of line end.
        p: Optional pressure field.
        num_points: Number of points along the line.

    Returns:
        LineProfile with interpolated data along the line.
    """
    # Create line coordinates
    line_x = np.linspace(start_point[0], end_point[0], num_points)
    line_y = np.linspace(start_point[1], end_point[1], num_points)

    # Calculate distance along line
    distance = np.sqrt((line_x - start_point[0]) ** 2 + (line_y - start_point[1]) ** 2)

    # Create interpolators (note: scipy expects (y, x) order for 2D arrays)
    interp_u = RegularGridInterpolator((y, x), u, bounds_error=False, fill_value=np.nan)
    interp_v = RegularGridInterpolator((y, x), v, bounds_error=False, fill_value=np.nan)

    # Extract values along line
    line_points = np.column_stack([line_y, line_x])
    u_line = interp_u(line_points)
    v_line = interp_v(line_points)
    vel_mag = np.sqrt(u_line**2 + v_line**2)

    # Pressure if available
    p_line = None
    if p is not None:
        interp_p = RegularGridInterpolator(
            (y, x), p, bounds_error=False, fill_value=np.nan
        )
        p_line = interp_p(line_points)

    return LineProfile(
        distance=distance,
        x_coords=line_x,
        y_coords=line_y,
        u=u_line,
        v=v_line,
        velocity_mag=vel_mag,
        pressure=p_line,
        start_point=start_point,
        end_point=end_point,
    )


def extract_vertical_profile(
    u: NDArray,
    v: NDArray,
    x: NDArray,
    y: NDArray,
    x_position: float,
    p: Optional[NDArray] = None,
) -> CrossSection:
    """Extract a vertical cross-section at a given x position.

    Args:
        u, v: Velocity component fields.
        x, y: 1D coordinate arrays.
        x_position: X coordinate for the vertical profile.
        p: Optional pressure field.

    Returns:
        CrossSection with data along vertical line.
    """
    # Find closest x index
    x_idx = np.argmin(np.abs(x - x_position))
    actual_x = x[x_idx]

    # Extract vertical profile
    u_profile = u[:, x_idx]
    v_profile = v[:, x_idx]
    vel_mag = np.sqrt(u_profile**2 + v_profile**2)

    p_profile = None
    if p is not None:
        p_profile = p[:, x_idx]

    return CrossSection(
        position=actual_x,
        coordinate=y.copy(),
        u=u_profile,
        v=v_profile,
        velocity_mag=vel_mag,
        pressure=p_profile,
        is_vertical=True,
    )


def extract_horizontal_profile(
    u: NDArray,
    v: NDArray,
    x: NDArray,
    y: NDArray,
    y_position: float,
    p: Optional[NDArray] = None,
) -> CrossSection:
    """Extract a horizontal cross-section at a given y position.

    Args:
        u, v: Velocity component fields.
        x, y: 1D coordinate arrays.
        y_position: Y coordinate for the horizontal profile.
        p: Optional pressure field.

    Returns:
        CrossSection with data along horizontal line.
    """
    # Find closest y index
    y_idx = np.argmin(np.abs(y - y_position))
    actual_y = y[y_idx]

    # Extract horizontal profile
    u_profile = u[y_idx, :]
    v_profile = v[y_idx, :]
    vel_mag = np.sqrt(u_profile**2 + v_profile**2)

    p_profile = None
    if p is not None:
        p_profile = p[y_idx, :]

    return CrossSection(
        position=actual_y,
        coordinate=x.copy(),
        u=u_profile,
        v=v_profile,
        velocity_mag=vel_mag,
        pressure=p_profile,
        is_vertical=False,
    )


def extract_multiple_profiles(
    u: NDArray,
    v: NDArray,
    x: NDArray,
    y: NDArray,
    positions: List[float],
    vertical: bool = True,
    p: Optional[NDArray] = None,
) -> MultipleProfiles:
    """Extract multiple cross-sectional profiles.

    Args:
        u, v: Velocity component fields.
        x, y: 1D coordinate arrays.
        positions: List of positions for profile extraction.
        vertical: If True, extract vertical profiles; otherwise horizontal.
        p: Optional pressure field.

    Returns:
        MultipleProfiles containing all extracted profiles.
    """
    profiles = []
    actual_positions = []

    for pos in positions:
        if vertical:
            profile = extract_vertical_profile(u, v, x, y, pos, p)
        else:
            profile = extract_horizontal_profile(u, v, x, y, pos, p)
        profiles.append(profile)
        actual_positions.append(profile.position)

    return MultipleProfiles(
        profiles=profiles,
        positions=actual_positions,
        is_vertical=vertical,
    )


def compute_centerline_profiles(
    u: NDArray,
    v: NDArray,
    x: NDArray,
    y: NDArray,
    p: Optional[NDArray] = None,
) -> Tuple[CrossSection, CrossSection]:
    """Extract both horizontal and vertical centerline profiles.

    Args:
        u, v: Velocity component fields.
        x, y: 1D coordinate arrays.
        p: Optional pressure field.

    Returns:
        Tuple of (horizontal_centerline, vertical_centerline).
    """
    x_center = (x[0] + x[-1]) / 2
    y_center = (y[0] + y[-1]) / 2

    horizontal = extract_horizontal_profile(u, v, x, y, y_center, p)
    vertical = extract_vertical_profile(u, v, x, y, x_center, p)

    return horizontal, vertical


def compute_profile_statistics(profile: CrossSection) -> dict:
    """Compute statistics for a cross-sectional profile.

    Args:
        profile: CrossSection to analyze.

    Returns:
        Dict with velocity statistics.
    """
    return {
        "max_velocity": float(np.max(profile.velocity_mag)),
        "min_velocity": float(np.min(profile.velocity_mag)),
        "mean_velocity": float(np.mean(profile.velocity_mag)),
        "std_velocity": float(np.std(profile.velocity_mag)),
        "max_u": float(np.max(profile.u)),
        "min_u": float(np.min(profile.u)),
        "max_v": float(np.max(profile.v)),
        "min_v": float(np.min(profile.v)),
        "position": profile.position,
    }


def compute_mass_flow_rate(
    profile: CrossSection,
    rho: float = 1.0,
    width: float = 1.0,
) -> float:
    """Compute mass flow rate through a cross-section.

    Args:
        profile: CrossSection to analyze.
        rho: Fluid density (default 1.0 for incompressible).
        width: Width in third dimension (for 2D, typically 1.0).

    Returns:
        Mass flow rate through the cross-section.
    """
    # Determine which velocity component is normal to the cross-section
    if profile.is_vertical:
        # Vertical profile: u is normal velocity
        normal_velocity = profile.u
    else:
        # Horizontal profile: v is normal velocity
        normal_velocity = profile.v

    # Integrate using trapezoidal rule
    dcoord = np.gradient(profile.coordinate)
    mass_flow = rho * width * np.sum(normal_velocity * dcoord)

    return float(mass_flow)


def compute_momentum_flux(
    profile: CrossSection,
    rho: float = 1.0,
    width: float = 1.0,
) -> Tuple[float, float]:
    """Compute momentum flux through a cross-section.

    Args:
        profile: CrossSection to analyze.
        rho: Fluid density.
        width: Width in third dimension.

    Returns:
        Tuple of (x_momentum_flux, y_momentum_flux).
    """
    if profile.is_vertical:
        normal_velocity = profile.u
    else:
        normal_velocity = profile.v

    dcoord = np.gradient(profile.coordinate)

    # Momentum flux = rho * u_normal * u_i
    x_momentum = rho * width * np.sum(normal_velocity * profile.u * dcoord)
    y_momentum = rho * width * np.sum(normal_velocity * profile.v * dcoord)

    return float(x_momentum), float(y_momentum)
