"""Pure functions for boundary layer analysis.

This module provides functions for analyzing boundary layer profiles,
calculating integral parameters, and characterizing near-wall flow.
All functions return data structures rather than producing plots directly.

Example:
    >>> from cfd_viz.analysis.boundary_layer import analyze_boundary_layer
    >>> bl = analyze_boundary_layer(u, v, x, y, wall_y=0.0, x_location=0.5)
    >>> print(f"Boundary layer thickness: {bl.delta_99}")
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class BoundaryLayerProfile:
    """Boundary layer profile at a specific streamwise location.

    Attributes:
        x_location: Streamwise position of the profile.
        wall_distance: Distance from wall (y - y_wall).
        u: Streamwise velocity profile.
        v: Wall-normal velocity profile.
        u_edge: Edge (freestream) velocity.
        delta_99: 99% boundary layer thickness.
        delta_star: Displacement thickness.
        theta: Momentum thickness.
        H: Shape factor (delta_star / theta).
        cf: Skin friction coefficient (if wall shear available).
        Re_theta: Reynolds number based on momentum thickness.
        Re_delta_star: Reynolds number based on displacement thickness.
    """

    x_location: float
    wall_distance: NDArray
    u: NDArray
    v: NDArray
    u_edge: float
    delta_99: float
    delta_star: float
    theta: float
    H: float
    cf: Optional[float]
    Re_theta: Optional[float]
    Re_delta_star: Optional[float]

    @property
    def u_normalized(self) -> NDArray:
        """Velocity normalized by edge velocity."""
        return self.u / self.u_edge if self.u_edge != 0 else self.u

    @property
    def y_normalized(self) -> NDArray:
        """Wall distance normalized by boundary layer thickness."""
        return (
            self.wall_distance / self.delta_99
            if self.delta_99 > 0
            else self.wall_distance
        )


@dataclass
class BoundaryLayerDevelopment:
    """Boundary layer development along a surface.

    Attributes:
        x_locations: Streamwise positions.
        profiles: List of BoundaryLayerProfile at each location.
        delta_99: Array of boundary layer thicknesses.
        delta_star: Array of displacement thicknesses.
        theta: Array of momentum thicknesses.
        H: Array of shape factors.
        cf: Array of skin friction coefficients (if available).
    """

    x_locations: NDArray
    profiles: List[BoundaryLayerProfile]
    delta_99: NDArray
    delta_star: NDArray
    theta: NDArray
    H: NDArray
    cf: Optional[NDArray]

    def get_profile_at(self, x: float) -> Optional[BoundaryLayerProfile]:
        """Get profile closest to specified x location."""
        if not self.profiles:
            return None
        idx = np.argmin(np.abs(self.x_locations - x))
        return self.profiles[idx]


@dataclass
class WallShearStress:
    """Wall shear stress distribution.

    Attributes:
        x: Streamwise coordinates.
        tau_w: Wall shear stress.
        cf: Skin friction coefficient.
        u_tau: Friction velocity.
    """

    x: NDArray
    tau_w: NDArray
    cf: NDArray
    u_tau: NDArray


def find_boundary_layer_edge(
    wall_distance: NDArray,
    u: NDArray,
    threshold: float = 0.99,
) -> Tuple[float, float]:
    """Find the boundary layer edge and edge velocity.

    Args:
        wall_distance: Distance from wall.
        u: Streamwise velocity profile.
        threshold: Fraction of freestream velocity for BL edge (default 0.99).

    Returns:
        Tuple of (delta_99, u_edge).
    """
    # Assume the last point is in the freestream
    u_edge = u[-1]

    if u_edge <= 0:
        return 0.0, 0.0

    # Find where u/u_edge crosses threshold
    u_ratio = u / u_edge
    idx = np.where(u_ratio >= threshold)[0]

    if len(idx) == 0:
        # BL extends beyond the domain
        delta_99 = wall_distance[-1]
    else:
        # Interpolate to find exact location
        idx_first = idx[0]
        if idx_first == 0:
            delta_99 = wall_distance[0]
        else:
            # Linear interpolation
            y1, y2 = wall_distance[idx_first - 1], wall_distance[idx_first]
            u1, u2 = u_ratio[idx_first - 1], u_ratio[idx_first]
            delta_99 = y1 + (threshold - u1) * (y2 - y1) / (u2 - u1 + 1e-10)

    return float(delta_99), float(u_edge)


def compute_displacement_thickness(
    wall_distance: NDArray,
    u: NDArray,
    u_edge: float,
) -> float:
    """Compute displacement thickness.

    delta_star = integral(0, inf) of (1 - u/u_edge) dy

    Args:
        wall_distance: Distance from wall.
        u: Streamwise velocity profile.
        u_edge: Edge (freestream) velocity.

    Returns:
        Displacement thickness.
    """
    if u_edge <= 0:
        return 0.0

    integrand = 1.0 - u / u_edge
    delta_star = np.trapz(integrand, wall_distance)
    return float(max(delta_star, 0.0))


def compute_momentum_thickness(
    wall_distance: NDArray,
    u: NDArray,
    u_edge: float,
) -> float:
    """Compute momentum thickness.

    theta = integral(0, inf) of (u/u_edge) * (1 - u/u_edge) dy

    Args:
        wall_distance: Distance from wall.
        u: Streamwise velocity profile.
        u_edge: Edge (freestream) velocity.

    Returns:
        Momentum thickness.
    """
    if u_edge <= 0:
        return 0.0

    u_ratio = u / u_edge
    integrand = u_ratio * (1.0 - u_ratio)
    theta = np.trapz(integrand, wall_distance)
    return float(max(theta, 0.0))


def compute_wall_shear(
    u: NDArray,
    wall_distance: NDArray,
    mu: float,
) -> float:
    """Compute wall shear stress from velocity gradient at wall.

    Args:
        u: Streamwise velocity profile (from wall outward).
        wall_distance: Distance from wall.
        mu: Dynamic viscosity.

    Returns:
        Wall shear stress tau_w.
    """
    # Use forward difference at wall
    if len(u) < 2:
        return 0.0

    du_dy_wall = (u[1] - u[0]) / (wall_distance[1] - wall_distance[0] + 1e-15)
    tau_w = mu * du_dy_wall
    return float(tau_w)


def analyze_boundary_layer(
    u: NDArray,
    v: NDArray,
    x: NDArray,
    y: NDArray,
    wall_y: float,
    x_location: float,
    mu: Optional[float] = None,
    rho: float = 1.0,
    threshold: float = 0.99,
) -> BoundaryLayerProfile:
    """Analyze boundary layer at a specific streamwise location.

    Args:
        u, v: 2D velocity fields.
        x, y: 1D coordinate arrays.
        wall_y: Y-coordinate of the wall.
        x_location: Streamwise position for analysis.
        mu: Dynamic viscosity (for shear calculations).
        rho: Density (default 1.0).
        threshold: Fraction for BL edge detection (default 0.99).

    Returns:
        BoundaryLayerProfile with all boundary layer parameters.
    """
    # Find closest x index
    x_idx = np.argmin(np.abs(x - x_location))
    actual_x = x[x_idx]

    # Find wall index and extract profile from wall
    wall_idx = np.argmin(np.abs(y - wall_y))

    # Determine which direction is "away from wall"
    if y[wall_idx] < y[-1]:
        # Wall at bottom, flow goes upward
        y_profile = y[wall_idx:]
        u_profile = u[wall_idx:, x_idx]
        v_profile = v[wall_idx:, x_idx]
    else:
        # Wall at top, flow goes downward
        y_profile = y[: wall_idx + 1][::-1]
        u_profile = u[: wall_idx + 1, x_idx][::-1]
        v_profile = v[: wall_idx + 1, x_idx][::-1]

    wall_distance = np.abs(y_profile - wall_y)

    # Find boundary layer edge
    delta_99, u_edge = find_boundary_layer_edge(wall_distance, u_profile, threshold)

    # Compute integral thicknesses
    delta_star = compute_displacement_thickness(wall_distance, u_profile, u_edge)
    theta = compute_momentum_thickness(wall_distance, u_profile, u_edge)

    # Shape factor
    H = delta_star / theta if theta > 0 else 0.0

    # Wall shear and friction coefficient
    cf = None
    Re_theta = None
    Re_delta_star = None

    if mu is not None and mu > 0:
        tau_w = compute_wall_shear(u_profile, wall_distance, mu)
        q_inf = 0.5 * rho * u_edge**2
        cf = tau_w / q_inf if q_inf > 0 else 0.0

        # Reynolds numbers
        nu = mu / rho
        if nu > 0:
            Re_theta = u_edge * theta / nu
            Re_delta_star = u_edge * delta_star / nu

    return BoundaryLayerProfile(
        x_location=actual_x,
        wall_distance=wall_distance,
        u=u_profile,
        v=v_profile,
        u_edge=u_edge,
        delta_99=delta_99,
        delta_star=delta_star,
        theta=theta,
        H=H,
        cf=cf,
        Re_theta=Re_theta,
        Re_delta_star=Re_delta_star,
    )


def analyze_boundary_layer_development(
    u: NDArray,
    v: NDArray,
    x: NDArray,
    y: NDArray,
    wall_y: float,
    x_locations: Optional[List[float]] = None,
    num_profiles: int = 10,
    mu: Optional[float] = None,
    rho: float = 1.0,
) -> BoundaryLayerDevelopment:
    """Analyze boundary layer development along a surface.

    Args:
        u, v: 2D velocity fields.
        x, y: 1D coordinate arrays.
        wall_y: Y-coordinate of the wall.
        x_locations: Specific x-locations for analysis (optional).
        num_profiles: Number of evenly-spaced profiles if x_locations not given.
        mu: Dynamic viscosity.
        rho: Density.

    Returns:
        BoundaryLayerDevelopment with profiles at all locations.
    """
    if x_locations is None:
        # Create evenly spaced locations
        x_start = x[len(x) // 10]  # Skip first 10% to avoid inlet effects
        x_end = x[-len(x) // 10]  # Skip last 10%
        x_locations = list(np.linspace(x_start, x_end, num_profiles))

    profiles = []
    for x_loc in x_locations:
        profile = analyze_boundary_layer(u, v, x, y, wall_y, x_loc, mu, rho)
        profiles.append(profile)

    # Extract arrays
    x_locs = np.array([p.x_location for p in profiles])
    delta_99 = np.array([p.delta_99 for p in profiles])
    delta_star = np.array([p.delta_star for p in profiles])
    theta_arr = np.array([p.theta for p in profiles])
    H = np.array([p.H for p in profiles])

    cf = None
    if profiles[0].cf is not None:
        cf = np.array([p.cf for p in profiles])

    return BoundaryLayerDevelopment(
        x_locations=x_locs,
        profiles=profiles,
        delta_99=delta_99,
        delta_star=delta_star,
        theta=theta_arr,
        H=H,
        cf=cf,
    )


def compute_wall_shear_distribution(
    u: NDArray,
    x: NDArray,
    y: NDArray,
    wall_y: float,
    mu: float,
    rho: float = 1.0,
    u_ref: Optional[float] = None,
) -> WallShearStress:
    """Compute wall shear stress distribution along a surface.

    Args:
        u: 2D streamwise velocity field.
        x, y: 1D coordinate arrays.
        wall_y: Y-coordinate of the wall.
        mu: Dynamic viscosity.
        rho: Density.
        u_ref: Reference velocity for cf calculation (optional).

    Returns:
        WallShearStress with distribution along wall.
    """
    # Find wall index
    wall_idx = np.argmin(np.abs(y - wall_y))

    # Compute wall shear at each x location
    tau_w = np.zeros(len(x))
    for i in range(len(x)):
        if y[wall_idx] < y[-1]:
            # Wall at bottom
            du_dy = (u[wall_idx + 1, i] - u[wall_idx, i]) / (
                y[wall_idx + 1] - y[wall_idx]
            )
        else:
            # Wall at top
            du_dy = (u[wall_idx, i] - u[wall_idx - 1, i]) / (
                y[wall_idx] - y[wall_idx - 1]
            )
        tau_w[i] = mu * du_dy

    # Friction coefficient
    if u_ref is None:
        # Use max velocity as reference
        u_ref = np.max(np.abs(u))

    q_ref = 0.5 * rho * u_ref**2 if u_ref > 0 else 1.0
    cf = tau_w / q_ref

    # Friction velocity
    u_tau = np.sqrt(np.abs(tau_w) / rho)

    return WallShearStress(
        x=x.copy(),
        tau_w=tau_w,
        cf=cf,
        u_tau=u_tau,
    )


def blasius_solution(
    eta: NDArray,
    u_inf: float = 1.0,
) -> Tuple[NDArray, NDArray]:
    """Return Blasius flat plate boundary layer solution.

    This is the analytical solution for laminar flow over a flat plate.

    Args:
        eta: Similarity variable (y * sqrt(U_inf / (nu * x))).
        u_inf: Freestream velocity.

    Returns:
        Tuple of (f_prime, f_double_prime) where:
            - f_prime = u / u_inf
            - f_double_prime is related to wall shear
    """
    # Blasius solution approximation using polynomial fit
    # f' = u/U_inf as function of eta
    f_prime = np.tanh(0.332057 * eta)  # Simplified approximation

    # More accurate for small eta
    mask = eta < 5
    if np.any(mask):
        # Use Blasius table data interpolation for accuracy
        # This is an approximation; real implementation would use ODE solution
        f_prime[mask] = 1 - np.exp(-0.332057 * eta[mask] ** 2 / 2)

    # Saturate to 1 for large eta
    f_prime = np.clip(f_prime, 0, 1)

    # f'' at wall is approximately 0.332
    f_double_prime = 0.332 * np.exp(-0.166 * eta**2)

    return u_inf * f_prime, f_double_prime
