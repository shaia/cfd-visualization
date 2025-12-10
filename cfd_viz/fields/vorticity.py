"""
Vorticity Field Computations
============================

Pure functions for computing vorticity-related quantities from CFD data.
All functions take numpy arrays as input and return numpy arrays.
"""

from typing import Tuple

import numpy as np
from numpy.typing import NDArray


def vorticity(u: NDArray, v: NDArray, dx: float, dy: float) -> NDArray:
    """Calculate vorticity field (z-component of curl of velocity).

    For 2D flow, vorticity omega_z = dv/dx - du/dy

    Args:
        u: X-velocity component (2D array)
        v: Y-velocity component (2D array)
        dx: Grid spacing in x-direction
        dy: Grid spacing in y-direction

    Returns:
        Vorticity field (2D array)
    """
    du_dy = np.gradient(u, dy, axis=0)
    dv_dx = np.gradient(v, dx, axis=1)
    return dv_dx - du_dy


def vorticity_from_gradients(du_dy: NDArray, dv_dx: NDArray) -> NDArray:
    """Calculate vorticity from pre-computed velocity gradients.

    Args:
        du_dy: Gradient of u with respect to y
        dv_dx: Gradient of v with respect to x

    Returns:
        Vorticity field (2D array)
    """
    return dv_dx - du_dy


def q_criterion(u: NDArray, v: NDArray, dx: float, dy: float) -> NDArray:
    """Calculate Q-criterion for vortex identification.

    Q = 0.5 * (||Omega||^2 - ||S||^2)

    where Omega is the rotation rate tensor and S is the strain rate tensor.
    Positive Q indicates regions where rotation dominates over strain (vortices).

    Args:
        u: X-velocity component (2D array)
        v: Y-velocity component (2D array)
        dx: Grid spacing in x-direction
        dy: Grid spacing in y-direction

    Returns:
        Q-criterion field (2D array). Positive values indicate vortex cores.
    """
    # Calculate velocity gradients
    du_dx = np.gradient(u, dx, axis=1)
    du_dy = np.gradient(u, dy, axis=0)
    dv_dx = np.gradient(v, dx, axis=1)
    dv_dy = np.gradient(v, dy, axis=0)

    # Strain rate tensor components (symmetric part)
    S11 = du_dx
    S12 = 0.5 * (du_dy + dv_dx)
    S22 = dv_dy

    # Rotation rate tensor components (antisymmetric part)
    # For 2D: Omega_12 = 0.5 * (du_dy - dv_dx)
    O12 = 0.5 * (du_dy - dv_dx)

    # ||Omega||^2 = 2 * Omega_12^2 (for 2D)
    omega_squared = 2 * O12**2

    # ||S||^2 = 2 * (S11^2 + 2*S12^2 + S22^2) (for 2D symmetric tensor)
    strain_squared = 2 * (S11**2 + 2 * S12**2 + S22**2)

    # Q-criterion
    Q = 0.5 * (omega_squared - strain_squared)

    return Q


def lambda2_criterion(u: NDArray, v: NDArray, dx: float, dy: float) -> NDArray:
    """Calculate lambda2 criterion for vortex identification.

    Lambda2 is the second eigenvalue of (S^2 + Omega^2).
    Negative lambda2 indicates vortex cores.

    For 2D, this simplifies to a scalar calculation.

    Args:
        u: X-velocity component (2D array)
        v: Y-velocity component (2D array)
        dx: Grid spacing in x-direction
        dy: Grid spacing in y-direction

    Returns:
        Lambda2 field (2D array). Negative values indicate vortex cores.
    """
    # Calculate velocity gradients
    du_dx = np.gradient(u, dx, axis=1)
    du_dy = np.gradient(u, dy, axis=0)
    dv_dx = np.gradient(v, dx, axis=1)
    dv_dy = np.gradient(v, dy, axis=0)

    # For 2D, the velocity gradient tensor is:
    # L = [[du_dx, du_dy],
    #      [dv_dx, dv_dy]]
    #
    # S = 0.5 * (L + L^T)  (strain rate)
    # Omega = 0.5 * (L - L^T)  (rotation rate)
    #
    # S^2 + Omega^2 for 2D simplifies to finding eigenvalues of a 2x2 matrix

    S11 = du_dx
    S12 = 0.5 * (du_dy + dv_dx)
    S22 = dv_dy

    O12 = 0.5 * (du_dy - dv_dx)

    # Components of M = S^2 + Omega^2
    # M11 = S11^2 + S12^2 - O12^2
    # M12 = S11*S12 + S12*S22
    # M22 = S12^2 + S22^2 - O12^2

    M11 = S11**2 + S12**2 - O12**2
    M12 = S11 * S12 + S12 * S22
    M22 = S12**2 + S22**2 - O12**2

    # Eigenvalues of 2x2 symmetric matrix
    # lambda = 0.5 * (trace +/- sqrt(trace^2 - 4*det))
    trace = M11 + M22
    det = M11 * M22 - M12**2

    discriminant = trace**2 - 4 * det
    discriminant = np.maximum(discriminant, 0)  # Numerical safety

    sqrt_disc = np.sqrt(discriminant)
    lambda1 = 0.5 * (trace + sqrt_disc)
    lambda2 = 0.5 * (trace - sqrt_disc)

    # Return the smaller eigenvalue (lambda2)
    return np.minimum(lambda1, lambda2)


def enstrophy(u: NDArray, v: NDArray, dx: float, dy: float) -> NDArray:
    """Calculate enstrophy (squared vorticity).

    Enstrophy = 0.5 * omega^2

    Args:
        u: X-velocity component (2D array)
        v: Y-velocity component (2D array)
        dx: Grid spacing in x-direction
        dy: Grid spacing in y-direction

    Returns:
        Enstrophy field (2D array)
    """
    omega = vorticity(u, v, dx, dy)
    return 0.5 * omega**2


def circulation(
    u: NDArray,
    v: NDArray,
    x: NDArray,
    y: NDArray,
    center: Tuple[float, float],
    radius: float,
    num_points: int = 100,
) -> float:
    """Calculate circulation around a circular path.

    Circulation Gamma = integral of v dot dl around closed path.

    Args:
        u: X-velocity component (2D array)
        v: Y-velocity component (2D array)
        x: 1D array of x-coordinates
        y: 1D array of y-coordinates
        center: (x, y) center of circular path
        radius: Radius of circular path
        num_points: Number of points to discretize path

    Returns:
        Circulation value (scalar)
    """
    from scipy.interpolate import RegularGridInterpolator

    # Create circular path
    theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    path_x = center[0] + radius * np.cos(theta)
    path_y = center[1] + radius * np.sin(theta)

    # Create interpolators
    interp_u = RegularGridInterpolator((y, x), u, bounds_error=False, fill_value=0)
    interp_v = RegularGridInterpolator((y, x), v, bounds_error=False, fill_value=0)

    # Get velocity along path (note: interpolator expects (y, x) order)
    path_points = np.column_stack([path_y, path_x])
    u_path = interp_u(path_points)
    v_path = interp_v(path_points)

    # Calculate tangent vectors (dl = (dx, dy))
    dx_dt = -radius * np.sin(theta)
    dy_dt = radius * np.cos(theta)

    # Circulation = integral of v dot dl
    # Using trapezoidal rule with periodic boundary
    d_theta = 2 * np.pi / num_points
    integrand = u_path * dx_dt + v_path * dy_dt
    gamma = np.sum(integrand) * d_theta

    return float(gamma)


def detect_vortex_cores(
    omega: NDArray,
    Q: NDArray,
    omega_threshold_factor: float = 0.1,
    Q_threshold_factor: float = 0.1,
) -> NDArray:
    """Detect vortex cores using combined vorticity and Q-criterion.

    Args:
        omega: Vorticity field (2D array)
        Q: Q-criterion field (2D array)
        omega_threshold_factor: Fraction of max vorticity for threshold
        Q_threshold_factor: Fraction of max Q for threshold

    Returns:
        Boolean mask of vortex core regions (2D array)
    """
    # Normalize Q-criterion
    Q_max = np.max(np.abs(Q))
    Q_normalized = Q / Q_max if Q_max > 0 else Q

    # Calculate thresholds
    omega_threshold = omega_threshold_factor * np.max(np.abs(omega))
    Q_threshold = Q_threshold_factor

    # Vortex cores: high Q AND high vorticity
    vortex_cores = (Q_normalized > Q_threshold) & (np.abs(omega) > omega_threshold)

    return vortex_cores
