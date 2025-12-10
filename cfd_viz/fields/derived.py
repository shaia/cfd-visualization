"""
Derived Field Computations
==========================

Pure functions for computing derived quantities from CFD data.
Includes energy, pressure coefficients, and statistical measures.
All functions take numpy arrays as input and return numpy arrays.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray


@dataclass
class FlowStatistics:
    """Container for flow field statistics."""

    max_velocity: float
    mean_velocity: float
    min_velocity: float
    velocity_std: float

    max_pressure: float
    mean_pressure: float
    min_pressure: float
    pressure_std: float

    total_kinetic_energy: float
    mean_kinetic_energy: float

    max_vorticity: float
    mean_vorticity: float

    velocity_uniformity: float  # Coefficient of variation

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "max_velocity": self.max_velocity,
            "mean_velocity": self.mean_velocity,
            "min_velocity": self.min_velocity,
            "velocity_std": self.velocity_std,
            "max_pressure": self.max_pressure,
            "mean_pressure": self.mean_pressure,
            "min_pressure": self.min_pressure,
            "pressure_std": self.pressure_std,
            "total_kinetic_energy": self.total_kinetic_energy,
            "mean_kinetic_energy": self.mean_kinetic_energy,
            "max_vorticity": self.max_vorticity,
            "mean_vorticity": self.mean_vorticity,
            "velocity_uniformity": self.velocity_uniformity,
        }


def kinetic_energy(u: NDArray, v: NDArray, rho: float = 1.0) -> NDArray:
    """Calculate kinetic energy density field.

    KE = 0.5 * rho * (u^2 + v^2)

    Args:
        u: X-velocity component (2D array)
        v: Y-velocity component (2D array)
        rho: Density (default: 1.0 for non-dimensional)

    Returns:
        Kinetic energy density field (2D array)
    """
    return 0.5 * rho * (u**2 + v**2)


def total_kinetic_energy(
    u: NDArray, v: NDArray, dx: float, dy: float, rho: float = 1.0
) -> float:
    """Calculate total kinetic energy in the domain.

    TKE = integral of 0.5 * rho * |v|^2 dA

    Args:
        u: X-velocity component (2D array)
        v: Y-velocity component (2D array)
        dx: Grid spacing in x-direction
        dy: Grid spacing in y-direction
        rho: Density (default: 1.0)

    Returns:
        Total kinetic energy (scalar)
    """
    ke = kinetic_energy(u, v, rho)
    return float(np.sum(ke) * dx * dy)


def dynamic_pressure(u: NDArray, v: NDArray, rho: float = 1.0) -> NDArray:
    """Calculate dynamic pressure field.

    q = 0.5 * rho * (u^2 + v^2)

    Args:
        u: X-velocity component (2D array)
        v: Y-velocity component (2D array)
        rho: Density (default: 1.0)

    Returns:
        Dynamic pressure field (2D array)
    """
    return kinetic_energy(u, v, rho)


def pressure_coefficient(
    p: NDArray, p_inf: float, rho_inf: float, U_inf: float
) -> NDArray:
    """Calculate pressure coefficient.

    Cp = (p - p_inf) / (0.5 * rho_inf * U_inf^2)

    Args:
        p: Static pressure field (2D array)
        p_inf: Freestream pressure
        rho_inf: Freestream density
        U_inf: Freestream velocity

    Returns:
        Pressure coefficient field (2D array)
    """
    q_inf = 0.5 * rho_inf * U_inf**2
    if q_inf == 0:
        return np.zeros_like(p)
    return (p - p_inf) / q_inf


def total_pressure(p: NDArray, u: NDArray, v: NDArray, rho: float = 1.0) -> NDArray:
    """Calculate total (stagnation) pressure field.

    p_0 = p + 0.5 * rho * (u^2 + v^2)

    Args:
        p: Static pressure field (2D array)
        u: X-velocity component (2D array)
        v: Y-velocity component (2D array)
        rho: Density (default: 1.0)

    Returns:
        Total pressure field (2D array)
    """
    return p + dynamic_pressure(u, v, rho)


def mach_number(u: NDArray, v: NDArray, a: float) -> NDArray:
    """Calculate Mach number field.

    M = |v| / a

    Args:
        u: X-velocity component (2D array)
        v: Y-velocity component (2D array)
        a: Speed of sound

    Returns:
        Mach number field (2D array)
    """
    velocity_mag = np.sqrt(u**2 + v**2)
    return velocity_mag / a


def reynolds_number_local(u: NDArray, v: NDArray, x: NDArray, nu: float) -> NDArray:
    """Calculate local Reynolds number based on x-distance.

    Re_x = |v| * x / nu

    Args:
        u: X-velocity component (2D array)
        v: Y-velocity component (2D array)
        x: X-coordinate array (will be broadcast)
        nu: Kinematic viscosity

    Returns:
        Local Reynolds number field (2D array)
    """
    velocity_mag = np.sqrt(u**2 + v**2)
    # Broadcast x to match velocity shape
    if x.ndim == 1:
        x_2d = np.broadcast_to(x, u.shape)
    else:
        x_2d = x
    return velocity_mag * x_2d / nu


def stream_function(u: NDArray, v: NDArray, dx: float, dy: float) -> NDArray:
    """Calculate stream function from velocity field.

    For incompressible 2D flow:
    u = d(psi)/dy
    v = -d(psi)/dx

    Computed by integrating v along x (then correcting with u along y).

    Args:
        u: X-velocity component (2D array)
        v: Y-velocity component (2D array)
        dx: Grid spacing in x-direction
        dy: Grid spacing in y-direction

    Returns:
        Stream function field (2D array)
    """
    ny, nx = u.shape

    # Initialize stream function
    psi = np.zeros_like(u)

    # Integrate -v along x direction (psi increases with y for positive u)
    for i in range(1, nx):
        psi[:, i] = psi[:, i - 1] - v[:, i] * dx

    # Average with integration of u along y for consistency
    psi2 = np.zeros_like(u)
    for j in range(1, ny):
        psi2[j, :] = psi2[j - 1, :] + u[j, :] * dy

    # Return average of both integrations
    return 0.5 * (psi + psi2)


def helicity_density(
    u: NDArray, v: NDArray, omega: NDArray, w: Optional[NDArray] = None
) -> NDArray:
    """Calculate helicity density (for 2D, this is simplified).

    H = v dot omega

    For 2D flow with omega only in z-direction, and assuming w=0:
    H = 0 (helicity is identically zero in 2D)

    This function is included for completeness and 3D extension.

    Args:
        u: X-velocity component (2D array)
        v: Y-velocity component (2D array)
        omega: Vorticity (z-component for 2D) (2D array)
        w: Z-velocity component (optional, default: 0)

    Returns:
        Helicity density field (2D array), zero for pure 2D flow
    """
    if w is None:
        # Pure 2D flow: helicity is zero
        return np.zeros_like(u)
    else:
        # Quasi-3D: H = w * omega_z
        return w * omega


def dissipation_rate(
    u: NDArray, v: NDArray, dx: float, dy: float, nu: float
) -> NDArray:
    """Calculate viscous dissipation rate.

    epsilon = 2 * nu * S_ij * S_ij

    Args:
        u: X-velocity component (2D array)
        v: Y-velocity component (2D array)
        dx: Grid spacing in x-direction
        dy: Grid spacing in y-direction
        nu: Kinematic viscosity

    Returns:
        Dissipation rate field (2D array)
    """
    from .gradients import strain_rate_tensor

    S = strain_rate_tensor(u, v, dx, dy)

    # S_ij * S_ij = S11^2 + 2*S12^2 + S22^2 (for 2D symmetric tensor)
    S_squared = S.S11**2 + 2 * S.S12**2 + S.S22**2

    return 2 * nu * S_squared


def calculate_flow_statistics(
    u: NDArray,
    v: NDArray,
    p: Optional[NDArray],
    dx: float,
    dy: float,
) -> FlowStatistics:
    """Calculate comprehensive flow statistics.

    Args:
        u: X-velocity component (2D array)
        v: Y-velocity component (2D array)
        p: Pressure field (2D array), or None
        dx: Grid spacing in x-direction
        dy: Grid spacing in y-direction

    Returns:
        FlowStatistics dataclass with all computed statistics
    """
    from .vorticity import vorticity

    # Velocity statistics
    velocity_mag = np.sqrt(u**2 + v**2)
    max_vel = float(np.max(velocity_mag))
    mean_vel = float(np.mean(velocity_mag))
    min_vel = float(np.min(velocity_mag))
    std_vel = float(np.std(velocity_mag))

    # Pressure statistics
    if p is not None:
        max_p = float(np.max(p))
        mean_p = float(np.mean(p))
        min_p = float(np.min(p))
        std_p = float(np.std(p))
    else:
        max_p = mean_p = min_p = std_p = 0.0

    # Energy statistics
    ke = kinetic_energy(u, v)
    total_ke = float(np.sum(ke) * dx * dy)
    mean_ke = float(np.mean(ke))

    # Vorticity statistics
    omega = vorticity(u, v, dx, dy)
    max_vort = float(np.max(np.abs(omega)))
    mean_vort = float(np.mean(np.abs(omega)))

    # Velocity uniformity (coefficient of variation)
    uniformity = std_vel / (mean_vel + 1e-10)

    return FlowStatistics(
        max_velocity=max_vel,
        mean_velocity=mean_vel,
        min_velocity=min_vel,
        velocity_std=std_vel,
        max_pressure=max_p,
        mean_pressure=mean_p,
        min_pressure=min_p,
        pressure_std=std_p,
        total_kinetic_energy=total_ke,
        mean_kinetic_energy=mean_ke,
        max_vorticity=max_vort,
        mean_vorticity=mean_vort,
        velocity_uniformity=uniformity,
    )
