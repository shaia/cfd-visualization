"""
Velocity Gradient Computations
==============================

Pure functions for computing velocity gradients and related tensors.
All functions take numpy arrays as input and return numpy arrays.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class VelocityGradients:
    """Container for velocity gradient tensor components.

    For 2D flow, the velocity gradient tensor L is:
    L = [[du_dx, du_dy],
         [dv_dx, dv_dy]]
    """

    du_dx: NDArray
    du_dy: NDArray
    dv_dx: NDArray
    dv_dy: NDArray

    @property
    def divergence(self) -> NDArray:
        """Velocity divergence (should be ~0 for incompressible flow)."""
        return self.du_dx + self.dv_dy

    @property
    def vorticity(self) -> NDArray:
        """Z-component of vorticity."""
        return self.dv_dx - self.du_dy


@dataclass
class StrainRateTensor:
    """Container for strain rate tensor components.

    The strain rate tensor S = 0.5 * (L + L^T) is symmetric:
    S = [[S11, S12],
         [S12, S22]]
    """

    S11: NDArray  # du_dx
    S12: NDArray  # 0.5 * (du_dy + dv_dx)
    S22: NDArray  # dv_dy

    @property
    def magnitude(self) -> NDArray:
        """Frobenius norm of strain rate tensor."""
        return np.sqrt(2 * (self.S11**2 + 2 * self.S12**2 + self.S22**2))

    @property
    def principal_strain_max(self) -> NDArray:
        """Maximum principal strain rate."""
        trace = self.S11 + self.S22
        det = self.S11 * self.S22 - self.S12**2
        discriminant = np.maximum(trace**2 - 4 * det, 0)
        return 0.5 * (trace + np.sqrt(discriminant))

    @property
    def principal_strain_min(self) -> NDArray:
        """Minimum principal strain rate."""
        trace = self.S11 + self.S22
        det = self.S11 * self.S22 - self.S12**2
        discriminant = np.maximum(trace**2 - 4 * det, 0)
        return 0.5 * (trace - np.sqrt(discriminant))


def velocity_gradients(
    u: NDArray, v: NDArray, dx: float, dy: float
) -> VelocityGradients:
    """Calculate all components of the velocity gradient tensor.

    Args:
        u: X-velocity component (2D array)
        v: Y-velocity component (2D array)
        dx: Grid spacing in x-direction
        dy: Grid spacing in y-direction

    Returns:
        VelocityGradients dataclass with all tensor components
    """
    du_dx = np.gradient(u, dx, axis=1)
    du_dy = np.gradient(u, dy, axis=0)
    dv_dx = np.gradient(v, dx, axis=1)
    dv_dy = np.gradient(v, dy, axis=0)

    return VelocityGradients(du_dx=du_dx, du_dy=du_dy, dv_dx=dv_dx, dv_dy=dv_dy)


def strain_rate_tensor(
    u: NDArray, v: NDArray, dx: float, dy: float
) -> StrainRateTensor:
    """Calculate the strain rate tensor S = 0.5 * (L + L^T).

    Args:
        u: X-velocity component (2D array)
        v: Y-velocity component (2D array)
        dx: Grid spacing in x-direction
        dy: Grid spacing in y-direction

    Returns:
        StrainRateTensor dataclass with symmetric tensor components
    """
    grads = velocity_gradients(u, v, dx, dy)

    S11 = grads.du_dx
    S12 = 0.5 * (grads.du_dy + grads.dv_dx)
    S22 = grads.dv_dy

    return StrainRateTensor(S11=S11, S12=S12, S22=S22)


def strain_rate_components(
    u: NDArray, v: NDArray, dx: float, dy: float
) -> Tuple[NDArray, NDArray, NDArray]:
    """Calculate strain rate tensor components as tuple.

    Convenience function for direct unpacking.

    Args:
        u: X-velocity component (2D array)
        v: Y-velocity component (2D array)
        dx: Grid spacing in x-direction
        dy: Grid spacing in y-direction

    Returns:
        Tuple of (S11, S12, S22) strain rate components
    """
    S = strain_rate_tensor(u, v, dx, dy)
    return S.S11, S.S12, S.S22


def divergence(u: NDArray, v: NDArray, dx: float, dy: float) -> NDArray:
    """Calculate velocity divergence.

    For incompressible flow, div(v) = du/dx + dv/dy should be approximately zero.

    Args:
        u: X-velocity component (2D array)
        v: Y-velocity component (2D array)
        dx: Grid spacing in x-direction
        dy: Grid spacing in y-direction

    Returns:
        Divergence field (2D array)
    """
    du_dx = np.gradient(u, dx, axis=1)
    dv_dy = np.gradient(v, dy, axis=0)
    return du_dx + dv_dy


def shear_strain_rate(u: NDArray, v: NDArray, dx: float, dy: float) -> NDArray:
    """Calculate shear strain rate (off-diagonal component).

    Shear strain rate = 0.5 * (du/dy + dv/dx)

    Args:
        u: X-velocity component (2D array)
        v: Y-velocity component (2D array)
        dx: Grid spacing in x-direction
        dy: Grid spacing in y-direction

    Returns:
        Shear strain rate field (2D array)
    """
    du_dy = np.gradient(u, dy, axis=0)
    dv_dx = np.gradient(v, dx, axis=1)
    return 0.5 * (du_dy + dv_dx)


def normal_strain_rates(
    u: NDArray, v: NDArray, dx: float, dy: float
) -> Tuple[NDArray, NDArray]:
    """Calculate normal strain rates (diagonal components).

    Args:
        u: X-velocity component (2D array)
        v: Y-velocity component (2D array)
        dx: Grid spacing in x-direction
        dy: Grid spacing in y-direction

    Returns:
        Tuple of (du/dx, dv/dy) normal strain rates
    """
    du_dx = np.gradient(u, dx, axis=1)
    dv_dy = np.gradient(v, dy, axis=0)
    return du_dx, dv_dy


def strain_rate_magnitude(u: NDArray, v: NDArray, dx: float, dy: float) -> NDArray:
    """Calculate magnitude of strain rate tensor.

    |S| = sqrt(2 * S_ij * S_ij)

    Args:
        u: X-velocity component (2D array)
        v: Y-velocity component (2D array)
        dx: Grid spacing in x-direction
        dy: Grid spacing in y-direction

    Returns:
        Strain rate magnitude field (2D array)
    """
    S = strain_rate_tensor(u, v, dx, dy)
    return S.magnitude


def rotation_rate(u: NDArray, v: NDArray, dx: float, dy: float) -> NDArray:
    """Calculate rotation rate (antisymmetric part of velocity gradient).

    For 2D, Omega_12 = 0.5 * (du/dy - dv/dx) = -0.5 * vorticity

    Args:
        u: X-velocity component (2D array)
        v: Y-velocity component (2D array)
        dx: Grid spacing in x-direction
        dy: Grid spacing in y-direction

    Returns:
        Rotation rate field (2D array)
    """
    du_dy = np.gradient(u, dy, axis=0)
    dv_dx = np.gradient(v, dx, axis=1)
    return 0.5 * (du_dy - dv_dx)


def wall_shear_stress(
    u: NDArray, v: NDArray, dy: float, mu: float, wall_index: int = 0
) -> NDArray:
    """Calculate wall shear stress at a horizontal wall.

    tau_w = mu * (du/dy)|_wall

    Args:
        u: X-velocity component (2D array)
        v: Y-velocity component (2D array)
        dy: Grid spacing in y-direction
        mu: Dynamic viscosity
        wall_index: Row index of the wall (0 for bottom, -1 for top)

    Returns:
        Wall shear stress along the wall (1D array)
    """
    # Use one-sided difference at wall
    if wall_index == 0:
        # Bottom wall: forward difference
        du_dy_wall = (u[1, :] - u[0, :]) / dy
    elif wall_index == -1:
        # Top wall: backward difference
        du_dy_wall = (u[-1, :] - u[-2, :]) / dy
    else:
        # Interior "wall": central difference
        du_dy_wall = (u[wall_index + 1, :] - u[wall_index - 1, :]) / (2 * dy)

    return mu * du_dy_wall
