"""
Velocity Field Computations
===========================

Pure functions for computing velocity-related quantities from CFD data.
All functions take numpy arrays as input and return numpy arrays.
"""

from typing import Tuple

import numpy as np
from numpy.typing import NDArray


def magnitude(u: NDArray, v: NDArray) -> NDArray:
    """Calculate velocity magnitude from u and v components.

    Args:
        u: X-velocity component (2D array)
        v: Y-velocity component (2D array)

    Returns:
        Velocity magnitude field (2D array)

    Example:
        >>> u = np.array([[1, 2], [3, 4]])
        >>> v = np.array([[0, 0], [0, 0]])
        >>> magnitude(u, v)
        array([[1., 2.],
               [3., 4.]])
    """
    return np.sqrt(u**2 + v**2)


def speed(u: NDArray, v: NDArray) -> NDArray:
    """Alias for magnitude(). Calculate speed from velocity components.

    Args:
        u: X-velocity component (2D array)
        v: Y-velocity component (2D array)

    Returns:
        Speed field (2D array)
    """
    return magnitude(u, v)


def normalize(u: NDArray, v: NDArray) -> Tuple[NDArray, NDArray]:
    """Normalize velocity vectors to unit length.

    Args:
        u: X-velocity component (2D array)
        v: Y-velocity component (2D array)

    Returns:
        Tuple of (u_normalized, v_normalized) unit vectors.
        Where magnitude is zero, returns (0, 0).
    """
    mag = magnitude(u, v)
    # Avoid division by zero
    mag_safe = np.where(mag > 0, mag, 1.0)
    u_norm = np.where(mag > 0, u / mag_safe, 0.0)
    v_norm = np.where(mag > 0, v / mag_safe, 0.0)
    return u_norm, v_norm


def angle(u: NDArray, v: NDArray) -> NDArray:
    """Calculate velocity angle (direction) in radians.

    Args:
        u: X-velocity component (2D array)
        v: Y-velocity component (2D array)

    Returns:
        Angle in radians, measured counter-clockwise from positive x-axis.
        Range: [-pi, pi]
    """
    return np.arctan2(v, u)


def angle_degrees(u: NDArray, v: NDArray) -> NDArray:
    """Calculate velocity angle (direction) in degrees.

    Args:
        u: X-velocity component (2D array)
        v: Y-velocity component (2D array)

    Returns:
        Angle in degrees, measured counter-clockwise from positive x-axis.
        Range: [-180, 180]
    """
    return np.degrees(angle(u, v))


def components_from_magnitude_angle(
    mag: NDArray, theta: NDArray
) -> Tuple[NDArray, NDArray]:
    """Convert magnitude and angle to u, v components.

    Args:
        mag: Velocity magnitude (2D array)
        theta: Angle in radians (2D array)

    Returns:
        Tuple of (u, v) velocity components
    """
    u = mag * np.cos(theta)
    v = mag * np.sin(theta)
    return u, v


def fluctuations(
    u: NDArray, v: NDArray, axis: int = 1
) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """Calculate velocity fluctuations from mean flow.

    Computes u' = u - <u> and v' = v - <v> where <> denotes
    averaging along the specified axis.

    Args:
        u: X-velocity component (2D array)
        v: Y-velocity component (2D array)
        axis: Axis along which to compute mean (default: 1, streamwise)

    Returns:
        Tuple of (u_mean, v_mean, u_prime, v_prime)
    """
    u_mean = np.mean(u, axis=axis, keepdims=True)
    v_mean = np.mean(v, axis=axis, keepdims=True)
    u_prime = u - u_mean
    v_prime = v - v_mean
    return u_mean, v_mean, u_prime, v_prime


def turbulent_intensity(u: NDArray, v: NDArray, axis: int = 1) -> NDArray:
    """Calculate turbulent intensity.

    TI = sqrt(u'^2 + v'^2) / U_mean

    Args:
        u: X-velocity component (2D array)
        v: Y-velocity component (2D array)
        axis: Axis along which to compute statistics

    Returns:
        Turbulent intensity field
    """
    u_mean, v_mean, u_prime, v_prime = fluctuations(u, v, axis)
    U_mean = np.sqrt(u_mean**2 + v_mean**2)
    rms_fluct = np.sqrt(u_prime**2 + v_prime**2)
    # Avoid division by zero
    U_mean_safe = np.where(U_mean > 0, U_mean, 1.0)
    return np.where(U_mean > 0, rms_fluct / U_mean_safe, 0.0)
