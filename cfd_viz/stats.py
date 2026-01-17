"""Field statistics with optional cfd-python acceleration."""

from typing import Dict, List, Union

import numpy as np
from numpy.typing import NDArray

from .cfd_python_integration import get_cfd_python, has_cfd_python
from .common import VTKData


def calculate_field_stats(
    data: Union[NDArray, List[float]],
    use_cfd_python: bool = True,
) -> Dict[str, float]:
    """Calculate field statistics (min, max, avg, sum).

    Uses cfd-python's optimized implementation when available,
    falls back to NumPy otherwise.

    Args:
        data: Field data as array or flat list
        use_cfd_python: Whether to use cfd-python (if available)

    Returns:
        Dict with 'min', 'max', 'avg', 'sum' keys
    """
    if use_cfd_python and has_cfd_python():
        cfd = get_cfd_python()
        flat = data.flatten().tolist() if hasattr(data, "flatten") else list(data)
        return cfd.calculate_field_stats(flat)

    # NumPy fallback
    arr = np.asarray(data)
    return {
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "avg": float(np.nanmean(arr)),
        "sum": float(np.nansum(arr)),
    }


def compute_flow_statistics(
    data: VTKData,
    use_cfd_python: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Compute comprehensive flow statistics.

    Args:
        data: VTKData with u, v, and optionally p fields
        use_cfd_python: Whether to use cfd-python (if available)

    Returns:
        Dict with 'u', 'v', 'p', 'velocity_magnitude' sub-dicts,
        each containing 'min', 'max', 'avg', 'sum'

    Raises:
        ValueError: If u or v fields are missing from data
    """
    if data.u is None or data.v is None:
        raise ValueError("VTKData must have both u and v fields")

    if use_cfd_python and has_cfd_python():
        cfd = get_cfd_python()
        u_flat = data.u.flatten().tolist()
        v_flat = data.v.flatten().tolist()
        p_field = data.get("p")
        if p_field is not None:
            p_flat = p_field.flatten().tolist()
        else:
            # cfd-python requires p, use zeros as placeholder
            p_flat = [0.0] * len(u_flat)

        result = cfd.compute_flow_statistics(u_flat, v_flat, p_flat, data.nx, data.ny)

        # Remove p stats if original data had no pressure
        if p_field is None and "p" in result:
            del result["p"]

        return result

    # NumPy fallback
    vel_mag = np.sqrt(data.u**2 + data.v**2)
    result: Dict[str, Dict[str, float]] = {
        "u": calculate_field_stats(data.u, use_cfd_python=False),
        "v": calculate_field_stats(data.v, use_cfd_python=False),
        "velocity_magnitude": calculate_field_stats(vel_mag, use_cfd_python=False),
    }
    p_field = data.get("p")
    if p_field is not None:
        result["p"] = calculate_field_stats(p_field, use_cfd_python=False)
    return result


def compute_velocity_magnitude(
    data: VTKData,
    use_cfd_python: bool = True,
) -> NDArray:
    """Compute velocity magnitude field.

    Args:
        data: VTKData with u, v fields
        use_cfd_python: Whether to use cfd-python (if available)

    Returns:
        2D array of velocity magnitudes

    Raises:
        ValueError: If u or v fields are missing from data
    """
    if data.u is None or data.v is None:
        raise ValueError("VTKData must have both u and v fields")

    if use_cfd_python and has_cfd_python():
        cfd = get_cfd_python()
        u_flat = data.u.flatten().tolist()
        v_flat = data.v.flatten().tolist()
        mag_flat = cfd.compute_velocity_magnitude(u_flat, v_flat, data.nx, data.ny)
        return np.array(mag_flat).reshape((data.ny, data.nx))

    # NumPy fallback
    return np.sqrt(data.u**2 + data.v**2)
