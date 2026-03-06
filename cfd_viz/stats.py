"""Field statistics with pluggable backend (cfd-python or NumPy)."""

from typing import Dict, List, Union

from numpy.typing import NDArray

from .backends import get_stats_backend, validate_field_stats, validate_flow_statistics
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
    backend = get_stats_backend(use_cfd_python)
    return validate_field_stats(backend.calculate_field_stats(data))


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

    backend = get_stats_backend(use_cfd_python)
    p_field = data.get("p")
    return validate_flow_statistics(
        backend.compute_flow_statistics(data.u, data.v, data.nx, data.ny, p_field)
    )


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

    backend = get_stats_backend(use_cfd_python)
    return backend.compute_velocity_magnitude(data.u, data.v, data.nx, data.ny)
