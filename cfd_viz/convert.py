"""Conversion utilities for cfd-python integration."""

from typing import Any, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

from .common import VTKData


def from_cfd_python(
    u: List[float],
    v: List[float],
    nx: int,
    ny: int,
    p: Optional[List[float]] = None,
    xmin: float = 0.0,
    xmax: float = 1.0,
    ymin: float = 0.0,
    ymax: float = 1.0,
) -> VTKData:
    """Convert cfd_python simulation results to VTKData for visualization.

    Args:
        u: Flat list of u-velocity values (row-major order)
        v: Flat list of v-velocity values
        nx: Number of grid points in x
        ny: Number of grid points in y
        p: Flat list of pressure values (optional)
        xmin, xmax, ymin, ymax: Domain bounds

    Returns:
        VTKData object ready for visualization

    Example:
        >>> result = cfd_python.run_simulation_with_params(nx=50, ny=50, ...)
        >>> data = from_cfd_python(
        ...     result['u'], result['v'],
        ...     result['nx'], result['ny'],
        ...     p=result.get('p')
        ... )
        >>> plot_velocity_field(data.X, data.Y, data.u, data.v)
    """
    if nx <= 0 or ny <= 0:
        raise ValueError(f"Invalid grid dimensions: {nx}x{ny}")

    expected_size = nx * ny
    if len(u) != expected_size:
        raise ValueError(f"u has {len(u)} elements, expected {expected_size}")
    if len(v) != expected_size:
        raise ValueError(f"v has {len(v)} elements, expected {expected_size}")

    dx = (xmax - xmin) / (nx - 1) if nx > 1 else 1.0
    dy = (ymax - ymin) / (ny - 1) if ny > 1 else 1.0

    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y)

    fields: Dict[str, NDArray] = {
        "u": np.array(u).reshape((ny, nx)),
        "v": np.array(v).reshape((ny, nx)),
    }
    if p is not None:
        if len(p) != expected_size:
            raise ValueError(f"p has {len(p)} elements, expected {expected_size}")
        fields["p"] = np.array(p).reshape((ny, nx))

    return VTKData(
        x=x,
        y=y,
        X=X,
        Y=Y,
        fields=fields,
        nx=nx,
        ny=ny,
        dx=dx,
        dy=dy,
    )


def from_simulation_result(result: Dict[str, Any]) -> VTKData:
    """Convert cfd_python simulation result dict to VTKData.

    Args:
        result: Dictionary returned by run_simulation_with_params()

    Returns:
        VTKData object ready for visualization

    Example:
        >>> result = cfd_python.run_simulation_with_params(nx=50, ny=50, steps=100)
        >>> data = from_simulation_result(result)
        >>> quick_plot(data)
    """
    return from_cfd_python(
        u=result["u"],
        v=result["v"],
        nx=result["nx"],
        ny=result["ny"],
        p=result.get("p"),
        xmin=result.get("xmin", 0.0),
        xmax=result.get("xmax", 1.0),
        ymin=result.get("ymin", 0.0),
        ymax=result.get("ymax", 1.0),
    )


def to_cfd_python(data: VTKData) -> Dict[str, Any]:
    """Convert VTKData to cfd_python-compatible dictionary.

    Args:
        data: VTKData object

    Returns:
        Dictionary with flat lists compatible with cfd_python functions

    Example:
        >>> data = read_vtk_file("simulation.vtk")
        >>> result = to_cfd_python(data)
        >>> # Use with cfd_python functions
    """
    result: Dict[str, Any] = {
        "u": data.u.flatten().tolist() if data.u is not None else [],
        "v": data.v.flatten().tolist() if data.v is not None else [],
        "p": data.get("p").flatten().tolist() if data.get("p") is not None else None,
        "nx": data.nx,
        "ny": data.ny,
        "xmin": float(data.x.min()) if len(data.x) > 0 else 0.0,
        "xmax": float(data.x.max()) if len(data.x) > 0 else 1.0,
        "ymin": float(data.y.min()) if len(data.y) > 0 else 0.0,
        "ymax": float(data.y.max()) if len(data.y) > 0 else 1.0,
    }
    return result
