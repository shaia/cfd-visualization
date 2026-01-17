"""Quick visualization functions for cfd-python results.

One-liner visualization of simulation results with sensible defaults.
"""

from typing import Any, Literal, Optional

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .convert import from_cfd_python, from_simulation_result
from .fields import magnitude, vorticity
from .plotting import plot_contour_field

FieldType = Literal["velocity_magnitude", "vorticity", "u", "v", "p"]


def quick_plot(
    u: list[float],
    v: list[float],
    nx: int,
    ny: int,
    field: FieldType = "velocity_magnitude",
    p: Optional[list[float]] = None,
    xmin: float = 0.0,
    xmax: float = 1.0,
    ymin: float = 0.0,
    ymax: float = 1.0,
    ax: Optional[Axes] = None,
    figsize: tuple[float, float] = (8, 6),
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Quick visualization of cfd-python simulation results.

    Args:
        u: Flat list of u-velocity values from cfd-python
        v: Flat list of v-velocity values from cfd-python
        nx: Number of grid points in x
        ny: Number of grid points in y
        field: Field to plot - "velocity_magnitude", "vorticity", "u", "v", "p"
        p: Flat list of pressure values (required if field="p")
        xmin, xmax, ymin, ymax: Domain bounds
        ax: Matplotlib axes to plot on (created if None)
        figsize: Figure size if creating new figure
        **kwargs: Additional arguments passed to plot_contour_field

    Returns:
        Tuple of (figure, axes)

    Raises:
        ValueError: If field="p" but pressure is not provided
        ValueError: If unknown field type

    Example:
        >>> import cfd_python
        >>> result = cfd_python.run_simulation_with_params(nx=50, ny=50, steps=500)
        >>> fig, ax = quick_plot(result['u'], result['v'], result['nx'], result['ny'])
        >>> plt.show()
    """
    data = from_cfd_python(
        u, v, nx=nx, ny=ny, p=p, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if field == "velocity_magnitude":
        field_data = magnitude(data.u, data.v)
        title = kwargs.pop("title", "Velocity Magnitude")
        colorbar_label = kwargs.pop("colorbar_label", "|V|")
    elif field == "vorticity":
        field_data = vorticity(data.u, data.v, data.dx, data.dy)
        title = kwargs.pop("title", "Vorticity")
        colorbar_label = kwargs.pop("colorbar_label", "ω")
    elif field == "u":
        field_data = data.u
        title = kwargs.pop("title", "U Velocity")
        colorbar_label = kwargs.pop("colorbar_label", "u")
    elif field == "v":
        field_data = data.v
        title = kwargs.pop("title", "V Velocity")
        colorbar_label = kwargs.pop("colorbar_label", "v")
    elif field == "p":
        p_field = data.get("p")
        if p_field is None:
            raise ValueError("Pressure field required but not provided")
        field_data = p_field
        title = kwargs.pop("title", "Pressure")
        colorbar_label = kwargs.pop("colorbar_label", "p")
    else:
        raise ValueError(f"Unknown field: {field}")

    plot_contour_field(
        data.X,
        data.Y,
        field_data,
        ax=ax,
        title=title,
        colorbar_label=colorbar_label,
        **kwargs,
    )

    return fig, ax


def quick_plot_result(
    result: dict[str, Any],
    field: FieldType = "velocity_magnitude",
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Quick visualization of run_simulation_with_params() result.

    Convenience wrapper around quick_plot that extracts values from
    the simulation result dictionary.

    Args:
        result: Dictionary returned by cfd_python.run_simulation_with_params()
        field: Field to plot - "velocity_magnitude", "vorticity", "u", "v", "p"
        **kwargs: Additional arguments passed to quick_plot

    Returns:
        Tuple of (figure, axes)

    Example:
        >>> import cfd_python
        >>> result = cfd_python.run_simulation_with_params(nx=50, ny=50, steps=500)
        >>> fig, ax = quick_plot_result(result, field="vorticity")
        >>> plt.show()
    """
    return quick_plot(
        u=result["u"],
        v=result["v"],
        nx=result["nx"],
        ny=result["ny"],
        field=field,
        p=result.get("p"),
        xmin=result.get("xmin", 0.0),
        xmax=result.get("xmax", 1.0),
        ymin=result.get("ymin", 0.0),
        ymax=result.get("ymax", 1.0),
        **kwargs,
    )


def quick_plot_data(
    data: Any,
    field: FieldType = "velocity_magnitude",
    ax: Optional[Axes] = None,
    figsize: tuple[float, float] = (8, 6),
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Quick visualization of VTKData object.

    Args:
        data: VTKData object with u, v, and optionally p fields
        field: Field to plot - "velocity_magnitude", "vorticity", "u", "v", "p"
        ax: Matplotlib axes to plot on (created if None)
        figsize: Figure size if creating new figure
        **kwargs: Additional arguments passed to plot_contour_field

    Returns:
        Tuple of (figure, axes)

    Example:
        >>> from cfd_viz import read_vtk_file
        >>> data = read_vtk_file("simulation.vtk")
        >>> fig, ax = quick_plot_data(data, field="velocity_magnitude")
        >>> plt.show()
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if field == "velocity_magnitude":
        if data.u is None or data.v is None:
            raise ValueError("VTKData must have both u and v fields")
        field_data = magnitude(data.u, data.v)
        title = kwargs.pop("title", "Velocity Magnitude")
        colorbar_label = kwargs.pop("colorbar_label", "|V|")
    elif field == "vorticity":
        if data.u is None or data.v is None:
            raise ValueError("VTKData must have both u and v fields")
        field_data = vorticity(data.u, data.v, data.dx, data.dy)
        title = kwargs.pop("title", "Vorticity")
        colorbar_label = kwargs.pop("colorbar_label", "ω")
    elif field == "u":
        if data.u is None:
            raise ValueError("VTKData must have u field")
        field_data = data.u
        title = kwargs.pop("title", "U Velocity")
        colorbar_label = kwargs.pop("colorbar_label", "u")
    elif field == "v":
        if data.v is None:
            raise ValueError("VTKData must have v field")
        field_data = data.v
        title = kwargs.pop("title", "V Velocity")
        colorbar_label = kwargs.pop("colorbar_label", "v")
    elif field == "p":
        p_field = data.get("p")
        if p_field is None:
            p_field = data.get("pressure")
        if p_field is None:
            raise ValueError("Pressure field required but not available")
        field_data = p_field
        title = kwargs.pop("title", "Pressure")
        colorbar_label = kwargs.pop("colorbar_label", "p")
    else:
        raise ValueError(f"Unknown field: {field}")

    plot_contour_field(
        data.X,
        data.Y,
        field_data,
        ax=ax,
        title=title,
        colorbar_label=colorbar_label,
        **kwargs,
    )

    return fig, ax
