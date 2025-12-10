"""Field Plotting Functions.

Functions for plotting scalar and vector fields from CFD data.
All functions accept an optional axes parameter and return the axes
for further customization.
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray


def plot_contour_field(
    X: NDArray,
    Y: NDArray,
    field: NDArray,
    ax: Optional[Axes] = None,
    levels: int = 20,
    cmap: str = "viridis",
    colorbar: bool = True,
    colorbar_label: str = "",
    title: str = "",
    xlabel: str = "x",
    ylabel: str = "y",
    aspect: str = "equal",
    alpha: float = 1.0,
) -> Axes:
    """Plot a scalar field as filled contours.

    Args:
        X: 2D array of x-coordinates (meshgrid).
        Y: 2D array of y-coordinates (meshgrid).
        field: 2D array of field values.
        ax: Matplotlib axes to plot on. If None, creates new axes.
        levels: Number of contour levels.
        cmap: Colormap name.
        colorbar: Whether to add a colorbar.
        colorbar_label: Label for the colorbar.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        aspect: Aspect ratio ('equal', 'auto', or numeric).
        alpha: Transparency (0-1).

    Returns:
        The matplotlib axes object.
    """
    if ax is None:
        _, ax = plt.subplots()

    cs = ax.contourf(X, Y, field, levels=levels, cmap=cmap, alpha=alpha)

    if colorbar:
        plt.colorbar(cs, ax=ax, label=colorbar_label)

    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect(aspect)

    return ax


def plot_velocity_field(
    X: NDArray,
    Y: NDArray,
    u: NDArray,
    v: NDArray,
    ax: Optional[Axes] = None,
    levels: int = 20,
    cmap: str = "viridis",
    colorbar: bool = True,
    title: str = "Velocity Magnitude",
    **kwargs,
) -> Axes:
    """Plot velocity magnitude field as filled contours.

    Args:
        X: 2D array of x-coordinates (meshgrid).
        Y: 2D array of y-coordinates (meshgrid).
        u: 2D array of x-velocity component.
        v: 2D array of y-velocity component.
        ax: Matplotlib axes to plot on. If None, creates new axes.
        levels: Number of contour levels.
        cmap: Colormap name.
        colorbar: Whether to add a colorbar.
        title: Plot title.
        **kwargs: Additional arguments passed to plot_contour_field.

    Returns:
        The matplotlib axes object.
    """
    velocity_mag = np.sqrt(u**2 + v**2)
    return plot_contour_field(
        X,
        Y,
        velocity_mag,
        ax=ax,
        levels=levels,
        cmap=cmap,
        colorbar=colorbar,
        colorbar_label="Velocity Magnitude (m/s)",
        title=title,
        **kwargs,
    )


def plot_pressure_field(
    X: NDArray,
    Y: NDArray,
    pressure: NDArray,
    ax: Optional[Axes] = None,
    levels: int = 20,
    cmap: str = "plasma",
    colorbar: bool = True,
    title: str = "Pressure Field",
    **kwargs,
) -> Axes:
    """Plot pressure field as filled contours.

    Args:
        X: 2D array of x-coordinates (meshgrid).
        Y: 2D array of y-coordinates (meshgrid).
        pressure: 2D array of pressure values.
        ax: Matplotlib axes to plot on. If None, creates new axes.
        levels: Number of contour levels.
        cmap: Colormap name.
        colorbar: Whether to add a colorbar.
        title: Plot title.
        **kwargs: Additional arguments passed to plot_contour_field.

    Returns:
        The matplotlib axes object.
    """
    return plot_contour_field(
        X,
        Y,
        pressure,
        ax=ax,
        levels=levels,
        cmap=cmap,
        colorbar=colorbar,
        colorbar_label="Pressure",
        title=title,
        **kwargs,
    )


def plot_vorticity_field(
    X: NDArray,
    Y: NDArray,
    omega: NDArray,
    ax: Optional[Axes] = None,
    levels: int = 20,
    cmap: str = "RdBu_r",
    colorbar: bool = True,
    title: str = "Vorticity Field",
    symmetric: bool = True,
    **kwargs,
) -> Axes:
    """Plot vorticity field as filled contours.

    Args:
        X: 2D array of x-coordinates (meshgrid).
        Y: 2D array of y-coordinates (meshgrid).
        omega: 2D array of vorticity values.
        ax: Matplotlib axes to plot on. If None, creates new axes.
        levels: Number of contour levels.
        cmap: Colormap name (RdBu_r is good for diverging data).
        colorbar: Whether to add a colorbar.
        title: Plot title.
        symmetric: Whether to make levels symmetric around zero.
        **kwargs: Additional arguments passed to plot_contour_field.

    Returns:
        The matplotlib axes object.
    """
    if ax is None:
        _, ax = plt.subplots()

    if symmetric:
        max_abs = np.max(np.abs(omega))
        vort_levels = np.linspace(-max_abs, max_abs, levels)
        cs = ax.contourf(X, Y, omega, levels=vort_levels, cmap=cmap, extend="both")
    else:
        cs = ax.contourf(X, Y, omega, levels=levels, cmap=cmap)

    if colorbar:
        plt.colorbar(cs, ax=ax, label="Vorticity (1/s)")

    if title:
        ax.set_title(title)
    ax.set_xlabel(kwargs.get("xlabel", "x"))
    ax.set_ylabel(kwargs.get("ylabel", "y"))
    ax.set_aspect(kwargs.get("aspect", "equal"))

    return ax


def plot_vector_field(
    X: NDArray,
    Y: NDArray,
    u: NDArray,
    v: NDArray,
    ax: Optional[Axes] = None,
    density: int = 20,
    scale: Optional[float] = None,
    color: str = "black",
    alpha: float = 0.7,
    title: str = "Velocity Vectors",
    **kwargs,
) -> Axes:
    """Plot velocity vectors using quiver plot.

    Args:
        X: 2D array of x-coordinates (meshgrid).
        Y: 2D array of y-coordinates (meshgrid).
        u: 2D array of x-velocity component.
        v: 2D array of y-velocity component.
        ax: Matplotlib axes to plot on. If None, creates new axes.
        density: Vector density (skip every nth point).
        scale: Quiver scale parameter (None for auto).
        color: Vector color.
        alpha: Transparency (0-1).
        title: Plot title.
        **kwargs: Additional arguments passed to quiver.

    Returns:
        The matplotlib axes object.
    """
    if ax is None:
        _, ax = plt.subplots()

    # Subsample for cleaner visualization
    skip = max(1, len(X) // density)
    X_sub = X[::skip, ::skip]
    Y_sub = Y[::skip, ::skip]
    u_sub = u[::skip, ::skip]
    v_sub = v[::skip, ::skip]

    ax.quiver(X_sub, Y_sub, u_sub, v_sub, scale=scale, color=color, alpha=alpha)

    if title:
        ax.set_title(title)
    ax.set_xlabel(kwargs.get("xlabel", "x"))
    ax.set_ylabel(kwargs.get("ylabel", "y"))
    ax.set_aspect(kwargs.get("aspect", "equal"))

    return ax


def plot_streamlines(
    X: NDArray,
    Y: NDArray,
    u: NDArray,
    v: NDArray,
    ax: Optional[Axes] = None,
    density: float = 1.5,
    color: str = "black",
    linewidth: float = 0.8,
    arrowsize: float = 1.2,
    title: str = "Streamlines",
    **kwargs,
) -> Axes:
    """Plot streamlines of the velocity field.

    Args:
        X: 2D array of x-coordinates (meshgrid).
        Y: 2D array of y-coordinates (meshgrid).
        u: 2D array of x-velocity component.
        v: 2D array of y-velocity component.
        ax: Matplotlib axes to plot on. If None, creates new axes.
        density: Streamline density.
        color: Streamline color.
        linewidth: Line width.
        arrowsize: Arrow size.
        title: Plot title.
        **kwargs: Additional arguments passed to streamplot.

    Returns:
        The matplotlib axes object.
    """
    if ax is None:
        _, ax = plt.subplots()

    ax.streamplot(
        X,
        Y,
        u,
        v,
        density=density,
        color=color,
        linewidth=linewidth,
        arrowsize=arrowsize,
    )

    if title:
        ax.set_title(title)
    ax.set_xlabel(kwargs.get("xlabel", "x"))
    ax.set_ylabel(kwargs.get("ylabel", "y"))
    ax.set_aspect(kwargs.get("aspect", "equal"))

    return ax


def plot_vorticity_with_streamlines(
    X: NDArray,
    Y: NDArray,
    omega: NDArray,
    u: NDArray,
    v: NDArray,
    ax: Optional[Axes] = None,
    vort_levels: int = 20,
    vort_cmap: str = "RdBu_r",
    vort_alpha: float = 0.8,
    stream_density: float = 1.5,
    stream_color: str = "black",
    colorbar: bool = True,
    title: str = "Vorticity with Streamlines",
    **kwargs,
) -> Axes:
    """Plot vorticity field with overlaid streamlines.

    Args:
        X: 2D array of x-coordinates (meshgrid).
        Y: 2D array of y-coordinates (meshgrid).
        omega: 2D array of vorticity values.
        u: 2D array of x-velocity component.
        v: 2D array of y-velocity component.
        ax: Matplotlib axes to plot on. If None, creates new axes.
        vort_levels: Number of vorticity contour levels.
        vort_cmap: Colormap for vorticity.
        vort_alpha: Transparency for vorticity contours.
        stream_density: Streamline density.
        stream_color: Streamline color.
        colorbar: Whether to add a colorbar.
        title: Plot title.
        **kwargs: Additional arguments.

    Returns:
        The matplotlib axes object.
    """
    if ax is None:
        _, ax = plt.subplots()

    # Plot vorticity
    max_abs = np.max(np.abs(omega))
    levels = np.linspace(-max_abs, max_abs, vort_levels)
    cs = ax.contourf(X, Y, omega, levels=levels, cmap=vort_cmap, alpha=vort_alpha)

    if colorbar:
        plt.colorbar(cs, ax=ax, label="Vorticity (1/s)")

    # Add streamlines
    ax.streamplot(X, Y, u, v, density=stream_density, color=stream_color, linewidth=0.8)

    if title:
        ax.set_title(title)
    ax.set_xlabel(kwargs.get("xlabel", "x"))
    ax.set_ylabel(kwargs.get("ylabel", "y"))
    ax.set_aspect(kwargs.get("aspect", "equal"))

    return ax
