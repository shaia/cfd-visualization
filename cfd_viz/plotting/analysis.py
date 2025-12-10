"""Analysis Result Plotting Functions.

Functions for plotting results from the analysis module including
case comparisons, field differences, wake regions, and flow statistics.
"""

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray

from cfd_viz.analysis.case_comparison import CaseComparison, FieldDifference
from cfd_viz.analysis.flow_features import SpatialFluctuations, WakeRegion


def plot_field_difference(
    diff: FieldDifference,
    X: NDArray,
    Y: NDArray,
    ax: Optional[Axes] = None,
    cmap: str = "RdBu_r",
    levels: int = 20,
    colorbar: bool = True,
    title: Optional[str] = None,
    **kwargs,
) -> Axes:
    """Plot a field difference from case comparison.

    Args:
        diff: FieldDifference dataclass from compute_field_difference.
        X: 2D array of x-coordinates (meshgrid).
        Y: 2D array of y-coordinates (meshgrid).
        ax: Matplotlib axes to plot on. If None, creates new axes.
        cmap: Colormap name (diverging colormap recommended).
        levels: Number of contour levels.
        colorbar: Whether to add a colorbar.
        title: Plot title.
        **kwargs: Additional arguments.

    Returns:
        The matplotlib axes object.
    """
    if ax is None:
        _, ax = plt.subplots()

    # Make levels symmetric around zero
    max_abs = np.max(np.abs(diff.diff))
    level_values = np.linspace(-max_abs, max_abs, levels)

    cs = ax.contourf(X, Y, diff.diff, levels=level_values, cmap=cmap, extend="both")

    if colorbar:
        cbar = plt.colorbar(cs, ax=ax)
        cbar.set_label(f"Difference (RMS={diff.rms_diff:.4f})")

    ax.set_title(title or f"Field Difference (Max={diff.max_diff:.4f})")
    ax.set_xlabel(kwargs.get("xlabel", "x"))
    ax.set_ylabel(kwargs.get("ylabel", "y"))
    ax.set_aspect(kwargs.get("aspect", "equal"))

    return ax


def plot_case_comparison(
    comparison: CaseComparison,
    X: NDArray,
    Y: NDArray,
    figsize: tuple = (15, 10),
    **kwargs,
) -> plt.Figure:
    """Create a multi-panel comparison plot for two CFD cases.

    Args:
        comparison: CaseComparison dataclass from compare_fields.
        X: 2D array of x-coordinates (meshgrid).
        Y: 2D array of y-coordinates (meshgrid).
        figsize: Figure size.
        **kwargs: Additional arguments.

    Returns:
        The matplotlib figure object.
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    # Velocity difference
    plot_field_difference(
        comparison.velocity_diff,
        X,
        Y,
        ax=axes[0],
        title="Velocity Magnitude Difference",
    )

    # Pressure difference
    plot_field_difference(
        comparison.pressure_diff,
        X,
        Y,
        ax=axes[1],
        cmap="PuOr",
        title="Pressure Difference",
    )

    # Statistics panel
    axes[2].axis("off")
    stats_text = f"""
    Comparison Statistics:

    Velocity Difference:
      Max: {comparison.velocity_diff.max_diff:.4f}
      Mean: {comparison.velocity_diff.mean_diff:.4f}
      RMS: {comparison.velocity_diff.rms_diff:.4f}

    Pressure Difference:
      Max: {comparison.pressure_diff.max_diff:.4f}
      Mean: {comparison.pressure_diff.mean_diff:.4f}
      RMS: {comparison.pressure_diff.rms_diff:.4f}

    Vorticity Difference:
      Max: {comparison.vorticity_diff.max_diff:.4f}
      RMS: {comparison.vorticity_diff.rms_diff:.4f}
    """
    axes[2].text(
        0.1,
        0.9,
        stats_text,
        transform=axes[2].transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
    )

    # Vorticity difference
    plot_field_difference(
        comparison.vorticity_diff,
        X,
        Y,
        ax=axes[3],
        title="Vorticity Difference",
    )

    # Relative velocity change (%)
    if comparison.velocity_diff.relative_diff is not None:
        rel_diff = comparison.velocity_diff.relative_diff * 100  # Convert to percentage
        max_abs = min(np.max(np.abs(rel_diff)), 100)  # Cap at 100%
        level_values = np.linspace(-max_abs, max_abs, 20)
        cs = axes[4].contourf(
            X, Y, rel_diff, levels=level_values, cmap="RdBu_r", extend="both"
        )
        plt.colorbar(cs, ax=axes[4], label="Relative Difference (%)")
        axes[4].set_title("Relative Velocity Change (%)")
        axes[4].set_xlabel("x")
        axes[4].set_ylabel("y")
        axes[4].set_aspect("equal")
    else:
        axes[4].axis("off")

    # Summary metrics bar chart
    metrics = {
        "Vel RMS": comparison.velocity_diff.rms_diff,
        "Press RMS": comparison.pressure_diff.rms_diff,
        "Vort RMS": comparison.vorticity_diff.rms_diff,
    }
    axes[5].bar(metrics.keys(), metrics.values(), color=["blue", "orange", "green"])
    axes[5].set_ylabel("RMS Difference")
    axes[5].set_title("RMS Differences by Field")
    axes[5].tick_params(axis="x", rotation=45)

    fig.suptitle("CFD Case Comparison", fontsize=14)
    plt.tight_layout()

    return fig


def plot_wake_region(
    wake: WakeRegion,
    X: NDArray,
    Y: NDArray,
    velocity_mag: NDArray,
    ax: Optional[Axes] = None,
    field_cmap: str = "viridis",
    field_alpha: float = 0.7,
    wake_color: str = "red",
    wake_linewidth: float = 2,
    title: Optional[str] = None,
    **kwargs,
) -> Axes:
    """Plot velocity field with wake region highlighted.

    Args:
        wake: WakeRegion dataclass from detect_wake_regions.
        X: 2D array of x-coordinates (meshgrid).
        Y: 2D array of y-coordinates (meshgrid).
        velocity_mag: 2D array of velocity magnitude.
        ax: Matplotlib axes to plot on. If None, creates new axes.
        field_cmap: Colormap for velocity field.
        field_alpha: Transparency for velocity field.
        wake_color: Color for wake region outline.
        wake_linewidth: Line width for wake region outline.
        title: Plot title.
        **kwargs: Additional arguments.

    Returns:
        The matplotlib axes object.
    """
    if ax is None:
        _, ax = plt.subplots()

    # Plot velocity magnitude
    ax.contourf(X, Y, velocity_mag, levels=20, cmap=field_cmap, alpha=field_alpha)

    # Overlay wake region
    ax.contour(
        X,
        Y,
        wake.mask.astype(int),
        levels=[0.5],
        colors=wake_color,
        linewidths=wake_linewidth,
    )

    # Mark centroid if available
    if wake.centroid is not None:
        ax.plot(
            wake.centroid[0],
            wake.centroid[1],
            "w*",
            markersize=15,
            markeredgecolor="black",
        )

    ax.set_title(title or f"Wake Region ({wake.area_fraction:.1%} of domain)")
    ax.set_xlabel(kwargs.get("xlabel", "x (m)"))
    ax.set_ylabel(kwargs.get("ylabel", "y (m)"))
    ax.set_aspect(kwargs.get("aspect", "equal"))

    return ax


def plot_spatial_fluctuations(
    fluct: SpatialFluctuations,
    X: NDArray,
    Y: NDArray,
    ax: Optional[Axes] = None,
    cmap: str = "hot",
    levels: int = 15,
    colorbar: bool = True,
    title: Optional[str] = None,
    **kwargs,
) -> Axes:
    """Plot spatial velocity fluctuations.

    Args:
        fluct: SpatialFluctuations dataclass from compute_spatial_fluctuations.
        X: 2D array of x-coordinates (meshgrid).
        Y: 2D array of y-coordinates (meshgrid).
        ax: Matplotlib axes to plot on. If None, creates new axes.
        cmap: Colormap name.
        levels: Number of contour levels.
        colorbar: Whether to add a colorbar.
        title: Plot title.
        **kwargs: Additional arguments.

    Returns:
        The matplotlib axes object.
    """
    if ax is None:
        _, ax = plt.subplots()

    cs = ax.contourf(X, Y, fluct.fluct_magnitude, levels=levels, cmap=cmap)

    if colorbar:
        plt.colorbar(cs, ax=ax, label="Velocity Fluctuation Magnitude")

    ax.set_title(
        title or f"Velocity Fluctuations (TI={fluct.turbulence_intensity:.1%})"
    )
    ax.set_xlabel(kwargs.get("xlabel", "x (m)"))
    ax.set_ylabel(kwargs.get("ylabel", "y (m)"))
    ax.set_aspect(kwargs.get("aspect", "equal"))

    return ax


def plot_flow_statistics(
    stats: Dict[str, float],
    ax: Optional[Axes] = None,
    title: str = "Flow Statistics",
    fontsize: int = 10,
    **kwargs,
) -> Axes:
    """Plot a text panel with flow statistics.

    Args:
        stats: Dictionary of statistic names and values.
        ax: Matplotlib axes to plot on. If None, creates new axes.
        title: Plot title.
        fontsize: Font size for text.
        **kwargs: Additional arguments.

    Returns:
        The matplotlib axes object.
    """
    if ax is None:
        _, ax = plt.subplots()

    ax.axis("off")

    # Format statistics text
    lines = [f"  {name}: {value:.4f}" for name, value in stats.items()]
    stats_text = "\n".join(lines)

    ax.text(
        0.05,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=fontsize,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor=kwargs.get("facecolor", "lightblue"),
            alpha=kwargs.get("alpha", 0.8),
        ),
    )

    ax.set_title(title)

    return ax
