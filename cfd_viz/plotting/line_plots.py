"""Profile Plotting Functions.

Functions for plotting line profiles, boundary layer profiles,
and cross-sectional data from the analysis module.
"""

from collections.abc import Sequence
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray

from cfd_viz.analysis.boundary_layer import (
    BoundaryLayerDevelopment,
    BoundaryLayerProfile,
)
from cfd_viz.analysis.flow_features import CrossSectionalAverages
from cfd_viz.analysis.line_extraction import LineProfile, MultipleProfiles


def plot_line_profile(
    profile: LineProfile,
    ax: Optional[Axes] = None,
    plot_components: bool = True,
    plot_magnitude: bool = True,
    plot_pressure: bool = False,
    title: Optional[str] = None,
    **kwargs,
) -> Axes:
    """Plot a line profile extracted from flow field.

    Args:
        profile: LineProfile dataclass from extract_line_profile.
        ax: Matplotlib axes to plot on. If None, creates new axes.
        plot_components: Whether to plot u and v components.
        plot_magnitude: Whether to plot velocity magnitude.
        plot_pressure: Whether to plot pressure on secondary axis.
        title: Plot title. If None, uses profile description.
        **kwargs: Additional arguments for plot styling.

    Returns:
        The matplotlib axes object.
    """
    if ax is None:
        _, ax = plt.subplots()

    if plot_components:
        ax.plot(
            profile.distance,
            profile.u,
            "b-",
            label="u-velocity",
            linewidth=kwargs.get("linewidth", 1.5),
        )
        ax.plot(
            profile.distance,
            profile.v,
            "r-",
            label="v-velocity",
            linewidth=kwargs.get("linewidth", 1.5),
        )

    if plot_magnitude:
        ax.plot(
            profile.distance,
            profile.velocity_mag,
            "k-",
            label="|v|",
            linewidth=kwargs.get("linewidth", 2),
        )

    if plot_pressure and profile.pressure is not None:
        ax2 = ax.twinx()
        ax2.plot(
            profile.distance,
            profile.pressure,
            "g--",
            label="pressure",
            alpha=0.7,
        )
        ax2.set_ylabel("Pressure", color="g")
        ax2.tick_params(axis="y", labelcolor="g")

    ax.set_xlabel("Distance along line (m)")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title(title or f"Line Profile: {profile.start_point} to {profile.end_point}")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    return ax


def plot_multiple_profiles(
    profiles: MultipleProfiles,
    ax: Optional[Axes] = None,
    plot_type: str = "magnitude",
    colormap: str = "viridis",
    title: str = "Multiple Line Profiles",
    **kwargs,
) -> Axes:
    """Plot multiple line profiles on the same axes.

    Args:
        profiles: MultipleProfiles dataclass from extract_multiple_profiles.
        ax: Matplotlib axes to plot on. If None, creates new axes.
        plot_type: What to plot ('magnitude', 'u', 'v').
        colormap: Colormap for distinguishing profiles.
        title: Plot title.
        **kwargs: Additional arguments for plot styling.

    Returns:
        The matplotlib axes object.
    """
    if ax is None:
        _, ax = plt.subplots()

    colors = plt.cm.get_cmap(colormap)(np.linspace(0, 1, len(profiles.profiles)))

    for i, profile in enumerate(profiles.profiles):
        if plot_type == "magnitude":
            data = profile.velocity_mag
        elif plot_type == "u":
            data = profile.u
        elif plot_type == "v":
            data = profile.v
        else:
            raise ValueError(f"Unknown plot_type: {plot_type}")

        # CrossSection uses 'coordinate', not 'distance'
        x_data = profile.coordinate
        label = f"pos={profile.position:.2f}"
        ax.plot(
            x_data,
            data,
            color=colors[i],
            linewidth=kwargs.get("linewidth", 2),
            label=label,
        )

    ax.set_xlabel("Distance along line (m)")
    ax.set_ylabel(
        f"{'Velocity Magnitude' if plot_type == 'magnitude' else plot_type} (m/s)"
    )
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    return ax


def plot_velocity_profiles(
    x_stations: Sequence[float],
    y: NDArray,
    u_profiles: Sequence[NDArray],
    ax: Optional[Axes] = None,
    colormap: str = "viridis",
    title: str = "Velocity Profiles at Different Stations",
    **kwargs,
) -> Axes:
    """Plot vertical velocity profiles at different x-stations.

    Args:
        x_stations: List of x-locations for the profiles.
        y: 1D array of y-coordinates.
        u_profiles: List of 1D arrays of u-velocity at each station.
        ax: Matplotlib axes to plot on. If None, creates new axes.
        colormap: Colormap for distinguishing profiles.
        title: Plot title.
        **kwargs: Additional arguments for plot styling.

    Returns:
        The matplotlib axes object.
    """
    if ax is None:
        _, ax = plt.subplots()

    colors = plt.cm.get_cmap(colormap)(np.linspace(0, 1, len(x_stations)))

    for x_station, u_profile, color in zip(x_stations, u_profiles, colors):
        ax.plot(
            u_profile,
            y,
            color=color,
            linewidth=kwargs.get("linewidth", 2),
            label=f"x={x_station:.2f}",
        )

    ax.set_xlabel("u-velocity (m/s)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    return ax


def plot_boundary_layer_profile(
    bl_profile: BoundaryLayerProfile,
    ax: Optional[Axes] = None,
    normalized: bool = True,
    plot_blasius: bool = False,
    title: Optional[str] = None,
    **kwargs,
) -> Axes:
    """Plot a boundary layer velocity profile.

    Args:
        bl_profile: BoundaryLayerProfile dataclass from analyze_boundary_layer.
        ax: Matplotlib axes to plot on. If None, creates new axes.
        normalized: If True, normalize by delta_99 and freestream velocity.
        plot_blasius: If True, overlay Blasius solution.
        title: Plot title.
        **kwargs: Additional arguments for plot styling.

    Returns:
        The matplotlib axes object.
    """
    if ax is None:
        _, ax = plt.subplots()

    if normalized and bl_profile.delta_99 is not None and bl_profile.delta_99 > 0:
        y_plot = bl_profile.wall_distance / bl_profile.delta_99
        u_plot = bl_profile.u / bl_profile.u_edge
        ax.set_xlabel("u/u∞")
        ax.set_ylabel("y/δ")
        ax.set_ylim(0, 2)
    else:
        y_plot = bl_profile.wall_distance
        u_plot = bl_profile.u
        ax.set_xlabel("u (m/s)")
        ax.set_ylabel("y (m)")

    ax.plot(
        u_plot,
        y_plot,
        "b-",
        linewidth=kwargs.get("linewidth", 2),
        label=f"x={bl_profile.x_location:.3f}",
    )

    if plot_blasius and normalized:
        # Add simple Blasius reference curve
        eta = np.linspace(0, 8, 100)
        # Approximate Blasius solution: f' ≈ tanh(eta/2)
        u_blasius = np.tanh(eta / 2)
        y_blasius = eta / 5  # Scale to match δ ≈ 5η/Re^0.5
        ax.plot(
            u_blasius,
            y_blasius,
            "k--",
            linewidth=1.5,
            label="Blasius",
            alpha=0.7,
        )

    ax.set_title(title or f"Boundary Layer at x={bl_profile.x_location:.3f}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    return ax


def plot_boundary_layer_profiles(
    profiles: List[BoundaryLayerProfile],
    ax: Optional[Axes] = None,
    normalized: bool = True,
    colormap: str = "plasma",
    title: str = "Boundary Layer Profiles",
    **kwargs,
) -> Axes:
    """Plot multiple boundary layer profiles on the same axes.

    Args:
        profiles: List of BoundaryLayerProfile dataclasses.
        ax: Matplotlib axes to plot on. If None, creates new axes.
        normalized: If True, normalize by delta_99 and freestream velocity.
        colormap: Colormap for distinguishing profiles.
        title: Plot title.
        **kwargs: Additional arguments for plot styling.

    Returns:
        The matplotlib axes object.
    """
    if ax is None:
        _, ax = plt.subplots()

    colors = plt.cm.get_cmap(colormap)(np.linspace(0, 1, len(profiles)))

    for profile, color in zip(profiles, colors):
        if profile.delta_99 is None or profile.delta_99 <= 0:
            continue

        if normalized:
            y_plot = profile.wall_distance / profile.delta_99
            u_plot = profile.u / profile.u_edge
        else:
            y_plot = profile.wall_distance
            u_plot = profile.u

        ax.plot(
            u_plot,
            y_plot,
            color=color,
            linewidth=kwargs.get("linewidth", 2),
            label=f"x={profile.x_location:.2f}",
        )

    if normalized:
        ax.set_xlabel("u/u∞")
        ax.set_ylabel("y/δ")
        ax.set_ylim(0, 2)
    else:
        ax.set_xlabel("u (m/s)")
        ax.set_ylabel("y (m)")

    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    return ax


def plot_boundary_layer_development(
    development: BoundaryLayerDevelopment,
    ax: Optional[Axes] = None,
    what: str = "all",
    title: str = "Boundary Layer Development",
    **kwargs,
) -> Axes:
    """Plot boundary layer thickness development along x.

    Args:
        development: BoundaryLayerDevelopment dataclass.
        ax: Matplotlib axes to plot on. If None, creates new axes.
        what: What to plot ('delta', 'delta_star', 'theta', 'H', 'cf', 'all').
        title: Plot title.
        **kwargs: Additional arguments for plot styling.

    Returns:
        The matplotlib axes object.
    """
    if ax is None:
        _, ax = plt.subplots()

    x = development.x_locations

    if what in ("delta", "all"):
        ax.plot(x, development.delta_99, "b-o", label="δ₉₉", linewidth=2, markersize=4)
    if what in ("delta_star", "all"):
        ax.plot(x, development.delta_star, "r-s", label="δ*", linewidth=2, markersize=4)
    if what in ("theta", "all"):
        ax.plot(x, development.theta, "g-^", label="θ", linewidth=2, markersize=4)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("Thickness (m)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Add shape factor on secondary axis if plotting all
    if what == "all" and development.shape_factor is not None:
        ax2 = ax.twinx()
        ax2.plot(x, development.shape_factor, "m--", label="H", linewidth=2, alpha=0.7)
        ax2.set_ylabel("Shape Factor H", color="m")
        ax2.tick_params(axis="y", labelcolor="m")

    return ax


def plot_cross_sectional_averages(
    averages: CrossSectionalAverages,
    ax: Optional[Axes] = None,
    plot_components: bool = True,
    plot_magnitude: bool = True,
    plot_pressure: bool = False,
    title: Optional[str] = None,
    **kwargs,
) -> Axes:
    """Plot cross-sectional averaged profiles.

    Args:
        averages: CrossSectionalAverages dataclass from compute_cross_sectional_averages.
        ax: Matplotlib axes to plot on. If None, creates new axes.
        plot_components: Whether to plot u_avg and v_avg.
        plot_magnitude: Whether to plot velocity magnitude average.
        plot_pressure: Whether to plot pressure average.
        title: Plot title.
        **kwargs: Additional arguments for plot styling.

    Returns:
        The matplotlib axes object.
    """
    if ax is None:
        _, ax = plt.subplots()

    coord = averages.coordinate

    if plot_components:
        ax.plot(
            averages.u_avg,
            coord,
            "b-",
            linewidth=kwargs.get("linewidth", 2),
            label="<u>",
        )
        ax.plot(
            averages.v_avg,
            coord,
            "r-",
            linewidth=kwargs.get("linewidth", 2),
            label="<v>",
        )

    if plot_magnitude:
        ax.plot(
            averages.velocity_mag_avg,
            coord,
            "k-",
            linewidth=kwargs.get("linewidth", 2),
            label="<|v|>",
        )

    if plot_pressure and averages.p_avg is not None:
        ax2 = ax.twinx()
        ax2.plot(
            averages.p_avg,
            coord,
            "g--",
            linewidth=kwargs.get("linewidth", 1.5),
            label="<p>",
            alpha=0.7,
        )
        ax2.set_xlabel("Pressure", color="g")
        ax2.tick_params(axis="x", labelcolor="g")

    axis_label = "y (m)" if averages.averaging_axis == "x" else "x (m)"
    ax.set_xlabel("Average Velocity (m/s)")
    ax.set_ylabel(axis_label)
    ax.set_title(
        title or f"Cross-sectional Averages (bulk={averages.bulk_velocity:.3f})"
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    return ax
