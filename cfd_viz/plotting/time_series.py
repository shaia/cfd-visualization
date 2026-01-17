"""Monitoring and Time Series Plotting Functions.

Functions for plotting time series data, convergence history,
and monitoring dashboards from the analysis module.
"""

from collections.abc import Sequence
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from cfd_viz.analysis.time_series import (
    ConvergenceMetrics,
    FlowMetrics,
    FlowMetricsTimeSeries,
    TemporalStatistics,
)


def plot_metric_time_series(
    time_values: NDArray,
    metric_values: NDArray,
    ax: Optional[Axes] = None,
    label: str = "",
    color: str = "blue",
    marker: str = "o",
    markersize: int = 4,
    linewidth: float = 1.5,
    title: str = "",
    xlabel: str = "Time Step",
    ylabel: str = "Value",
    grid: bool = True,
    **kwargs,
) -> Axes:
    """Plot a single metric as a time series.

    Args:
        time_values: Array of time/step values (x-axis).
        metric_values: Array of metric values (y-axis).
        ax: Matplotlib axes to plot on. If None, creates new axes.
        label: Legend label.
        color: Line/marker color.
        marker: Marker style.
        markersize: Marker size.
        linewidth: Line width.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        grid: Whether to show grid.
        **kwargs: Additional arguments passed to plot.

    Returns:
        The matplotlib axes object.
    """
    if ax is None:
        _, ax = plt.subplots()

    ax.plot(
        time_values,
        metric_values,
        f"{color[0]}-{marker}",
        markersize=markersize,
        linewidth=linewidth,
        label=label,
        **kwargs,
    )

    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if grid:
        ax.grid(True, alpha=0.3)
    if label:
        ax.legend(fontsize=8)

    return ax


def plot_convergence_history(
    history: FlowMetricsTimeSeries,
    metrics: Sequence[str] = ("max_velocity", "mean_velocity", "total_kinetic_energy"),
    figsize: tuple = (15, 4),
    **kwargs,
) -> Figure:
    """Plot convergence history for multiple metrics.

    Args:
        history: FlowMetricsTimeSeries with flow metrics over time.
        metrics: Tuple of metric names to plot.
        figsize: Figure size.
        **kwargs: Additional arguments passed to plotting functions.

    Returns:
        The matplotlib figure object.
    """
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
    if len(metrics) == 1:
        axes = [axes]

    time_steps = np.arange(len(history.snapshots))
    colors = ["blue", "red", "green", "orange", "purple"]
    labels = {
        "max_velocity": "Max Velocity (m/s)",
        "mean_velocity": "Mean Velocity (m/s)",
        "total_kinetic_energy": "Kinetic Energy",
        "max_pressure": "Max Pressure",
        "mean_pressure": "Mean Pressure",
        "max_vorticity": "Max |ω| (1/s)",
    }

    for i, (metric, ax) in enumerate(zip(metrics, axes)):
        values = history.get_metric_array(metric)
        plot_metric_time_series(
            time_steps,
            values,
            ax=ax,
            color=colors[i % len(colors)],
            title=f"{labels.get(metric, metric)} vs Time",
            ylabel=labels.get(metric, metric),
            **kwargs,
        )

    fig.suptitle("Convergence History", fontsize=14)
    plt.tight_layout()

    return fig


def plot_monitoring_dashboard(
    snapshot: FlowMetrics,
    history: FlowMetricsTimeSeries,
    X: NDArray,
    Y: NDArray,
    velocity_mag: NDArray,
    pressure: NDArray,
    figsize: tuple = (18, 10),
    **kwargs,
) -> Figure:
    """Create a real-time monitoring dashboard.

    Args:
        snapshot: Current FlowMetrics.
        history: FlowMetricsTimeSeries with past metrics.
        X: 2D array of x-coordinates (meshgrid).
        Y: 2D array of y-coordinates (meshgrid).
        velocity_mag: 2D array of current velocity magnitude.
        pressure: 2D array of current pressure field.
        figsize: Figure size.
        **kwargs: Additional arguments.

    Returns:
        The matplotlib figure object.
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle("CFD Real-time Monitoring Dashboard", fontsize=16)
    axes = axes.flatten()

    # 1. Velocity magnitude field
    cs1 = axes[0].contourf(X, Y, velocity_mag, levels=20, cmap="viridis")
    plt.colorbar(cs1, ax=axes[0], label="Velocity (m/s)")
    axes[0].set_title("Velocity Field")
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    axes[0].set_aspect("equal")

    # 2. Pressure field
    cs2 = axes[1].contourf(X, Y, pressure, levels=20, cmap="plasma")
    plt.colorbar(cs2, ax=axes[1], label="Pressure")
    axes[1].set_title("Pressure Field")
    axes[1].set_xlabel("x (m)")
    axes[1].set_ylabel("y (m)")
    axes[1].set_aspect("equal")

    # 3-5. Time series plots
    if len(history.snapshots) > 1:
        time_steps = np.arange(len(history.snapshots))

        # Max velocity
        max_vel = history.get_metric_array("max_velocity")
        axes[2].plot(time_steps, max_vel, "b-o", markersize=4)
        axes[2].set_title("Maximum Velocity vs Time")
        axes[2].set_xlabel("Time Step")
        axes[2].set_ylabel("Max Velocity (m/s)")
        axes[2].grid(True, alpha=0.3)

        # Mean velocity
        mean_vel = history.get_metric_array("mean_velocity")
        axes[3].plot(time_steps, mean_vel, "r-o", markersize=4)
        axes[3].set_title("Mean Velocity vs Time")
        axes[3].set_xlabel("Time Step")
        axes[3].set_ylabel("Mean Velocity (m/s)")
        axes[3].grid(True, alpha=0.3)

        # Kinetic energy
        ke = history.get_metric_array("total_kinetic_energy")
        axes[4].plot(time_steps, ke, "g-o", markersize=4)
        axes[4].set_title("Total Kinetic Energy vs Time")
        axes[4].set_xlabel("Time Step")
        axes[4].set_ylabel("Kinetic Energy")
        axes[4].grid(True, alpha=0.3)

    # 6. Statistics panel
    plot_statistics_panel(snapshot, history, ax=axes[5])

    plt.tight_layout()

    return fig


def plot_statistics_panel(
    snapshot: FlowMetrics,
    history: Optional[FlowMetricsTimeSeries] = None,
    ax: Optional[Axes] = None,
    title: str = "Current Statistics",
    **kwargs,
) -> Axes:
    """Plot a statistics summary panel.

    Args:
        snapshot: Current FlowMetrics.
        history: Optional FlowMetricsTimeSeries for convergence info.
        ax: Matplotlib axes to plot on. If None, creates new axes.
        title: Plot title.
        **kwargs: Additional arguments.

    Returns:
        The matplotlib axes object.
    """
    if ax is None:
        _, ax = plt.subplots()

    ax.axis("off")
    ax.set_title(title)

    stats_text = f"""
    Current Flow Statistics:

    Velocity:
      Max: {snapshot.max_velocity:.4f} m/s
      Mean: {snapshot.mean_velocity:.4f} m/s

    Pressure:
      Max: {snapshot.max_pressure:.4f}
      Mean: {snapshot.mean_pressure:.4f}

    Energy:
      Total KE: {snapshot.total_kinetic_energy:.4f}

    Vorticity:
      Max |ω|: {snapshot.max_vorticity:.4f} 1/s
    """

    ax.text(
        0.05,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
    )

    # Add convergence info if history available
    if history is not None and len(history.snapshots) > 5:
        max_vel_trend = history.estimate_convergence_trend("max_velocity", window=5)
        mean_vel_trend = history.estimate_convergence_trend("mean_velocity", window=5)
        is_converged = history.is_converged(threshold=1e-5)

        if max_vel_trend is not None and mean_vel_trend is not None:
            convergence_text = f"""
    Convergence Analysis:
      Max Velocity Trend: {max_vel_trend:+.6f}/step
      Mean Velocity Trend: {mean_vel_trend:+.6f}/step

      Status: {"Converged" if is_converged else "Still Changing"}
            """

            ax.text(
                0.05,
                0.40,
                convergence_text,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8),
            )

    return ax


def plot_temporal_statistics(
    stats: TemporalStatistics,
    X: NDArray,
    Y: NDArray,
    figsize: tuple = (15, 5),
    **kwargs,
) -> Figure:
    """Plot temporal statistics (mean, std, min, max fields).

    Args:
        stats: TemporalStatistics dataclass from compute_temporal_statistics.
        X: 2D array of x-coordinates (meshgrid).
        Y: 2D array of y-coordinates (meshgrid).
        figsize: Figure size.
        **kwargs: Additional arguments.

    Returns:
        The matplotlib figure object.
    """
    fig, axes = plt.subplots(1, 4, figsize=figsize)

    # Mean field
    cs1 = axes[0].contourf(X, Y, stats.mean, levels=20, cmap="viridis")
    plt.colorbar(cs1, ax=axes[0], label="Mean")
    axes[0].set_title("Time-Averaged Field")
    axes[0].set_aspect("equal")

    # Standard deviation
    cs2 = axes[1].contourf(X, Y, stats.std, levels=20, cmap="hot")
    plt.colorbar(cs2, ax=axes[1], label="Std Dev")
    axes[1].set_title("Standard Deviation")
    axes[1].set_aspect("equal")

    # Minimum
    cs3 = axes[2].contourf(X, Y, stats.min, levels=20, cmap="viridis")
    plt.colorbar(cs3, ax=axes[2], label="Min")
    axes[2].set_title("Minimum Values")
    axes[2].set_aspect("equal")

    # Maximum
    cs4 = axes[3].contourf(X, Y, stats.max, levels=20, cmap="viridis")
    plt.colorbar(cs4, ax=axes[3], label="Max")
    axes[3].set_title("Maximum Values")
    axes[3].set_aspect("equal")

    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.suptitle("Temporal Statistics", fontsize=14)
    plt.tight_layout()

    return fig


def plot_convergence_metrics(
    metrics: ConvergenceMetrics,
    ax: Optional[Axes] = None,
    title: str = "Convergence Metrics",
    **kwargs,
) -> Axes:
    """Plot convergence metrics summary.

    Args:
        metrics: ConvergenceMetrics dataclass from analyze_convergence.
        ax: Matplotlib axes to plot on. If None, creates new axes.
        title: Plot title.
        **kwargs: Additional arguments.

    Returns:
        The matplotlib axes object.
    """
    if ax is None:
        _, ax = plt.subplots()

    ax.axis("off")
    ax.set_title(title)

    status = "✓ Converged" if metrics.is_converged else "✗ Not Converged"
    color = "green" if metrics.is_converged else "red"

    stats_text = f"""
    Convergence Analysis:

    Status: {status}

    Final Value: {metrics.final_value:.6f}
    Final Rate: {metrics.final_rate:.2e}/step

    Statistics:
      Mean Change: {metrics.mean_change:.2e}
      Std Change: {metrics.std_change:.2e}
      Max Change: {metrics.max_change:.2e}

    Steps to Convergence: {metrics.steps_to_convergence or "N/A"}
    """

    ax.text(
        0.05,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
    )

    # Add colored indicator
    ax.text(
        0.05,
        0.15,
        f"● {status}",
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        color=color,
    )

    return ax
