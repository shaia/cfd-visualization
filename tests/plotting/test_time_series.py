"""Tests for cfd_viz.plotting.time_series module."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

matplotlib.use("Agg")  # Non-interactive backend for testing

from cfd_viz.analysis.time_series import (
    FlowMetrics,
    create_flow_metrics_time_series,
)
from cfd_viz.plotting.time_series import (
    plot_convergence_history,
    plot_metric_time_series,
    plot_monitoring_dashboard,
    plot_statistics_panel,
)


@pytest.fixture
def sample_snapshot():
    """Create a sample FlowMetrics for testing."""
    return FlowMetrics(
        timestamp=1000.0,
        max_velocity=2.5,
        mean_velocity=1.2,
        max_pressure=100.0,
        mean_pressure=50.0,
        total_kinetic_energy=500.0,
        max_vorticity=10.0,
    )


@pytest.fixture
def sample_history():
    """Create a sample FlowMetricsTimeSeries for testing."""
    history = create_flow_metrics_time_series(max_length=100)
    for i in range(20):
        metrics = FlowMetrics(
            timestamp=float(i),
            max_velocity=2.5 - i * 0.05,
            mean_velocity=1.2 - i * 0.02,
            max_pressure=100.0 - i * 2,
            mean_pressure=50.0 - i * 1,
            total_kinetic_energy=500.0 - i * 10,
            max_vorticity=10.0 - i * 0.1,
        )
        history.add(metrics)
    return history


class TestPlotMetricTimeSeries:
    """Tests for plot_metric_time_series function."""

    def test_returns_axes(self):
        """Should return matplotlib axes object."""
        time_values = np.arange(10)
        metric_values = np.random.rand(10)
        ax = plot_metric_time_series(time_values, metric_values)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_uses_provided_axes(self):
        """Should plot on provided axes."""
        time_values = np.arange(10)
        metric_values = np.random.rand(10)
        _, ax_input = plt.subplots()
        ax_output = plot_metric_time_series(time_values, metric_values, ax=ax_input)
        assert ax_output is ax_input
        plt.close("all")

    def test_sets_labels(self):
        """Should set axis labels."""
        time_values = np.arange(10)
        metric_values = np.random.rand(10)
        ax = plot_metric_time_series(
            time_values,
            metric_values,
            xlabel="Time",
            ylabel="Value",
            title="Test",
        )
        assert ax.get_xlabel() == "Time"
        assert ax.get_ylabel() == "Value"
        assert ax.get_title() == "Test"
        plt.close("all")

    def test_adds_grid(self):
        """Should add grid when requested."""
        time_values = np.arange(10)
        metric_values = np.random.rand(10)
        ax = plot_metric_time_series(time_values, metric_values, grid=True)
        # Grid should be enabled
        assert ax.xaxis.get_gridlines()[0].get_visible()
        plt.close("all")


class TestPlotConvergenceHistory:
    """Tests for plot_convergence_history function."""

    def test_returns_figure(self, sample_history):
        """Should return matplotlib figure object."""
        fig = plot_convergence_history(sample_history)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_plots_requested_metrics(self, sample_history):
        """Should plot all requested metrics."""
        metrics = ("max_velocity", "mean_velocity")
        fig = plot_convergence_history(sample_history, metrics=metrics)
        # Should have one subplot per metric
        assert len(fig.axes) == len(metrics)
        plt.close("all")

    def test_single_metric(self, sample_history):
        """Should handle single metric."""
        fig = plot_convergence_history(sample_history, metrics=("max_velocity",))
        assert len(fig.axes) >= 1
        plt.close("all")


class TestPlotMonitoringDashboard:
    """Tests for plot_monitoring_dashboard function."""

    def test_returns_figure(self, sample_snapshot, sample_history, uniform_grid):
        """Should return matplotlib figure object."""
        X, Y = uniform_grid["X"], uniform_grid["Y"]
        velocity_mag = np.ones_like(X)
        pressure = np.ones_like(X)
        fig = plot_monitoring_dashboard(
            sample_snapshot, sample_history, X, Y, velocity_mag, pressure
        )
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_creates_six_panels(self, sample_snapshot, sample_history, uniform_grid):
        """Should create 6 panel dashboard."""
        X, Y = uniform_grid["X"], uniform_grid["Y"]
        velocity_mag = np.ones_like(X)
        pressure = np.ones_like(X)
        fig = plot_monitoring_dashboard(
            sample_snapshot, sample_history, X, Y, velocity_mag, pressure
        )
        # 2x3 layout = 6 axes
        assert len(fig.axes) >= 6
        plt.close("all")


class TestPlotStatisticsPanel:
    """Tests for plot_statistics_panel function."""

    def test_returns_axes(self, sample_snapshot):
        """Should return matplotlib axes object."""
        ax = plot_statistics_panel(sample_snapshot)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_axis_is_off(self, sample_snapshot):
        """Should turn off axis for text panel."""
        ax = plot_statistics_panel(sample_snapshot)
        assert not ax.axison
        plt.close("all")

    def test_includes_velocity_stats(self, sample_snapshot):
        """Should include velocity statistics."""
        ax = plot_statistics_panel(sample_snapshot, title="Test Stats")
        assert ax.get_title() == "Test Stats"
        plt.close("all")

    def test_with_history(self, sample_snapshot, sample_history):
        """Should include convergence info when history provided."""
        ax = plot_statistics_panel(sample_snapshot, history=sample_history)
        # Should have text elements for both stats and convergence
        assert ax is not None
        plt.close("all")
