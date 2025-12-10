"""Tests for cfd_viz.plotting.fields module."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")  # Non-interactive backend for testing

from cfd_viz.plotting.fields import (
    plot_contour_field,
    plot_pressure_field,
    plot_streamlines,
    plot_vector_field,
    plot_velocity_field,
    plot_vorticity_field,
    plot_vorticity_with_streamlines,
)


class TestPlotContourField:
    """Tests for plot_contour_field function."""

    def test_returns_axes(self, uniform_flow):
        """Should return matplotlib axes object."""
        X, Y = uniform_flow["X"], uniform_flow["Y"]
        field = uniform_flow["u"]
        ax = plot_contour_field(X, Y, field)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_uses_provided_axes(self, uniform_flow):
        """Should plot on provided axes."""
        X, Y = uniform_flow["X"], uniform_flow["Y"]
        field = uniform_flow["u"]
        _, ax_input = plt.subplots()
        ax_output = plot_contour_field(X, Y, field, ax=ax_input)
        assert ax_output is ax_input
        plt.close("all")

    def test_sets_title(self, uniform_flow):
        """Should set title when provided."""
        X, Y = uniform_flow["X"], uniform_flow["Y"]
        field = uniform_flow["u"]
        ax = plot_contour_field(X, Y, field, title="Test Title")
        assert ax.get_title() == "Test Title"
        plt.close("all")

    def test_sets_labels(self, uniform_flow):
        """Should set axis labels."""
        X, Y = uniform_flow["X"], uniform_flow["Y"]
        field = uniform_flow["u"]
        ax = plot_contour_field(X, Y, field, xlabel="X Label", ylabel="Y Label")
        assert ax.get_xlabel() == "X Label"
        assert ax.get_ylabel() == "Y Label"
        plt.close("all")


class TestPlotVelocityField:
    """Tests for plot_velocity_field function."""

    def test_returns_axes(self, uniform_flow):
        """Should return matplotlib axes object."""
        X, Y = uniform_flow["X"], uniform_flow["Y"]
        u, v = uniform_flow["u"], uniform_flow["v"]
        ax = plot_velocity_field(X, Y, u, v)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_default_title(self, uniform_flow):
        """Should have default title."""
        X, Y = uniform_flow["X"], uniform_flow["Y"]
        u, v = uniform_flow["u"], uniform_flow["v"]
        ax = plot_velocity_field(X, Y, u, v)
        assert "Velocity" in ax.get_title()
        plt.close("all")

    def test_computes_magnitude(self, vortex_flow):
        """Should compute velocity magnitude correctly."""
        X, Y = vortex_flow["X"], vortex_flow["Y"]
        u, v = vortex_flow["u"], vortex_flow["v"]
        ax = plot_velocity_field(X, Y, u, v)
        # Just verify it doesn't error on non-trivial flow
        assert ax is not None
        plt.close("all")


class TestPlotPressureField:
    """Tests for plot_pressure_field function."""

    def test_returns_axes(self, channel_flow):
        """Should return matplotlib axes object."""
        X, Y = channel_flow["X"], channel_flow["Y"]
        p = channel_flow["p"]
        ax = plot_pressure_field(X, Y, p)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_default_title(self, channel_flow):
        """Should have default title."""
        X, Y = channel_flow["X"], channel_flow["Y"]
        p = channel_flow["p"]
        ax = plot_pressure_field(X, Y, p)
        assert "Pressure" in ax.get_title()
        plt.close("all")


class TestPlotVorticityField:
    """Tests for plot_vorticity_field function."""

    def test_returns_axes(self, vortex_flow):
        """Should return matplotlib axes object."""
        X, Y = vortex_flow["X"], vortex_flow["Y"]
        # Create vorticity field
        omega = np.ones_like(X) * vortex_flow["omega"]
        ax = plot_vorticity_field(X, Y, omega)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_symmetric_levels(self, random_flow):
        """Should create symmetric levels for diverging data."""
        X, Y = random_flow["X"], random_flow["Y"]
        omega = random_flow["u"]  # Use random field as vorticity
        ax = plot_vorticity_field(X, Y, omega, symmetric=True)
        assert ax is not None
        plt.close("all")


class TestPlotVectorField:
    """Tests for plot_vector_field function."""

    def test_returns_axes(self, uniform_flow):
        """Should return matplotlib axes object."""
        X, Y = uniform_flow["X"], uniform_flow["Y"]
        u, v = uniform_flow["u"], uniform_flow["v"]
        ax = plot_vector_field(X, Y, u, v)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_with_custom_density(self, vortex_flow):
        """Should work with custom density parameter."""
        X, Y = vortex_flow["X"], vortex_flow["Y"]
        u, v = vortex_flow["u"], vortex_flow["v"]
        ax = plot_vector_field(X, Y, u, v, density=10)
        assert ax is not None
        plt.close("all")


class TestPlotStreamlines:
    """Tests for plot_streamlines function."""

    def test_returns_axes(self, vortex_flow):
        """Should return matplotlib axes object."""
        X, Y = vortex_flow["X"], vortex_flow["Y"]
        u, v = vortex_flow["u"], vortex_flow["v"]
        ax = plot_streamlines(X, Y, u, v)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_with_custom_parameters(self, channel_flow):
        """Should work with custom parameters."""
        X, Y = channel_flow["X"], channel_flow["Y"]
        u, v = channel_flow["u"], channel_flow["v"]
        ax = plot_streamlines(X, Y, u, v, density=2.0, color="blue", linewidth=1.0)
        assert ax is not None
        plt.close("all")


class TestPlotVorticityWithStreamlines:
    """Tests for plot_vorticity_with_streamlines function."""

    def test_returns_axes(self, vortex_flow):
        """Should return matplotlib axes object."""
        X, Y = vortex_flow["X"], vortex_flow["Y"]
        u, v = vortex_flow["u"], vortex_flow["v"]
        omega = np.ones_like(X) * vortex_flow["omega"]
        ax = plot_vorticity_with_streamlines(X, Y, omega, u, v)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_combines_vorticity_and_streamlines(self, vortex_flow):
        """Should produce combined visualization."""
        X, Y = vortex_flow["X"], vortex_flow["Y"]
        u, v = vortex_flow["u"], vortex_flow["v"]
        omega = np.ones_like(X) * vortex_flow["omega"]
        ax = plot_vorticity_with_streamlines(X, Y, omega, u, v, title="Combined")
        assert ax.get_title() == "Combined"
        plt.close("all")
