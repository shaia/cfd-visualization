"""Tests for cfd_viz.plotting.line_plots module."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

matplotlib.use("Agg")  # Non-interactive backend for testing

from cfd_viz.analysis.boundary_layer import BoundaryLayerProfile
from cfd_viz.analysis.flow_features import CrossSectionalAverages
from cfd_viz.analysis.line_extraction import CrossSection, LineProfile, MultipleProfiles
from cfd_viz.plotting.line_plots import (
    plot_boundary_layer_profile,
    plot_boundary_layer_profiles,
    plot_cross_sectional_averages,
    plot_line_profile,
    plot_multiple_profiles,
    plot_velocity_profiles,
)


@pytest.fixture
def sample_line_profile():
    """Create a sample LineProfile for testing."""
    distance = np.linspace(0, 1, 50)
    x_coords = np.linspace(0, 1, 50)
    y_coords = np.ones(50) * 0.5
    u = np.sin(np.pi * distance)
    v = np.cos(np.pi * distance) * 0.1
    velocity_mag = np.sqrt(u**2 + v**2)
    return LineProfile(
        distance=distance,
        x_coords=x_coords,
        y_coords=y_coords,
        u=u,
        v=v,
        velocity_mag=velocity_mag,
        pressure=1 - distance,
        start_point=(0, 0.5),
        end_point=(1, 0.5),
    )


@pytest.fixture
def sample_multiple_profiles():
    """Create sample MultipleProfiles for testing."""
    profiles = []
    positions = [0.25, 0.5, 0.75]
    for i, pos in enumerate(positions):
        coord = np.linspace(0, 1, 50)
        u = np.sin(np.pi * coord) * (1 + i * 0.2)
        v = np.zeros(50)
        velocity_mag = np.abs(u)
        profiles.append(
            CrossSection(
                position=pos,
                coordinate=coord,
                u=u,
                v=v,
                velocity_mag=velocity_mag,
                pressure=None,
                is_vertical=True,
            )
        )
    return MultipleProfiles(profiles=profiles, positions=positions, is_vertical=True)


@pytest.fixture
def sample_bl_profile():
    """Create a sample BoundaryLayerProfile for testing."""
    wall_distance = np.linspace(0, 0.1, 50)
    # Approximate BL profile
    delta_99 = 0.05
    u_edge = 1.0
    u = u_edge * np.tanh(3 * wall_distance / delta_99)
    return BoundaryLayerProfile(
        x_location=0.5,
        wall_distance=wall_distance,
        u=u,
        v=np.zeros(50),
        u_edge=u_edge,
        delta_99=delta_99,
        delta_star=delta_99 / 3,
        theta=delta_99 / 8,
        H=2.67,
        cf=0.005,
        Re_theta=None,
        Re_delta_star=None,
    )


@pytest.fixture
def sample_cross_averages():
    """Create sample CrossSectionalAverages for testing."""
    coord = np.linspace(0, 1, 50)
    return CrossSectionalAverages(
        coordinate=coord,
        u_avg=1 - 4 * (coord - 0.5) ** 2,  # Parabolic
        v_avg=np.zeros(50),
        velocity_mag_avg=1 - 4 * (coord - 0.5) ** 2,
        p_avg=np.linspace(1, 0, 50),
        averaging_axis="x",
    )


class TestPlotLineProfile:
    """Tests for plot_line_profile function."""

    def test_returns_axes(self, sample_line_profile):
        """Should return matplotlib axes object."""
        ax = plot_line_profile(sample_line_profile)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_uses_provided_axes(self, sample_line_profile):
        """Should plot on provided axes."""
        _, ax_input = plt.subplots()
        ax_output = plot_line_profile(sample_line_profile, ax=ax_input)
        assert ax_output is ax_input
        plt.close("all")

    def test_plots_components(self, sample_line_profile):
        """Should plot velocity components when requested."""
        ax = plot_line_profile(sample_line_profile, plot_components=True)
        # Check that lines were added
        assert len(ax.lines) >= 2
        plt.close("all")

    def test_plots_magnitude(self, sample_line_profile):
        """Should plot velocity magnitude when requested."""
        ax = plot_line_profile(
            sample_line_profile, plot_components=False, plot_magnitude=True
        )
        assert len(ax.lines) >= 1
        plt.close("all")


class TestPlotMultipleProfiles:
    """Tests for plot_multiple_profiles function."""

    def test_returns_axes(self, sample_multiple_profiles):
        """Should return matplotlib axes object."""
        ax = plot_multiple_profiles(sample_multiple_profiles)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_plots_all_profiles(self, sample_multiple_profiles):
        """Should plot all profiles."""
        ax = plot_multiple_profiles(sample_multiple_profiles)
        # Should have one line per profile
        assert len(ax.lines) == len(sample_multiple_profiles.profiles)
        plt.close("all")

    def test_plot_type_u(self, sample_multiple_profiles):
        """Should plot u-velocity when requested."""
        ax = plot_multiple_profiles(sample_multiple_profiles, plot_type="u")
        assert ax is not None
        plt.close("all")

    def test_plot_type_v(self, sample_multiple_profiles):
        """Should plot v-velocity when requested."""
        ax = plot_multiple_profiles(sample_multiple_profiles, plot_type="v")
        assert ax is not None
        plt.close("all")


class TestPlotVelocityProfiles:
    """Tests for plot_velocity_profiles function."""

    def test_returns_axes(self):
        """Should return matplotlib axes object."""
        x_stations = [0.2, 0.4, 0.6]
        y = np.linspace(0, 1, 50)
        u_profiles = [np.sin(np.pi * y) for _ in x_stations]
        ax = plot_velocity_profiles(x_stations, y, u_profiles)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_plots_all_stations(self):
        """Should plot profile at each station."""
        x_stations = [0.2, 0.4, 0.6, 0.8]
        y = np.linspace(0, 1, 50)
        u_profiles = [np.sin(np.pi * y) * (1 + i * 0.1) for i in range(4)]
        ax = plot_velocity_profiles(x_stations, y, u_profiles)
        assert len(ax.lines) == len(x_stations)
        plt.close("all")


class TestPlotBoundaryLayerProfile:
    """Tests for plot_boundary_layer_profile function."""

    def test_returns_axes(self, sample_bl_profile):
        """Should return matplotlib axes object."""
        ax = plot_boundary_layer_profile(sample_bl_profile)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_normalized_plot(self, sample_bl_profile):
        """Should normalize when requested."""
        ax = plot_boundary_layer_profile(sample_bl_profile, normalized=True)
        # Y-axis should be normalized (y/δ)
        assert ax.get_ylabel() == "y/δ"
        plt.close("all")

    def test_unnormalized_plot(self, sample_bl_profile):
        """Should not normalize when not requested."""
        ax = plot_boundary_layer_profile(sample_bl_profile, normalized=False)
        # Y-axis should be in meters
        assert "m" in ax.get_ylabel()
        plt.close("all")


class TestPlotBoundaryLayerProfiles:
    """Tests for plot_boundary_layer_profiles function."""

    def test_returns_axes(self, sample_bl_profile):
        """Should return matplotlib axes object."""
        profiles = [sample_bl_profile, sample_bl_profile]
        ax = plot_boundary_layer_profiles(profiles)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_plots_all_profiles(self, sample_bl_profile):
        """Should plot all profiles."""
        profiles = [sample_bl_profile, sample_bl_profile, sample_bl_profile]
        ax = plot_boundary_layer_profiles(profiles)
        assert len(ax.lines) == len(profiles)
        plt.close("all")


class TestPlotCrossSectionalAverages:
    """Tests for plot_cross_sectional_averages function."""

    def test_returns_axes(self, sample_cross_averages):
        """Should return matplotlib axes object."""
        ax = plot_cross_sectional_averages(sample_cross_averages)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_plots_components(self, sample_cross_averages):
        """Should plot u_avg and v_avg when requested."""
        ax = plot_cross_sectional_averages(
            sample_cross_averages, plot_components=True, plot_magnitude=False
        )
        assert len(ax.lines) == 2  # u_avg and v_avg
        plt.close("all")

    def test_includes_bulk_velocity_in_title(self, sample_cross_averages):
        """Should include bulk velocity in title."""
        ax = plot_cross_sectional_averages(sample_cross_averages)
        assert "bulk=" in ax.get_title()
        plt.close("all")
