"""Tests for cfd_viz.analysis.line_extraction module."""

import numpy as np
import pytest

from cfd_viz.analysis.line_extraction import (
    CrossSection,
    LineProfile,
    MultipleProfiles,
    compute_centerline_profiles,
    compute_mass_flow_rate,
    compute_profile_statistics,
    extract_horizontal_profile,
    extract_line_profile,
    extract_multiple_profiles,
    extract_vertical_profile,
)


class TestExtractLineProfile:
    """Tests for extract_line_profile function."""

    def test_returns_line_profile(self, uniform_flow):
        """Should return a LineProfile dataclass."""
        u, v = uniform_flow["u"], uniform_flow["v"]
        x, y = uniform_flow["x"], uniform_flow["y"]
        profile = extract_line_profile(u, v, x, y, (0, 0.5), (1, 0.5))
        assert isinstance(profile, LineProfile)

    def test_horizontal_line_length(self, uniform_flow):
        """Horizontal line should have correct length."""
        u, v = uniform_flow["u"], uniform_flow["v"]
        x, y = uniform_flow["x"], uniform_flow["y"]
        profile = extract_line_profile(u, v, x, y, (0, 0.5), (1, 0.5))
        np.testing.assert_almost_equal(profile.length, 1.0)

    def test_vertical_line_length(self, uniform_flow):
        """Vertical line should have correct length."""
        u, v = uniform_flow["u"], uniform_flow["v"]
        x, y = uniform_flow["x"], uniform_flow["y"]
        profile = extract_line_profile(u, v, x, y, (0.5, 0), (0.5, 1))
        np.testing.assert_almost_equal(profile.length, 1.0)

    def test_uniform_flow_constant_velocity(self, uniform_flow):
        """Uniform flow should have constant velocity along any line."""
        u, v = uniform_flow["u"], uniform_flow["v"]
        x, y = uniform_flow["x"], uniform_flow["y"]
        profile = extract_line_profile(u, v, x, y, (0.1, 0.1), (0.9, 0.9))
        # u=1, v=0 everywhere
        np.testing.assert_array_almost_equal(profile.velocity_mag, 1.0, decimal=2)

    def test_includes_pressure_when_provided(self, uniform_flow):
        """Should include pressure when provided."""
        u, v, p = uniform_flow["u"], uniform_flow["v"], uniform_flow["p"]
        x, y = uniform_flow["x"], uniform_flow["y"]
        profile = extract_line_profile(u, v, x, y, (0, 0.5), (1, 0.5), p=p)
        assert profile.pressure is not None
        assert len(profile.pressure) == len(profile.distance)

    def test_stores_endpoints(self, uniform_flow):
        """Should store start and end points."""
        u, v = uniform_flow["u"], uniform_flow["v"]
        x, y = uniform_flow["x"], uniform_flow["y"]
        start = (0.2, 0.3)
        end = (0.8, 0.7)
        profile = extract_line_profile(u, v, x, y, start, end)
        assert profile.start_point == start
        assert profile.end_point == end


class TestExtractVerticalProfile:
    """Tests for extract_vertical_profile function."""

    def test_returns_cross_section(self, uniform_flow):
        """Should return a CrossSection dataclass."""
        u, v = uniform_flow["u"], uniform_flow["v"]
        x, y = uniform_flow["x"], uniform_flow["y"]
        profile = extract_vertical_profile(u, v, x, y, x_position=0.5)
        assert isinstance(profile, CrossSection)

    def test_is_vertical_flag(self, uniform_flow):
        """is_vertical should be True."""
        u, v = uniform_flow["u"], uniform_flow["v"]
        x, y = uniform_flow["x"], uniform_flow["y"]
        profile = extract_vertical_profile(u, v, x, y, x_position=0.5)
        assert profile.is_vertical is True

    def test_coordinate_is_y(self, uniform_flow):
        """Coordinate should be y values."""
        u, v = uniform_flow["u"], uniform_flow["v"]
        x, y = uniform_flow["x"], uniform_flow["y"]
        profile = extract_vertical_profile(u, v, x, y, x_position=0.5)
        np.testing.assert_array_almost_equal(profile.coordinate, y)

    def test_channel_flow_parabolic_profile(self, channel_flow):
        """Channel flow should have parabolic velocity profile."""
        u, v = channel_flow["u"], channel_flow["v"]
        x, y = channel_flow["x"], channel_flow["y"]
        profile = extract_vertical_profile(u, v, x, y, x_position=0.5)
        # Max should be at center
        center_idx = len(profile.coordinate) // 2
        max_idx = np.argmax(profile.u)
        assert abs(max_idx - center_idx) <= 2


class TestExtractHorizontalProfile:
    """Tests for extract_horizontal_profile function."""

    def test_returns_cross_section(self, uniform_flow):
        """Should return a CrossSection dataclass."""
        u, v = uniform_flow["u"], uniform_flow["v"]
        x, y = uniform_flow["x"], uniform_flow["y"]
        profile = extract_horizontal_profile(u, v, x, y, y_position=0.5)
        assert isinstance(profile, CrossSection)

    def test_is_vertical_flag_false(self, uniform_flow):
        """is_vertical should be False."""
        u, v = uniform_flow["u"], uniform_flow["v"]
        x, y = uniform_flow["x"], uniform_flow["y"]
        profile = extract_horizontal_profile(u, v, x, y, y_position=0.5)
        assert profile.is_vertical is False

    def test_coordinate_is_x(self, uniform_flow):
        """Coordinate should be x values."""
        u, v = uniform_flow["u"], uniform_flow["v"]
        x, y = uniform_flow["x"], uniform_flow["y"]
        profile = extract_horizontal_profile(u, v, x, y, y_position=0.5)
        np.testing.assert_array_almost_equal(profile.coordinate, x)


class TestExtractMultipleProfiles:
    """Tests for extract_multiple_profiles function."""

    def test_returns_multiple_profiles(self, uniform_flow):
        """Should return MultipleProfiles dataclass."""
        u, v = uniform_flow["u"], uniform_flow["v"]
        x, y = uniform_flow["x"], uniform_flow["y"]
        positions = [0.25, 0.5, 0.75]
        result = extract_multiple_profiles(u, v, x, y, positions, vertical=True)
        assert isinstance(result, MultipleProfiles)
        assert len(result.profiles) == 3

    def test_get_profile_at(self, uniform_flow):
        """get_profile_at should return closest profile."""
        u, v = uniform_flow["u"], uniform_flow["v"]
        x, y = uniform_flow["x"], uniform_flow["y"]
        positions = [0.25, 0.5, 0.75]
        result = extract_multiple_profiles(u, v, x, y, positions, vertical=True)
        profile = result.get_profile_at(0.52)  # Should get the 0.5 profile
        assert profile is not None
        np.testing.assert_almost_equal(profile.position, 0.5, decimal=1)


class TestComputeCenterlineProfiles:
    """Tests for compute_centerline_profiles function."""

    def test_returns_two_profiles(self, uniform_flow):
        """Should return horizontal and vertical centerlines."""
        u, v = uniform_flow["u"], uniform_flow["v"]
        x, y = uniform_flow["x"], uniform_flow["y"]
        horizontal, vertical = compute_centerline_profiles(u, v, x, y)
        assert horizontal.is_vertical is False
        assert vertical.is_vertical is True


class TestComputeProfileStatistics:
    """Tests for compute_profile_statistics function."""

    def test_returns_dict(self, uniform_flow):
        """Should return a dictionary of statistics."""
        u, v = uniform_flow["u"], uniform_flow["v"]
        x, y = uniform_flow["x"], uniform_flow["y"]
        profile = extract_vertical_profile(u, v, x, y, x_position=0.5)
        stats = compute_profile_statistics(profile)
        assert isinstance(stats, dict)
        assert "max_velocity" in stats
        assert "mean_velocity" in stats

    def test_uniform_flow_statistics(self, uniform_flow):
        """Uniform flow should have predictable statistics."""
        u, v = uniform_flow["u"], uniform_flow["v"]
        x, y = uniform_flow["x"], uniform_flow["y"]
        profile = extract_vertical_profile(u, v, x, y, x_position=0.5)
        stats = compute_profile_statistics(profile)
        np.testing.assert_almost_equal(stats["max_velocity"], 1.0)
        np.testing.assert_almost_equal(stats["mean_velocity"], 1.0)


class TestComputeMassFlowRate:
    """Tests for compute_mass_flow_rate function."""

    def test_uniform_flow(self, uniform_flow):
        """Uniform flow should have predictable mass flow rate."""
        u, v = uniform_flow["u"], uniform_flow["v"]
        x, y = uniform_flow["x"], uniform_flow["y"]
        # Vertical profile - u is normal velocity
        profile = extract_vertical_profile(u, v, x, y, x_position=0.5)
        # u=1 everywhere, height=1, width=1, rho=1 -> mass_flow = 1
        mass_flow = compute_mass_flow_rate(profile, rho=1.0, width=1.0)
        # Allow small numerical error due to grid spacing
        np.testing.assert_almost_equal(mass_flow, 1.0, decimal=1)


class TestCrossSectionProperties:
    """Tests for CrossSection dataclass properties."""

    def test_bulk_velocity(self, channel_flow):
        """bulk_velocity should return mean."""
        u, v = channel_flow["u"], channel_flow["v"]
        x, y = channel_flow["x"], channel_flow["y"]
        profile = extract_vertical_profile(u, v, x, y, x_position=0.5)
        # Bulk velocity should be mean of velocity magnitude
        expected = np.mean(profile.velocity_mag)
        assert profile.bulk_velocity == pytest.approx(expected)

    def test_max_velocity(self, channel_flow):
        """max_velocity should return maximum."""
        u, v = channel_flow["u"], channel_flow["v"]
        x, y = channel_flow["x"], channel_flow["y"]
        profile = extract_vertical_profile(u, v, x, y, x_position=0.5)
        expected = np.max(profile.velocity_mag)
        assert profile.max_velocity == pytest.approx(expected)
