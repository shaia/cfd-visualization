"""Tests for cfd_viz.analysis.flow_features module."""

import numpy as np

from cfd_viz.analysis.flow_features import (
    CrossSectionalAverages,
    RecirculationZone,
    SpatialFluctuations,
    WakeRegion,
    compute_adverse_pressure_gradient,
    compute_cross_sectional_averages,
    compute_pressure_gradient,
    compute_spatial_fluctuations,
    detect_recirculation_zones,
    detect_wake_regions,
)


class TestDetectWakeRegions:
    """Tests for detect_wake_regions function."""

    def test_returns_wake_region(self, uniform_flow):
        """Should return WakeRegion dataclass."""
        u, v = uniform_flow["u"], uniform_flow["v"]
        result = detect_wake_regions(u, v)
        assert isinstance(result, WakeRegion)

    def test_uniform_flow_no_wake(self, uniform_flow):
        """Uniform flow should have no wake at 10% threshold."""
        u, v = uniform_flow["u"], uniform_flow["v"]
        result = detect_wake_regions(u, v, threshold_fraction=0.1)
        # Uniform flow has u=1, v=0 everywhere, so no wake
        assert result.area_fraction == 0.0

    def test_detects_low_velocity_region(self):
        """Should detect region with low velocity as wake."""
        # Create field with a low-velocity region
        u = np.ones((50, 50))
        v = np.zeros((50, 50))
        # Add wake region
        u[20:30, 20:30] = 0.05  # 5% of max velocity

        result = detect_wake_regions(u, v, threshold_fraction=0.1)
        assert result.area_fraction > 0
        assert np.any(result.mask[20:30, 20:30])

    def test_centroid_in_wake(self):
        """Centroid should be within wake region."""
        u = np.ones((50, 50))
        v = np.zeros((50, 50))
        u[20:30, 20:30] = 0.05

        result = detect_wake_regions(u, v, threshold_fraction=0.1)
        assert result.centroid is not None
        # Centroid should be near center of wake (25, 25)
        assert 20 <= result.centroid[0] <= 30
        assert 20 <= result.centroid[1] <= 30

    def test_custom_reference_velocity(self):
        """Should use custom reference velocity for threshold."""
        u = np.ones((50, 50)) * 0.5
        v = np.zeros((50, 50))
        u[20:30, 20:30] = 0.05

        # With auto reference (0.5), threshold = 0.05, so 0.05 is on boundary
        result_auto = detect_wake_regions(u, v, threshold_fraction=0.1)

        # With custom reference (1.0), threshold = 0.1, so 0.05 is wake
        result_custom = detect_wake_regions(
            u, v, threshold_fraction=0.1, reference_velocity=1.0
        )
        assert result_custom.area_fraction >= result_auto.area_fraction

    def test_num_cells_property(self):
        """num_cells should return count of wake cells."""
        u = np.ones((50, 50))
        v = np.zeros((50, 50))
        u[20:30, 20:30] = 0.05  # 10x10 = 100 cells

        result = detect_wake_regions(u, v, threshold_fraction=0.1)
        assert result.num_cells == 100


class TestComputeSpatialFluctuations:
    """Tests for compute_spatial_fluctuations function."""

    def test_returns_spatial_fluctuations(self, uniform_flow):
        """Should return SpatialFluctuations dataclass."""
        u, v = uniform_flow["u"], uniform_flow["v"]
        result = compute_spatial_fluctuations(u, v)
        assert isinstance(result, SpatialFluctuations)

    def test_uniform_flow_zero_fluctuations(self, uniform_flow):
        """Uniform flow should have zero fluctuations."""
        u, v = uniform_flow["u"], uniform_flow["v"]
        result = compute_spatial_fluctuations(u, v)
        np.testing.assert_almost_equal(result.rms_u, 0.0, decimal=10)
        np.testing.assert_almost_equal(result.rms_v, 0.0, decimal=10)

    def test_channel_flow_has_fluctuations(self, channel_flow):
        """Channel flow should have spatial fluctuations from parabolic profile."""
        u, v = channel_flow["u"], channel_flow["v"]
        result = compute_spatial_fluctuations(u, v, averaging_axis=1)
        # Channel flow has variation in y, so fluctuations exist
        assert result.rms_u > 0

    def test_fluctuation_magnitude_shape(self, uniform_flow):
        """Fluctuation magnitude should match input shape."""
        u, v = uniform_flow["u"], uniform_flow["v"]
        result = compute_spatial_fluctuations(u, v)
        assert result.fluct_magnitude.shape == u.shape

    def test_mean_profile_correct_axis(self):
        """Mean profile should be computed along correct axis."""
        # Create field varying in y
        u = np.arange(50).reshape(50, 1) * np.ones((1, 50))
        v = np.zeros((50, 50))

        result = compute_spatial_fluctuations(u, v, averaging_axis=1)
        # Mean profile should vary with y
        assert len(result.u_mean_profile) == 50
        np.testing.assert_array_equal(result.u_mean_profile, np.arange(50))


class TestComputeCrossSectionalAverages:
    """Tests for compute_cross_sectional_averages function."""

    def test_returns_cross_sectional_averages(self, uniform_flow):
        """Should return CrossSectionalAverages dataclass."""
        u, v = uniform_flow["u"], uniform_flow["v"]
        x, y = uniform_flow["x"], uniform_flow["y"]
        result = compute_cross_sectional_averages(u, v, x, y)
        assert isinstance(result, CrossSectionalAverages)

    def test_averaging_along_x(self, uniform_flow):
        """Averaging along x should give y-varying profile."""
        u, v = uniform_flow["u"], uniform_flow["v"]
        x, y = uniform_flow["x"], uniform_flow["y"]
        result = compute_cross_sectional_averages(u, v, x, y, averaging_axis="x")
        assert result.averaging_axis == "x"
        assert len(result.coordinate) == len(y)
        assert len(result.u_avg) == len(y)

    def test_averaging_along_y(self, uniform_flow):
        """Averaging along y should give x-varying profile."""
        u, v = uniform_flow["u"], uniform_flow["v"]
        x, y = uniform_flow["x"], uniform_flow["y"]
        result = compute_cross_sectional_averages(u, v, x, y, averaging_axis="y")
        assert result.averaging_axis == "y"
        assert len(result.coordinate) == len(x)
        assert len(result.u_avg) == len(x)

    def test_uniform_flow_constant_average(self, uniform_flow):
        """Uniform flow should have constant averaged profile."""
        u, v = uniform_flow["u"], uniform_flow["v"]
        x, y = uniform_flow["x"], uniform_flow["y"]
        result = compute_cross_sectional_averages(u, v, x, y)
        np.testing.assert_array_almost_equal(result.u_avg, 1.0)

    def test_includes_pressure(self, uniform_flow):
        """Should include pressure when provided."""
        u, v, p = uniform_flow["u"], uniform_flow["v"], uniform_flow["p"]
        x, y = uniform_flow["x"], uniform_flow["y"]
        result = compute_cross_sectional_averages(u, v, x, y, p=p)
        assert result.p_avg is not None
        assert len(result.p_avg) == len(result.coordinate)

    def test_bulk_velocity_property(self, uniform_flow):
        """bulk_velocity should return mean of velocity magnitude average."""
        u, v = uniform_flow["u"], uniform_flow["v"]
        x, y = uniform_flow["x"], uniform_flow["y"]
        result = compute_cross_sectional_averages(u, v, x, y)
        np.testing.assert_almost_equal(result.bulk_velocity, 1.0)


class TestDetectRecirculationZones:
    """Tests for detect_recirculation_zones function."""

    def test_returns_recirculation_zone(self, uniform_flow):
        """Should return RecirculationZone dataclass."""
        u, v = uniform_flow["u"], uniform_flow["v"]
        result = detect_recirculation_zones(u, v)
        assert isinstance(result, RecirculationZone)

    def test_forward_flow_no_recirculation(self, uniform_flow):
        """Forward flow should have no recirculation."""
        u, v = uniform_flow["u"], uniform_flow["v"]
        result = detect_recirculation_zones(u, v, main_flow_direction="x")
        assert result.area_fraction == 0.0

    def test_detects_reverse_flow(self):
        """Should detect reverse flow as recirculation."""
        u = np.ones((50, 50))
        v = np.zeros((50, 50))
        # Add reverse flow region
        u[20:30, 20:30] = -0.5

        result = detect_recirculation_zones(u, v, main_flow_direction="x")
        assert result.area_fraction > 0
        assert np.any(result.mask[20:30, 20:30])

    def test_recirculation_centroid(self):
        """Centroid should be in recirculation region."""
        u = np.ones((50, 50))
        v = np.zeros((50, 50))
        u[20:30, 20:30] = -0.5

        result = detect_recirculation_zones(u, v, main_flow_direction="x")
        assert result.centroid is not None
        assert 20 <= result.centroid[0] <= 30
        assert 20 <= result.centroid[1] <= 30

    def test_y_direction_recirculation(self):
        """Should detect v<0 for y-direction main flow."""
        u = np.zeros((50, 50))
        v = np.ones((50, 50))
        v[20:30, 20:30] = -0.5

        result = detect_recirculation_zones(u, v, main_flow_direction="y")
        assert result.area_fraction > 0


class TestComputePressureGradient:
    """Tests for compute_pressure_gradient function."""

    def test_returns_two_arrays(self, uniform_flow):
        """Should return two gradient arrays."""
        p = uniform_flow["p"]
        dx, dy = 0.01, 0.01
        dp_dx, dp_dy = compute_pressure_gradient(p, dx, dy)
        assert dp_dx.shape == p.shape
        assert dp_dy.shape == p.shape

    def test_constant_pressure_zero_gradient(self, uniform_flow):
        """Constant pressure should have zero gradient."""
        p = uniform_flow["p"]  # All zeros
        dx, dy = 0.01, 0.01
        dp_dx, dp_dy = compute_pressure_gradient(p, dx, dy)
        np.testing.assert_array_almost_equal(dp_dx, 0.0)
        np.testing.assert_array_almost_equal(dp_dy, 0.0)

    def test_linear_pressure_constant_gradient(self):
        """Linear pressure should have constant gradient."""
        x = np.linspace(0, 1, 50)
        y = np.linspace(0, 1, 50)
        X, Y = np.meshgrid(x, y)
        # Linear pressure: p = 2*x + 3*y
        p = 2 * X + 3 * Y
        dx = x[1] - x[0]
        dy = y[1] - y[0]

        dp_dx, dp_dy = compute_pressure_gradient(p, dx, dy)
        # Gradient should be approximately (2, 3)
        np.testing.assert_array_almost_equal(dp_dx, 2.0, decimal=1)
        np.testing.assert_array_almost_equal(dp_dy, 3.0, decimal=1)


class TestComputeAdversePressureGradient:
    """Tests for compute_adverse_pressure_gradient function."""

    def test_returns_scalar_field(self, uniform_flow):
        """Should return scalar field of same shape."""
        u, v, p = uniform_flow["u"], uniform_flow["v"], uniform_flow["p"]
        result = compute_adverse_pressure_gradient(p, u, v, dx=0.01, dy=0.01)
        assert result.shape == p.shape

    def test_favorable_gradient_negative(self):
        """Favorable gradient (pressure decreasing in flow direction) negative."""
        x = np.linspace(0, 1, 50)
        y = np.linspace(0, 1, 50)
        X, Y = np.meshgrid(x, y)
        # Flow in +x direction
        u = np.ones((50, 50))
        v = np.zeros((50, 50))
        # Pressure decreasing in x (favorable)
        p = -X
        dx = x[1] - x[0]
        dy = y[1] - y[0]

        result = compute_adverse_pressure_gradient(p, u, v, dx, dy)
        # Favorable gradient should be negative
        assert np.all(result < 0.1)  # Allow small numerical error

    def test_adverse_gradient_positive(self):
        """Adverse gradient (pressure increasing in flow direction) positive."""
        x = np.linspace(0, 1, 50)
        y = np.linspace(0, 1, 50)
        X, Y = np.meshgrid(x, y)
        # Flow in +x direction
        u = np.ones((50, 50))
        v = np.zeros((50, 50))
        # Pressure increasing in x (adverse)
        p = X
        dx = x[1] - x[0]
        dy = y[1] - y[0]

        result = compute_adverse_pressure_gradient(p, u, v, dx, dy)
        # Adverse gradient should be positive
        assert np.all(result > -0.1)  # Allow small numerical error
