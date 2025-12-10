"""Tests for cfd_viz.analysis.boundary_layer module."""

import numpy as np
import pytest

from cfd_viz.analysis.boundary_layer import (
    BoundaryLayerDevelopment,
    BoundaryLayerProfile,
    WallShearStress,
    analyze_boundary_layer,
    analyze_boundary_layer_development,
    blasius_solution,
    compute_displacement_thickness,
    compute_momentum_thickness,
    compute_wall_shear,
    compute_wall_shear_distribution,
    find_boundary_layer_edge,
)


@pytest.fixture
def flat_plate_flow(uniform_grid):
    """Create a simplified flat plate boundary layer flow."""
    _, Y = uniform_grid["X"], uniform_grid["Y"]
    # Simplified boundary layer: u increases from wall
    # u/U_inf = y/delta for y < delta, u = U_inf for y >= delta
    U_inf = 1.0
    delta = 0.2  # BL thickness

    u = np.where(Y < delta, U_inf * Y / delta, U_inf)
    v = np.zeros_like(Y)
    p = np.zeros_like(Y)

    return {"u": u, "v": v, "p": p, "U_inf": U_inf, "delta": delta, **uniform_grid}


class TestFindBoundaryLayerEdge:
    """Tests for find_boundary_layer_edge function."""

    def test_detects_edge_linear_profile(self):
        """Should detect BL edge for linear profile."""
        wall_distance = np.linspace(0, 1, 100)
        u_edge = 1.0
        # Linear profile from 0 to u_edge
        u = u_edge * wall_distance

        delta_99, detected_u_edge = find_boundary_layer_edge(wall_distance, u)
        # For linear profile, 99% is at y = 0.99
        np.testing.assert_almost_equal(delta_99, 0.99, decimal=2)
        np.testing.assert_almost_equal(detected_u_edge, u_edge, decimal=5)

    def test_uniform_profile_no_edge(self):
        """Uniform profile should have edge at boundary."""
        wall_distance = np.linspace(0, 1, 100)
        u = np.ones(100) * 1.0

        delta_99, u_edge = find_boundary_layer_edge(wall_distance, u)
        # Already at freestream everywhere
        assert delta_99 >= 0


class TestComputeDisplacementThickness:
    """Tests for compute_displacement_thickness function."""

    def test_uniform_flow_zero_displacement(self):
        """Uniform flow should have zero displacement thickness."""
        wall_distance = np.linspace(0, 1, 100)
        u = np.ones(100) * 1.0
        u_edge = 1.0

        delta_star = compute_displacement_thickness(wall_distance, u, u_edge)
        np.testing.assert_almost_equal(delta_star, 0, decimal=5)

    def test_linear_profile_displacement(self):
        """Linear profile should have known displacement thickness."""
        wall_distance = np.linspace(0, 1, 100)
        u_edge = 1.0
        # Linear profile: u = u_edge * y
        u = u_edge * wall_distance

        # delta* = integral(1 - y) dy from 0 to 1 = 0.5
        delta_star = compute_displacement_thickness(wall_distance, u, u_edge)
        np.testing.assert_almost_equal(delta_star, 0.5, decimal=2)

    def test_non_negative(self):
        """Displacement thickness should be non-negative."""
        wall_distance = np.linspace(0, 1, 100)
        u = np.random.rand(100)  # Random positive
        u_edge = np.max(u)

        delta_star = compute_displacement_thickness(wall_distance, u, u_edge)
        assert delta_star >= 0


class TestComputeMomentumThickness:
    """Tests for compute_momentum_thickness function."""

    def test_uniform_flow_zero_momentum(self):
        """Uniform flow should have zero momentum thickness."""
        wall_distance = np.linspace(0, 1, 100)
        u = np.ones(100) * 1.0
        u_edge = 1.0

        theta = compute_momentum_thickness(wall_distance, u, u_edge)
        np.testing.assert_almost_equal(theta, 0, decimal=5)

    def test_linear_profile_momentum(self):
        """Linear profile should have known momentum thickness."""
        wall_distance = np.linspace(0, 1, 100)
        u_edge = 1.0
        u = u_edge * wall_distance

        # theta = integral(y * (1 - y)) dy from 0 to 1 = 1/6
        theta = compute_momentum_thickness(wall_distance, u, u_edge)
        np.testing.assert_almost_equal(theta, 1 / 6, decimal=2)


class TestComputeWallShear:
    """Tests for compute_wall_shear function."""

    def test_linear_profile_shear(self):
        """Linear profile should have constant shear."""
        wall_distance = np.linspace(0, 1, 100)
        u_edge = 1.0
        u = u_edge * wall_distance
        mu = 0.1

        # du/dy = u_edge = 1, so tau_w = mu * 1 = 0.1
        tau_w = compute_wall_shear(u, wall_distance, mu)
        np.testing.assert_almost_equal(tau_w, mu * u_edge, decimal=2)


class TestAnalyzeBoundaryLayer:
    """Tests for analyze_boundary_layer function."""

    def test_returns_boundary_layer_profile(self, flat_plate_flow):
        """Should return BoundaryLayerProfile dataclass."""
        u, v = flat_plate_flow["u"], flat_plate_flow["v"]
        x, y = flat_plate_flow["x"], flat_plate_flow["y"]

        bl = analyze_boundary_layer(u, v, x, y, wall_y=0.0, x_location=0.5)
        assert isinstance(bl, BoundaryLayerProfile)

    def test_detects_boundary_layer_thickness(self, flat_plate_flow):
        """Should detect approximate boundary layer thickness."""
        u, v = flat_plate_flow["u"], flat_plate_flow["v"]
        x, y = flat_plate_flow["x"], flat_plate_flow["y"]
        expected_delta = flat_plate_flow["delta"]

        bl = analyze_boundary_layer(u, v, x, y, wall_y=0.0, x_location=0.5)
        # Should be close to expected (within grid resolution)
        np.testing.assert_almost_equal(bl.delta_99, expected_delta, decimal=1)

    def test_shape_factor_positive(self, flat_plate_flow):
        """Shape factor should be positive."""
        u, v = flat_plate_flow["u"], flat_plate_flow["v"]
        x, y = flat_plate_flow["x"], flat_plate_flow["y"]

        bl = analyze_boundary_layer(u, v, x, y, wall_y=0.0, x_location=0.5)
        assert bl.H > 0

    def test_computes_cf_when_mu_provided(self, flat_plate_flow):
        """Should compute cf when viscosity is provided."""
        u, v = flat_plate_flow["u"], flat_plate_flow["v"]
        x, y = flat_plate_flow["x"], flat_plate_flow["y"]

        bl = analyze_boundary_layer(u, v, x, y, wall_y=0.0, x_location=0.5, mu=0.01)
        assert bl.cf is not None

    def test_normalized_properties(self, flat_plate_flow):
        """Normalized properties should work correctly."""
        u, v = flat_plate_flow["u"], flat_plate_flow["v"]
        x, y = flat_plate_flow["x"], flat_plate_flow["y"]

        bl = analyze_boundary_layer(u, v, x, y, wall_y=0.0, x_location=0.5)

        # Normalized velocity should be ~1 at edge
        u_norm = bl.u_normalized
        assert np.max(u_norm) <= 1.1  # Allow small overshoot

        # Normalized y should be ~1 at delta
        y_norm = bl.y_normalized
        assert np.any(y_norm >= 0.9)


class TestAnalyzeBoundaryLayerDevelopment:
    """Tests for analyze_boundary_layer_development function."""

    def test_returns_development(self, flat_plate_flow):
        """Should return BoundaryLayerDevelopment dataclass."""
        u, v = flat_plate_flow["u"], flat_plate_flow["v"]
        x, y = flat_plate_flow["x"], flat_plate_flow["y"]

        dev = analyze_boundary_layer_development(u, v, x, y, wall_y=0.0, num_profiles=5)
        assert isinstance(dev, BoundaryLayerDevelopment)
        assert len(dev.profiles) == 5

    def test_profiles_at_specified_locations(self, flat_plate_flow):
        """Should have profiles at specified locations."""
        u, v = flat_plate_flow["u"], flat_plate_flow["v"]
        x, y = flat_plate_flow["x"], flat_plate_flow["y"]

        x_locs = [0.3, 0.5, 0.7]
        dev = analyze_boundary_layer_development(
            u, v, x, y, wall_y=0.0, x_locations=x_locs
        )
        assert len(dev.profiles) == 3

    def test_get_profile_at(self, flat_plate_flow):
        """get_profile_at should return closest profile."""
        u, v = flat_plate_flow["u"], flat_plate_flow["v"]
        x, y = flat_plate_flow["x"], flat_plate_flow["y"]

        dev = analyze_boundary_layer_development(u, v, x, y, wall_y=0.0, num_profiles=5)
        profile = dev.get_profile_at(0.5)
        assert profile is not None


class TestComputeWallShearDistribution:
    """Tests for compute_wall_shear_distribution function."""

    def test_returns_wall_shear_stress(self, flat_plate_flow):
        """Should return WallShearStress dataclass."""
        u = flat_plate_flow["u"]
        x, y = flat_plate_flow["x"], flat_plate_flow["y"]

        result = compute_wall_shear_distribution(u, x, y, wall_y=0.0, mu=0.01)
        assert isinstance(result, WallShearStress)

    def test_arrays_match_x_length(self, flat_plate_flow):
        """Output arrays should match x array length."""
        u = flat_plate_flow["u"]
        x, y = flat_plate_flow["x"], flat_plate_flow["y"]

        result = compute_wall_shear_distribution(u, x, y, wall_y=0.0, mu=0.01)
        assert len(result.tau_w) == len(x)
        assert len(result.cf) == len(x)
        assert len(result.u_tau) == len(x)


class TestBlasiusSolution:
    """Tests for blasius_solution function."""

    def test_returns_arrays(self):
        """Should return two arrays."""
        eta = np.linspace(0, 10, 100)
        f_prime, f_double_prime = blasius_solution(eta)
        assert len(f_prime) == len(eta)
        assert len(f_double_prime) == len(eta)

    def test_f_prime_bounded(self):
        """f' should be bounded between 0 and u_inf."""
        eta = np.linspace(0, 10, 100)
        u_inf = 2.0
        f_prime, _ = blasius_solution(eta, u_inf)
        assert np.all(f_prime >= 0)
        assert np.all(f_prime <= u_inf * 1.01)  # Allow small numerical error

    def test_approaches_freestream(self):
        """f' should approach u_inf for large eta."""
        eta = np.array([10.0])
        u_inf = 1.0
        f_prime, _ = blasius_solution(eta, u_inf)
        np.testing.assert_almost_equal(f_prime[0], u_inf, decimal=1)
