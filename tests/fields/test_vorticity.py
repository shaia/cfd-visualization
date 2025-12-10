"""Tests for cfd_viz.fields.vorticity module."""

import numpy as np

from cfd_viz.fields.vorticity import (
    circulation,
    detect_vortex_cores,
    enstrophy,
    lambda2_criterion,
    q_criterion,
    vorticity,
    vorticity_from_gradients,
)


class TestVorticity:
    """Tests for vorticity.vorticity function."""

    def test_uniform_flow_zero_vorticity(self, uniform_flow):
        """Uniform flow should have zero vorticity."""
        u, v = uniform_flow["u"], uniform_flow["v"]
        dx, dy = uniform_flow["dx"], uniform_flow["dy"]
        omega = vorticity(u, v, dx, dy)
        # Interior points should be zero (boundary effects at edges)
        np.testing.assert_array_almost_equal(omega[2:-2, 2:-2], 0, decimal=10)

    def test_shear_flow_constant_vorticity(self, shear_flow):
        """Linear shear flow u=y should have constant vorticity = -1."""
        u, v = shear_flow["u"], shear_flow["v"]
        dx, dy = shear_flow["dx"], shear_flow["dy"]
        omega = vorticity(u, v, dx, dy)
        # omega = dv/dx - du/dy = 0 - 1 = -1
        # Check interior points
        np.testing.assert_array_almost_equal(omega[2:-2, 2:-2], -1.0, decimal=5)

    def test_solid_body_rotation(self, vortex_flow):
        """Solid body rotation should have constant vorticity = 2*omega."""
        u, v = vortex_flow["u"], vortex_flow["v"]
        dx, dy = vortex_flow["dx"], vortex_flow["dy"]
        angular_vel = vortex_flow["omega"]
        omega = vorticity(u, v, dx, dy)
        # For solid body rotation, vorticity = 2 * angular_velocity
        expected = 2 * angular_vel
        # Check interior points
        np.testing.assert_array_almost_equal(omega[5:-5, 5:-5], expected, decimal=3)

    def test_preserves_shape(self, random_flow):
        """Output should have same shape as input."""
        u, v = random_flow["u"], random_flow["v"]
        dx, dy = random_flow["dx"], random_flow["dy"]
        omega = vorticity(u, v, dx, dy)
        assert omega.shape == u.shape


class TestVorticityFromGradients:
    """Tests for vorticity.vorticity_from_gradients function."""

    def test_matches_vorticity_function(self, random_flow):
        """Should give same result as vorticity() when using same gradients."""
        u, v = random_flow["u"], random_flow["v"]
        dx, dy = random_flow["dx"], random_flow["dy"]

        # Compute gradients manually
        du_dy = np.gradient(u, dy, axis=0)
        dv_dx = np.gradient(v, dx, axis=1)

        omega1 = vorticity(u, v, dx, dy)
        omega2 = vorticity_from_gradients(du_dy, dv_dx)

        np.testing.assert_array_almost_equal(omega1, omega2)


class TestQCriterion:
    """Tests for vorticity.q_criterion function."""

    def test_uniform_flow_zero_q(self, uniform_flow):
        """Uniform flow should have Q = 0."""
        u, v = uniform_flow["u"], uniform_flow["v"]
        dx, dy = uniform_flow["dx"], uniform_flow["dy"]
        Q = q_criterion(u, v, dx, dy)
        np.testing.assert_array_almost_equal(Q[2:-2, 2:-2], 0, decimal=10)

    def test_solid_body_rotation_positive_q(self, vortex_flow):
        """Solid body rotation (vortex) should have positive Q."""
        u, v = vortex_flow["u"], vortex_flow["v"]
        dx, dy = vortex_flow["dx"], vortex_flow["dy"]
        Q = q_criterion(u, v, dx, dy)
        # Q > 0 indicates rotation dominates over strain
        assert np.mean(Q[5:-5, 5:-5]) > 0

    def test_stagnation_flow_negative_q(self, stagnation_flow):
        """Stagnation flow should have negative Q (strain dominates)."""
        u, v = stagnation_flow["u"], stagnation_flow["v"]
        dx, dy = stagnation_flow["dx"], stagnation_flow["dy"]
        Q = q_criterion(u, v, dx, dy)
        # Q < 0 indicates strain dominates over rotation
        assert np.mean(Q[5:-5, 5:-5]) < 0

    def test_preserves_shape(self, random_flow):
        """Output should have same shape as input."""
        u, v = random_flow["u"], random_flow["v"]
        dx, dy = random_flow["dx"], random_flow["dy"]
        Q = q_criterion(u, v, dx, dy)
        assert Q.shape == u.shape


class TestLambda2Criterion:
    """Tests for vorticity.lambda2_criterion function."""

    def test_solid_body_rotation_negative_lambda2(self, vortex_flow):
        """Solid body rotation should have negative lambda2 (vortex core)."""
        u, v = vortex_flow["u"], vortex_flow["v"]
        dx, dy = vortex_flow["dx"], vortex_flow["dy"]
        lam2 = lambda2_criterion(u, v, dx, dy)
        # Lambda2 < 0 indicates vortex core
        assert np.mean(lam2[5:-5, 5:-5]) < 0

    def test_preserves_shape(self, random_flow):
        """Output should have same shape as input."""
        u, v = random_flow["u"], random_flow["v"]
        dx, dy = random_flow["dx"], random_flow["dy"]
        lam2 = lambda2_criterion(u, v, dx, dy)
        assert lam2.shape == u.shape


class TestEnstrophy:
    """Tests for vorticity.enstrophy function."""

    def test_uniform_flow_zero_enstrophy(self, uniform_flow):
        """Uniform flow should have zero enstrophy."""
        u, v = uniform_flow["u"], uniform_flow["v"]
        dx, dy = uniform_flow["dx"], uniform_flow["dy"]
        ens = enstrophy(u, v, dx, dy)
        np.testing.assert_array_almost_equal(ens[2:-2, 2:-2], 0, decimal=10)

    def test_enstrophy_is_half_vorticity_squared(self, random_flow):
        """Enstrophy should equal 0.5 * omega^2."""
        u, v = random_flow["u"], random_flow["v"]
        dx, dy = random_flow["dx"], random_flow["dy"]
        omega = vorticity(u, v, dx, dy)
        ens = enstrophy(u, v, dx, dy)
        expected = 0.5 * omega**2
        np.testing.assert_array_almost_equal(ens, expected)

    def test_always_non_negative(self, random_flow):
        """Enstrophy should always be non-negative."""
        u, v = random_flow["u"], random_flow["v"]
        dx, dy = random_flow["dx"], random_flow["dy"]
        ens = enstrophy(u, v, dx, dy)
        assert np.all(ens >= 0)


class TestCirculation:
    """Tests for vorticity.circulation function."""

    def test_solid_body_rotation(self, vortex_flow):
        """Circulation around solid body rotation = 2*pi*r^2*omega."""
        u, v = vortex_flow["u"], vortex_flow["v"]
        x, y = vortex_flow["x"], vortex_flow["y"]
        angular_vel = vortex_flow["omega"]

        center = (0.5, 0.5)
        radius = 0.2

        gamma = circulation(u, v, x, y, center, radius)

        # For solid body rotation: Gamma = integral of omega dA = omega * pi * r^2
        # But circulation around path = vorticity * enclosed area = 2*omega * pi * r^2
        expected = 2 * angular_vel * np.pi * radius**2
        np.testing.assert_almost_equal(gamma, expected, decimal=2)

    def test_uniform_flow_zero_circulation(self, uniform_flow):
        """Uniform flow should have zero circulation."""
        u, v = uniform_flow["u"], uniform_flow["v"]
        x, y = uniform_flow["x"], uniform_flow["y"]

        center = (0.5, 0.5)
        radius = 0.2

        gamma = circulation(u, v, x, y, center, radius)
        np.testing.assert_almost_equal(gamma, 0, decimal=5)


class TestDetectVortexCores:
    """Tests for vorticity.detect_vortex_cores function."""

    def test_returns_boolean_array(self, vortex_flow):
        """Should return a boolean array."""
        u, v = vortex_flow["u"], vortex_flow["v"]
        dx, dy = vortex_flow["dx"], vortex_flow["dy"]

        omega = vorticity(u, v, dx, dy)
        Q = q_criterion(u, v, dx, dy)

        cores = detect_vortex_cores(omega, Q)
        assert cores.dtype == bool

    def test_detects_vortex_in_rotating_flow(self, vortex_flow):
        """Should detect vortex core in solid body rotation."""
        u, v = vortex_flow["u"], vortex_flow["v"]
        dx, dy = vortex_flow["dx"], vortex_flow["dy"]

        omega = vorticity(u, v, dx, dy)
        Q = q_criterion(u, v, dx, dy)

        cores = detect_vortex_cores(omega, Q)
        # Should have some vortex core detected
        assert np.any(cores)

    def test_uniform_flow_no_vortex(self, uniform_flow):
        """Uniform flow should have no detected vortex cores."""
        u, v = uniform_flow["u"], uniform_flow["v"]
        dx, dy = uniform_flow["dx"], uniform_flow["dy"]

        omega = vorticity(u, v, dx, dy)
        Q = q_criterion(u, v, dx, dy)

        cores = detect_vortex_cores(omega, Q)
        # Should have no vortex cores
        assert not np.any(cores)

    def test_preserves_shape(self, random_flow):
        """Output should have same shape as input."""
        u, v = random_flow["u"], random_flow["v"]
        dx, dy = random_flow["dx"], random_flow["dy"]

        omega = vorticity(u, v, dx, dy)
        Q = q_criterion(u, v, dx, dy)

        cores = detect_vortex_cores(omega, Q)
        assert cores.shape == u.shape
