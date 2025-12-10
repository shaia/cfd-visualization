"""Tests for cfd_viz.fields.gradients module."""

import numpy as np

from cfd_viz.fields import gradients


class TestVelocityGradients:
    """Tests for gradients.velocity_gradients function."""

    def test_returns_dataclass(self, random_flow):
        """Should return a VelocityGradients dataclass."""
        u, v = random_flow["u"], random_flow["v"]
        dx, dy = random_flow["dx"], random_flow["dy"]
        grads = gradients.velocity_gradients(u, v, dx, dy)
        assert isinstance(grads, gradients.VelocityGradients)

    def test_has_all_components(self, random_flow):
        """Dataclass should have all gradient components."""
        u, v = random_flow["u"], random_flow["v"]
        dx, dy = random_flow["dx"], random_flow["dy"]
        grads = gradients.velocity_gradients(u, v, dx, dy)
        assert hasattr(grads, "du_dx")
        assert hasattr(grads, "du_dy")
        assert hasattr(grads, "dv_dx")
        assert hasattr(grads, "dv_dy")

    def test_uniform_flow_zero_gradients(self, uniform_flow):
        """Uniform flow should have zero gradients."""
        u, v = uniform_flow["u"], uniform_flow["v"]
        dx, dy = uniform_flow["dx"], uniform_flow["dy"]
        grads = gradients.velocity_gradients(u, v, dx, dy)
        np.testing.assert_array_almost_equal(grads.du_dx[2:-2, 2:-2], 0)
        np.testing.assert_array_almost_equal(grads.du_dy[2:-2, 2:-2], 0)
        np.testing.assert_array_almost_equal(grads.dv_dx[2:-2, 2:-2], 0)
        np.testing.assert_array_almost_equal(grads.dv_dy[2:-2, 2:-2], 0)

    def test_shear_flow_du_dy(self, shear_flow):
        """For u=y flow, du/dy should equal 1."""
        u, v = shear_flow["u"], shear_flow["v"]
        dx, dy = shear_flow["dx"], shear_flow["dy"]
        grads = gradients.velocity_gradients(u, v, dx, dy)
        # du/dy = d(y)/dy = 1
        np.testing.assert_array_almost_equal(grads.du_dy[2:-2, 2:-2], 1.0, decimal=5)

    def test_divergence_property(self, random_flow):
        """Divergence property should equal du/dx + dv/dy."""
        u, v = random_flow["u"], random_flow["v"]
        dx, dy = random_flow["dx"], random_flow["dy"]
        grads = gradients.velocity_gradients(u, v, dx, dy)
        expected = grads.du_dx + grads.dv_dy
        np.testing.assert_array_almost_equal(grads.divergence, expected)

    def test_vorticity_property(self, random_flow):
        """Vorticity property should equal dv/dx - du/dy."""
        u, v = random_flow["u"], random_flow["v"]
        dx, dy = random_flow["dx"], random_flow["dy"]
        grads = gradients.velocity_gradients(u, v, dx, dy)
        expected = grads.dv_dx - grads.du_dy
        np.testing.assert_array_almost_equal(grads.vorticity, expected)


class TestStrainRateTensor:
    """Tests for gradients.strain_rate_tensor function."""

    def test_returns_dataclass(self, random_flow):
        """Should return a StrainRateTensor dataclass."""
        u, v = random_flow["u"], random_flow["v"]
        dx, dy = random_flow["dx"], random_flow["dy"]
        S = gradients.strain_rate_tensor(u, v, dx, dy)
        assert isinstance(S, gradients.StrainRateTensor)

    def test_has_all_components(self, random_flow):
        """Dataclass should have all strain rate components."""
        u, v = random_flow["u"], random_flow["v"]
        dx, dy = random_flow["dx"], random_flow["dy"]
        S = gradients.strain_rate_tensor(u, v, dx, dy)
        assert hasattr(S, "S11")
        assert hasattr(S, "S12")
        assert hasattr(S, "S22")

    def test_uniform_flow_zero_strain(self, uniform_flow):
        """Uniform flow should have zero strain rate."""
        u, v = uniform_flow["u"], uniform_flow["v"]
        dx, dy = uniform_flow["dx"], uniform_flow["dy"]
        S = gradients.strain_rate_tensor(u, v, dx, dy)
        np.testing.assert_array_almost_equal(S.S11[2:-2, 2:-2], 0)
        np.testing.assert_array_almost_equal(S.S12[2:-2, 2:-2], 0)
        np.testing.assert_array_almost_equal(S.S22[2:-2, 2:-2], 0)

    def test_shear_flow_S12(self, shear_flow):
        """For shear flow u=y, S12 should equal 0.5."""
        u, v = shear_flow["u"], shear_flow["v"]
        dx, dy = shear_flow["dx"], shear_flow["dy"]
        S = gradients.strain_rate_tensor(u, v, dx, dy)
        # S12 = 0.5 * (du/dy + dv/dx) = 0.5 * (1 + 0) = 0.5
        np.testing.assert_array_almost_equal(S.S12[2:-2, 2:-2], 0.5, decimal=5)

    def test_magnitude_property(self, random_flow):
        """Magnitude property should compute Frobenius norm."""
        u, v = random_flow["u"], random_flow["v"]
        dx, dy = random_flow["dx"], random_flow["dy"]
        S = gradients.strain_rate_tensor(u, v, dx, dy)
        expected = np.sqrt(2 * (S.S11**2 + 2 * S.S12**2 + S.S22**2))
        np.testing.assert_array_almost_equal(S.magnitude, expected)

    def test_principal_strains_order(self, random_flow):
        """Max principal strain should be >= min principal strain."""
        u, v = random_flow["u"], random_flow["v"]
        dx, dy = random_flow["dx"], random_flow["dy"]
        S = gradients.strain_rate_tensor(u, v, dx, dy)
        assert np.all(S.principal_strain_max >= S.principal_strain_min - 1e-10)


class TestDivergence:
    """Tests for gradients.divergence function."""

    def test_uniform_flow_zero_divergence(self, uniform_flow):
        """Uniform flow should have zero divergence."""
        u, v = uniform_flow["u"], uniform_flow["v"]
        dx, dy = uniform_flow["dx"], uniform_flow["dy"]
        div = gradients.divergence(u, v, dx, dy)
        np.testing.assert_array_almost_equal(div[2:-2, 2:-2], 0)

    def test_stagnation_flow_zero_divergence(self, stagnation_flow):
        """Incompressible stagnation flow should have zero divergence."""
        u, v = stagnation_flow["u"], stagnation_flow["v"]
        dx, dy = stagnation_flow["dx"], stagnation_flow["dy"]
        div = gradients.divergence(u, v, dx, dy)
        # du/dx + dv/dy = k + (-k) = 0
        np.testing.assert_array_almost_equal(div[2:-2, 2:-2], 0, decimal=5)

    def test_solid_body_rotation_zero_divergence(self, vortex_flow):
        """Solid body rotation should have zero divergence."""
        u, v = vortex_flow["u"], vortex_flow["v"]
        dx, dy = vortex_flow["dx"], vortex_flow["dy"]
        div = gradients.divergence(u, v, dx, dy)
        np.testing.assert_array_almost_equal(div[2:-2, 2:-2], 0, decimal=5)


class TestShearStrainRate:
    """Tests for gradients.shear_strain_rate function."""

    def test_uniform_flow_zero_shear(self, uniform_flow):
        """Uniform flow should have zero shear strain rate."""
        u, v = uniform_flow["u"], uniform_flow["v"]
        dx, dy = uniform_flow["dx"], uniform_flow["dy"]
        shear = gradients.shear_strain_rate(u, v, dx, dy)
        np.testing.assert_array_almost_equal(shear[2:-2, 2:-2], 0)

    def test_shear_flow(self, shear_flow):
        """Simple shear flow u=y should have shear rate = 0.5."""
        u, v = shear_flow["u"], shear_flow["v"]
        dx, dy = shear_flow["dx"], shear_flow["dy"]
        shear = gradients.shear_strain_rate(u, v, dx, dy)
        np.testing.assert_array_almost_equal(shear[2:-2, 2:-2], 0.5, decimal=5)


class TestNormalStrainRates:
    """Tests for gradients.normal_strain_rates function."""

    def test_returns_tuple(self, random_flow):
        """Should return a tuple of two arrays."""
        u, v = random_flow["u"], random_flow["v"]
        dx, dy = random_flow["dx"], random_flow["dy"]
        result = gradients.normal_strain_rates(u, v, dx, dy)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_stagnation_flow(self, stagnation_flow):
        """Stagnation flow u=kx, v=-ky should have du/dx=k, dv/dy=-k."""
        u, v = stagnation_flow["u"], stagnation_flow["v"]
        dx, dy = stagnation_flow["dx"], stagnation_flow["dy"]
        k = stagnation_flow["k"]
        du_dx, dv_dy = gradients.normal_strain_rates(u, v, dx, dy)
        np.testing.assert_array_almost_equal(du_dx[5:-5, 5:-5], k, decimal=3)
        np.testing.assert_array_almost_equal(dv_dy[5:-5, 5:-5], -k, decimal=3)


class TestStrainRateMagnitude:
    """Tests for gradients.strain_rate_magnitude function."""

    def test_uniform_flow_zero_magnitude(self, uniform_flow):
        """Uniform flow should have zero strain rate magnitude."""
        u, v = uniform_flow["u"], uniform_flow["v"]
        dx, dy = uniform_flow["dx"], uniform_flow["dy"]
        mag = gradients.strain_rate_magnitude(u, v, dx, dy)
        np.testing.assert_array_almost_equal(mag[2:-2, 2:-2], 0)

    def test_always_non_negative(self, random_flow):
        """Strain rate magnitude should always be non-negative."""
        u, v = random_flow["u"], random_flow["v"]
        dx, dy = random_flow["dx"], random_flow["dy"]
        mag = gradients.strain_rate_magnitude(u, v, dx, dy)
        assert np.all(mag >= 0)


class TestRotationRate:
    """Tests for gradients.rotation_rate function."""

    def test_uniform_flow_zero_rotation(self, uniform_flow):
        """Uniform flow should have zero rotation rate."""
        u, v = uniform_flow["u"], uniform_flow["v"]
        dx, dy = uniform_flow["dx"], uniform_flow["dy"]
        rot = gradients.rotation_rate(u, v, dx, dy)
        np.testing.assert_array_almost_equal(rot[2:-2, 2:-2], 0)

    def test_solid_body_rotation(self, vortex_flow):
        """Solid body rotation should have constant rotation rate."""
        u, v = vortex_flow["u"], vortex_flow["v"]
        dx, dy = vortex_flow["dx"], vortex_flow["dy"]
        angular_vel = vortex_flow["omega"]
        rot = gradients.rotation_rate(u, v, dx, dy)
        # Rotation rate Omega_12 = 0.5 * (du/dy - dv/dx) = -0.5 * vorticity
        # For solid body: Omega_12 = -omega (negative of angular velocity)
        np.testing.assert_array_almost_equal(rot[5:-5, 5:-5], -angular_vel, decimal=3)


class TestWallShearStress:
    """Tests for gradients.wall_shear_stress function."""

    def test_channel_flow_bottom_wall(self, channel_flow):
        """Channel flow should have non-zero wall shear at bottom."""
        u, v = channel_flow["u"], channel_flow["v"]
        dy = channel_flow["dy"]
        mu = 1.0  # Dynamic viscosity
        tau_w = gradients.wall_shear_stress(u, v, dy, mu, wall_index=0)
        # Wall shear should be positive (velocity increases from wall)
        assert np.all(tau_w > 0)

    def test_uniform_flow_zero_wall_shear(self, uniform_flow):
        """Uniform flow should have zero wall shear stress."""
        u, v = uniform_flow["u"], uniform_flow["v"]
        dy = uniform_flow["dy"]
        mu = 1.0
        tau_w = gradients.wall_shear_stress(u, v, dy, mu, wall_index=0)
        np.testing.assert_array_almost_equal(tau_w, 0)

    def test_returns_1d_array(self, random_flow):
        """Should return a 1D array (along wall)."""
        u, v = random_flow["u"], random_flow["v"]
        dy = random_flow["dy"]
        mu = 1.0
        tau_w = gradients.wall_shear_stress(u, v, dy, mu, wall_index=0)
        assert tau_w.ndim == 1
        assert tau_w.shape[0] == u.shape[1]
