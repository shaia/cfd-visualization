"""Tests for cfd_viz.fields.derived module."""

import numpy as np
import pytest

from cfd_viz.fields import derived


class TestKineticEnergy:
    """Tests for derived.kinetic_energy function."""

    def test_zero_velocity_zero_energy(self):
        """Zero velocity should have zero kinetic energy."""
        u = np.zeros((10, 10))
        v = np.zeros((10, 10))
        ke = derived.kinetic_energy(u, v)
        np.testing.assert_array_equal(ke, 0)

    def test_unit_velocity(self):
        """Unit velocity should have KE = 0.5 * rho."""
        u = np.ones((10, 10))
        v = np.zeros((10, 10))
        rho = 1.0
        ke = derived.kinetic_energy(u, v, rho)
        np.testing.assert_array_almost_equal(ke, 0.5 * rho)

    def test_density_scaling(self):
        """KE should scale linearly with density."""
        u = np.ones((10, 10))
        v = np.ones((10, 10))
        ke1 = derived.kinetic_energy(u, v, rho=1.0)
        ke2 = derived.kinetic_energy(u, v, rho=2.0)
        np.testing.assert_array_almost_equal(ke2, 2 * ke1)

    def test_always_non_negative(self, random_flow):
        """Kinetic energy should always be non-negative."""
        u, v = random_flow["u"], random_flow["v"]
        ke = derived.kinetic_energy(u, v)
        assert np.all(ke >= 0)


class TestTotalKineticEnergy:
    """Tests for derived.total_kinetic_energy function."""

    def test_zero_velocity_zero_total(self, uniform_grid):
        """Zero velocity should have zero total KE."""
        u = np.zeros((uniform_grid["ny"], uniform_grid["nx"]))
        v = np.zeros_like(u)
        dx, dy = uniform_grid["dx"], uniform_grid["dy"]
        tke = derived.total_kinetic_energy(u, v, dx, dy)
        assert tke == 0

    def test_returns_scalar(self, random_flow):
        """Should return a scalar value."""
        u, v = random_flow["u"], random_flow["v"]
        dx, dy = random_flow["dx"], random_flow["dy"]
        tke = derived.total_kinetic_energy(u, v, dx, dy)
        assert np.isscalar(tke)

    def test_non_negative(self, random_flow):
        """Total KE should always be non-negative."""
        u, v = random_flow["u"], random_flow["v"]
        dx, dy = random_flow["dx"], random_flow["dy"]
        tke = derived.total_kinetic_energy(u, v, dx, dy)
        assert tke >= 0


class TestDynamicPressure:
    """Tests for derived.dynamic_pressure function."""

    def test_equals_kinetic_energy(self):
        """Dynamic pressure should equal kinetic energy density."""
        u = np.random.rand(10, 10)
        v = np.random.rand(10, 10)
        rho = 1.225
        q = derived.dynamic_pressure(u, v, rho)
        ke = derived.kinetic_energy(u, v, rho)
        np.testing.assert_array_equal(q, ke)


class TestPressureCoefficient:
    """Tests for derived.pressure_coefficient function."""

    def test_freestream_conditions(self):
        """At freestream conditions, Cp should be 0."""
        p = np.full((10, 10), 101325.0)  # p = p_inf everywhere
        p_inf = 101325.0
        rho_inf = 1.225
        U_inf = 10.0
        Cp = derived.pressure_coefficient(p, p_inf, rho_inf, U_inf)
        np.testing.assert_array_almost_equal(Cp, 0)

    def test_stagnation_point(self):
        """At stagnation point (p = p_inf + q_inf), Cp should be 1."""
        rho_inf = 1.225
        U_inf = 10.0
        p_inf = 101325.0
        q_inf = 0.5 * rho_inf * U_inf**2
        p = np.full((10, 10), p_inf + q_inf)
        Cp = derived.pressure_coefficient(p, p_inf, rho_inf, U_inf)
        np.testing.assert_array_almost_equal(Cp, 1.0)

    def test_handles_zero_velocity(self):
        """Should handle zero freestream velocity gracefully."""
        p = np.ones((10, 10))
        Cp = derived.pressure_coefficient(p, 0, 1.0, 0)
        np.testing.assert_array_equal(Cp, 0)


class TestTotalPressure:
    """Tests for derived.total_pressure function."""

    def test_stagnation_equals_static_at_rest(self):
        """When velocity is zero, total pressure equals static pressure."""
        p = np.random.rand(10, 10) * 1000
        u = np.zeros((10, 10))
        v = np.zeros((10, 10))
        p0 = derived.total_pressure(p, u, v)
        np.testing.assert_array_equal(p0, p)

    def test_total_greater_than_static(self, random_flow):
        """Total pressure should be >= static pressure."""
        u, v, p = random_flow["u"], random_flow["v"], random_flow["p"]
        p0 = derived.total_pressure(p, u, v)
        assert np.all(p0 >= p - 1e-10)


class TestMachNumber:
    """Tests for derived.mach_number function."""

    def test_subsonic(self):
        """Mach < 1 for subsonic flow."""
        a = 343.0  # Speed of sound in air
        u = np.full((10, 10), 100.0)  # 100 m/s
        v = np.zeros((10, 10))
        M = derived.mach_number(u, v, a)
        assert np.all(M < 1)

    def test_supersonic(self):
        """Mach > 1 for supersonic flow."""
        a = 343.0
        u = np.full((10, 10), 400.0)  # 400 m/s
        v = np.zeros((10, 10))
        M = derived.mach_number(u, v, a)
        assert np.all(M > 1)

    def test_sonic(self):
        """Mach = 1 when velocity equals speed of sound."""
        a = 343.0
        u = np.full((10, 10), a)
        v = np.zeros((10, 10))
        M = derived.mach_number(u, v, a)
        np.testing.assert_array_almost_equal(M, 1.0)


class TestReynoldsNumberLocal:
    """Tests for derived.reynolds_number_local function."""

    def test_increases_with_x(self, uniform_flow):
        """Local Reynolds number should increase with x."""
        u, v = uniform_flow["u"], uniform_flow["v"]
        x = uniform_flow["x"]
        nu = 1e-5
        Re_x = derived.reynolds_number_local(u, v, x, nu)
        # Compare columns (constant y)
        assert np.all(Re_x[:, -1] >= Re_x[:, 0])

    def test_zero_at_origin(self, uniform_flow):
        """Reynolds number should be zero at x=0."""
        u, v = uniform_flow["u"], uniform_flow["v"]
        x = uniform_flow["x"]
        nu = 1e-5
        Re_x = derived.reynolds_number_local(u, v, x, nu)
        np.testing.assert_array_almost_equal(Re_x[:, 0], 0)


class TestStreamFunction:
    """Tests for derived.stream_function function."""

    def test_uniform_flow(self, uniform_flow):
        """Uniform flow psi should increase with y."""
        u, v = uniform_flow["u"], uniform_flow["v"]
        dx, dy = uniform_flow["dx"], uniform_flow["dy"]
        psi = derived.stream_function(u, v, dx, dy)
        # For uniform u-flow, psi should increase with y
        # Check that each row is greater than the previous
        for j in range(1, psi.shape[0]):
            assert np.mean(psi[j, :]) >= np.mean(psi[j - 1, :]) - 1e-10

    def test_preserves_shape(self, random_flow):
        """Output should have same shape as input."""
        u, v = random_flow["u"], random_flow["v"]
        dx, dy = random_flow["dx"], random_flow["dy"]
        psi = derived.stream_function(u, v, dx, dy)
        assert psi.shape == u.shape


class TestDissipationRate:
    """Tests for derived.dissipation_rate function."""

    def test_uniform_flow_zero_dissipation(self, uniform_flow):
        """Uniform flow should have zero dissipation."""
        u, v = uniform_flow["u"], uniform_flow["v"]
        dx, dy = uniform_flow["dx"], uniform_flow["dy"]
        nu = 1e-5
        eps = derived.dissipation_rate(u, v, dx, dy, nu)
        np.testing.assert_array_almost_equal(eps[2:-2, 2:-2], 0)

    def test_always_non_negative(self, random_flow):
        """Dissipation rate should always be non-negative."""
        u, v = random_flow["u"], random_flow["v"]
        dx, dy = random_flow["dx"], random_flow["dy"]
        nu = 1e-5
        eps = derived.dissipation_rate(u, v, dx, dy, nu)
        assert np.all(eps >= -1e-15)  # Allow tiny numerical errors

    def test_scales_with_viscosity(self, shear_flow):
        """Dissipation should scale linearly with viscosity."""
        u, v = shear_flow["u"], shear_flow["v"]
        dx, dy = shear_flow["dx"], shear_flow["dy"]
        eps1 = derived.dissipation_rate(u, v, dx, dy, nu=1e-5)
        eps2 = derived.dissipation_rate(u, v, dx, dy, nu=2e-5)
        np.testing.assert_array_almost_equal(eps2, 2 * eps1)


class TestFlowStatistics:
    """Tests for derived.FlowStatistics dataclass."""

    def test_to_dict(self, random_flow):
        """to_dict should return all fields."""
        u, v, p = random_flow["u"], random_flow["v"], random_flow["p"]
        dx, dy = random_flow["dx"], random_flow["dy"]
        stats = derived.calculate_flow_statistics(u, v, p, dx, dy)
        d = stats.to_dict()
        assert "max_velocity" in d
        assert "mean_velocity" in d
        assert "total_kinetic_energy" in d
        assert "max_vorticity" in d


class TestCalculateFlowStatistics:
    """Tests for derived.calculate_flow_statistics function."""

    def test_returns_dataclass(self, random_flow):
        """Should return a FlowStatistics dataclass."""
        u, v, p = random_flow["u"], random_flow["v"], random_flow["p"]
        dx, dy = random_flow["dx"], random_flow["dy"]
        stats = derived.calculate_flow_statistics(u, v, p, dx, dy)
        assert isinstance(stats, derived.FlowStatistics)

    def test_uniform_flow_statistics(self, uniform_flow):
        """Uniform flow should have predictable statistics."""
        u, v, p = uniform_flow["u"], uniform_flow["v"], uniform_flow["p"]
        dx, dy = uniform_flow["dx"], uniform_flow["dy"]
        stats = derived.calculate_flow_statistics(u, v, p, dx, dy)
        # Uniform u=1, v=0 flow
        assert stats.max_velocity == pytest.approx(1.0)
        assert stats.mean_velocity == pytest.approx(1.0)
        assert stats.min_velocity == pytest.approx(1.0)
        assert stats.velocity_std == pytest.approx(0.0)

    def test_handles_none_pressure(self, random_flow):
        """Should handle None pressure field."""
        u, v = random_flow["u"], random_flow["v"]
        dx, dy = random_flow["dx"], random_flow["dy"]
        stats = derived.calculate_flow_statistics(u, v, None, dx, dy)
        assert stats.max_pressure == 0
        assert stats.mean_pressure == 0

    def test_velocity_uniformity_uniform_flow(self, uniform_flow):
        """Uniform flow should have zero velocity uniformity (CoV)."""
        u, v, p = uniform_flow["u"], uniform_flow["v"], uniform_flow["p"]
        dx, dy = uniform_flow["dx"], uniform_flow["dy"]
        stats = derived.calculate_flow_statistics(u, v, p, dx, dy)
        assert stats.velocity_uniformity == pytest.approx(0.0)

    def test_max_greater_than_mean(self, random_flow):
        """Max velocity should be >= mean velocity."""
        u, v, p = random_flow["u"], random_flow["v"], random_flow["p"]
        dx, dy = random_flow["dx"], random_flow["dy"]
        stats = derived.calculate_flow_statistics(u, v, p, dx, dy)
        assert stats.max_velocity >= stats.mean_velocity

    def test_min_less_than_mean(self, random_flow):
        """Min velocity should be <= mean velocity."""
        u, v, p = random_flow["u"], random_flow["v"], random_flow["p"]
        dx, dy = random_flow["dx"], random_flow["dy"]
        stats = derived.calculate_flow_statistics(u, v, p, dx, dy)
        assert stats.min_velocity <= stats.mean_velocity
