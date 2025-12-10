"""Tests for cfd_viz.fields.velocity module."""

import numpy as np

from cfd_viz.fields import velocity


class TestMagnitude:
    """Tests for velocity.magnitude function."""

    def test_zero_velocity(self):
        """Magnitude of zero velocity should be zero."""
        u = np.zeros((10, 10))
        v = np.zeros((10, 10))
        mag = velocity.magnitude(u, v)
        np.testing.assert_array_equal(mag, 0)

    def test_unit_velocity_x(self):
        """Unit velocity in x direction should have magnitude 1."""
        u = np.ones((10, 10))
        v = np.zeros((10, 10))
        mag = velocity.magnitude(u, v)
        np.testing.assert_array_almost_equal(mag, 1.0)

    def test_unit_velocity_y(self):
        """Unit velocity in y direction should have magnitude 1."""
        u = np.zeros((10, 10))
        v = np.ones((10, 10))
        mag = velocity.magnitude(u, v)
        np.testing.assert_array_almost_equal(mag, 1.0)

    def test_diagonal_velocity(self):
        """Velocity (1,1) should have magnitude sqrt(2)."""
        u = np.ones((10, 10))
        v = np.ones((10, 10))
        mag = velocity.magnitude(u, v)
        np.testing.assert_array_almost_equal(mag, np.sqrt(2))

    def test_pythagorean_triple(self):
        """Test with 3-4-5 Pythagorean triple."""
        u = np.full((10, 10), 3.0)
        v = np.full((10, 10), 4.0)
        mag = velocity.magnitude(u, v)
        np.testing.assert_array_almost_equal(mag, 5.0)

    def test_preserves_shape(self):
        """Output should have same shape as input."""
        u = np.random.rand(20, 30)
        v = np.random.rand(20, 30)
        mag = velocity.magnitude(u, v)
        assert mag.shape == u.shape


class TestSpeed:
    """Tests for velocity.speed function (alias for magnitude)."""

    def test_speed_equals_magnitude(self):
        """Speed should be identical to magnitude."""
        u = np.random.rand(10, 10)
        v = np.random.rand(10, 10)
        np.testing.assert_array_equal(velocity.speed(u, v), velocity.magnitude(u, v))


class TestNormalize:
    """Tests for velocity.normalize function."""

    def test_unit_vector_magnitude(self):
        """Normalized vectors should have magnitude 1."""
        u = np.random.rand(10, 10) * 10
        v = np.random.rand(10, 10) * 10
        u_norm, v_norm = velocity.normalize(u, v)
        mag = np.sqrt(u_norm**2 + v_norm**2)
        # Exclude zero velocity points
        mask = velocity.magnitude(u, v) > 1e-10
        np.testing.assert_array_almost_equal(mag[mask], 1.0)

    def test_zero_velocity_stays_zero(self):
        """Zero velocity should remain zero after normalization."""
        u = np.zeros((10, 10))
        v = np.zeros((10, 10))
        u_norm, v_norm = velocity.normalize(u, v)
        np.testing.assert_array_equal(u_norm, 0)
        np.testing.assert_array_equal(v_norm, 0)

    def test_direction_preserved(self):
        """Normalization should preserve direction."""
        u = np.array([[3.0]])
        v = np.array([[4.0]])
        u_norm, v_norm = velocity.normalize(u, v)
        # Original ratio u/v = 3/4, should be preserved
        np.testing.assert_almost_equal(u_norm[0, 0] / v_norm[0, 0], 3 / 4)


class TestAngle:
    """Tests for velocity.angle function."""

    def test_positive_x_direction(self):
        """Velocity in +x direction should have angle 0."""
        u = np.array([[1.0]])
        v = np.array([[0.0]])
        ang = velocity.angle(u, v)
        np.testing.assert_almost_equal(ang[0, 0], 0)

    def test_positive_y_direction(self):
        """Velocity in +y direction should have angle pi/2."""
        u = np.array([[0.0]])
        v = np.array([[1.0]])
        ang = velocity.angle(u, v)
        np.testing.assert_almost_equal(ang[0, 0], np.pi / 2)

    def test_negative_x_direction(self):
        """Velocity in -x direction should have angle pi or -pi."""
        u = np.array([[-1.0]])
        v = np.array([[0.0]])
        ang = velocity.angle(u, v)
        np.testing.assert_almost_equal(np.abs(ang[0, 0]), np.pi)

    def test_diagonal_45_degrees(self):
        """Velocity (1,1) should have angle pi/4."""
        u = np.array([[1.0]])
        v = np.array([[1.0]])
        ang = velocity.angle(u, v)
        np.testing.assert_almost_equal(ang[0, 0], np.pi / 4)


class TestAngleDegrees:
    """Tests for velocity.angle_degrees function."""

    def test_positive_x_direction(self):
        """Velocity in +x direction should have angle 0 degrees."""
        u = np.array([[1.0]])
        v = np.array([[0.0]])
        ang = velocity.angle_degrees(u, v)
        np.testing.assert_almost_equal(ang[0, 0], 0)

    def test_positive_y_direction(self):
        """Velocity in +y direction should have angle 90 degrees."""
        u = np.array([[0.0]])
        v = np.array([[1.0]])
        ang = velocity.angle_degrees(u, v)
        np.testing.assert_almost_equal(ang[0, 0], 90)

    def test_diagonal_45_degrees(self):
        """Velocity (1,1) should have angle 45 degrees."""
        u = np.array([[1.0]])
        v = np.array([[1.0]])
        ang = velocity.angle_degrees(u, v)
        np.testing.assert_almost_equal(ang[0, 0], 45)


class TestComponentsFromMagnitudeAngle:
    """Tests for velocity.components_from_magnitude_angle function."""

    def test_round_trip(self):
        """Converting to mag/angle and back should preserve original."""
        u_orig = np.random.rand(10, 10) * 2 - 1
        v_orig = np.random.rand(10, 10) * 2 - 1
        mag = velocity.magnitude(u_orig, v_orig)
        ang = velocity.angle(u_orig, v_orig)
        u_new, v_new = velocity.components_from_magnitude_angle(mag, ang)
        np.testing.assert_array_almost_equal(u_new, u_orig)
        np.testing.assert_array_almost_equal(v_new, v_orig)

    def test_known_values(self):
        """Test with known magnitude and angle."""
        mag = np.array([[5.0]])
        theta = np.array([[np.arctan2(4, 3)]])  # 3-4-5 triangle
        u, v = velocity.components_from_magnitude_angle(mag, theta)
        np.testing.assert_almost_equal(u[0, 0], 3.0)
        np.testing.assert_almost_equal(v[0, 0], 4.0)


class TestFluctuations:
    """Tests for velocity.fluctuations function."""

    def test_uniform_flow_zero_fluctuations(self, uniform_flow):
        """Uniform flow should have zero fluctuations."""
        u, v = uniform_flow["u"], uniform_flow["v"]
        u_mean, v_mean, u_prime, v_prime = velocity.fluctuations(u, v)
        np.testing.assert_array_almost_equal(u_prime, 0)
        np.testing.assert_array_almost_equal(v_prime, 0)

    def test_mean_plus_fluctuation_equals_original(self, random_flow):
        """u_mean + u_prime should equal original u."""
        u, v = random_flow["u"], random_flow["v"]
        u_mean, v_mean, u_prime, v_prime = velocity.fluctuations(u, v)
        np.testing.assert_array_almost_equal(u_mean + u_prime, u)
        np.testing.assert_array_almost_equal(v_mean + v_prime, v)

    def test_fluctuation_mean_is_zero(self, random_flow):
        """Mean of fluctuations should be approximately zero."""
        u, v = random_flow["u"], random_flow["v"]
        _, _, u_prime, v_prime = velocity.fluctuations(u, v, axis=1)
        # Mean along same axis should be ~0
        np.testing.assert_array_almost_equal(np.mean(u_prime, axis=1), 0)
        np.testing.assert_array_almost_equal(np.mean(v_prime, axis=1), 0)


class TestTurbulentIntensity:
    """Tests for velocity.turbulent_intensity function."""

    def test_uniform_flow_zero_intensity(self, uniform_flow):
        """Uniform flow should have zero turbulent intensity."""
        u, v = uniform_flow["u"], uniform_flow["v"]
        ti = velocity.turbulent_intensity(u, v)
        np.testing.assert_array_almost_equal(ti, 0)

    def test_non_negative(self, random_flow):
        """Turbulent intensity should always be non-negative."""
        u, v = random_flow["u"], random_flow["v"]
        ti = velocity.turbulent_intensity(u, v)
        assert np.all(ti >= 0)
