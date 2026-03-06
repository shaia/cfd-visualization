"""Tests for NumPy backend implementations."""

import numpy as np
import pytest

from cfd_viz.backends._numpy import NumPyStatsBackend, NumPySystemInfoBackend


class TestNumPyStatsBackend:
    """Tests for NumPyStatsBackend."""

    def setup_method(self):
        self.backend = NumPyStatsBackend()

    def test_calculate_field_stats_list(self):
        result = self.backend.calculate_field_stats([1.0, 2.0, 3.0, 4.0, 5.0])
        assert result["min"] == pytest.approx(1.0)
        assert result["max"] == pytest.approx(5.0)
        assert result["avg"] == pytest.approx(3.0)
        assert result["sum"] == pytest.approx(15.0)

    def test_calculate_field_stats_array(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = self.backend.calculate_field_stats(data)
        assert result["min"] == pytest.approx(1.0)
        assert result["max"] == pytest.approx(4.0)

    def test_compute_flow_statistics_without_pressure(self):
        u = np.ones((10, 10)) * 3.0
        v = np.ones((10, 10)) * 4.0
        result = self.backend.compute_flow_statistics(u, v, 10, 10, None)
        assert "u" in result
        assert "v" in result
        assert "velocity_magnitude" in result
        assert "p" not in result
        assert result["velocity_magnitude"]["avg"] == pytest.approx(5.0)

    def test_compute_flow_statistics_with_pressure(self):
        u = np.ones((10, 10))
        v = np.ones((10, 10))
        p = np.ones((10, 10)) * 100.0
        result = self.backend.compute_flow_statistics(u, v, 10, 10, p)
        assert "p" in result
        assert result["p"]["avg"] == pytest.approx(100.0)

    def test_compute_velocity_magnitude(self):
        u = np.ones((10, 10)) * 3.0
        v = np.ones((10, 10)) * 4.0
        result = self.backend.compute_velocity_magnitude(u, v, 10, 10)
        assert result.shape == (10, 10)
        assert np.allclose(result, 5.0)


class TestNumPySystemInfoBackend:
    """Tests for NumPySystemInfoBackend."""

    def test_returns_not_available(self):
        backend = NumPySystemInfoBackend()
        info = backend.get_system_info()
        assert info["cfd_python_available"] is False
        assert info["cfd_python_version"] is None
        assert info["backends"] == []
        assert info["gpu_available"] is False
