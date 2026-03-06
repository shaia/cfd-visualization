"""Tests for cfd-python backend implementations."""

import numpy as np
import pytest

from cfd_viz import cfd_python_integration
from cfd_viz.backends._cfd_python import (
    CfdPythonStatsBackend,
    CfdPythonSystemInfoBackend,
)

pytestmark = pytest.mark.skipif(
    not cfd_python_integration.has_cfd_python(), reason="cfd-python not installed"
)


class TestCfdPythonStatsBackend:
    """Tests for CfdPythonStatsBackend."""

    def setup_method(self):
        self.backend = CfdPythonStatsBackend()

    def test_calculate_field_stats(self):
        result = self.backend.calculate_field_stats([1.0, 2.0, 3.0, 4.0, 5.0])
        assert result["min"] == pytest.approx(1.0)
        assert result["max"] == pytest.approx(5.0)
        assert result["avg"] == pytest.approx(3.0)
        assert result["sum"] == pytest.approx(15.0)

    def test_compute_flow_statistics_without_pressure(self):
        u = np.ones((10, 10)) * 3.0
        v = np.ones((10, 10)) * 4.0
        result = self.backend.compute_flow_statistics(u, v, 10, 10, None)
        assert "u" in result
        assert "v" in result
        assert "velocity_magnitude" in result
        assert "p" not in result

    def test_compute_flow_statistics_with_pressure(self):
        u = np.ones((10, 10))
        v = np.ones((10, 10))
        p = np.ones((10, 10)) * 100.0
        result = self.backend.compute_flow_statistics(u, v, 10, 10, p)
        assert "p" in result

    def test_compute_velocity_magnitude(self):
        u = np.ones((10, 10)) * 3.0
        v = np.ones((10, 10)) * 4.0
        result = self.backend.compute_velocity_magnitude(u, v, 10, 10)
        assert result.shape == (10, 10)
        assert np.allclose(result, 5.0)

    def test_matches_numpy_backend(self):
        """Results should match between backends."""
        from cfd_viz.backends._numpy import NumPyStatsBackend

        numpy_backend = NumPyStatsBackend()
        data = list(range(100))
        cfd_result = self.backend.calculate_field_stats(data)
        numpy_result = numpy_backend.calculate_field_stats(data)
        assert cfd_result["avg"] == pytest.approx(numpy_result["avg"])


class TestCfdPythonSystemInfoBackend:
    """Tests for CfdPythonSystemInfoBackend."""

    def test_returns_available(self):
        backend = CfdPythonSystemInfoBackend()
        info = backend.get_system_info()
        assert info["cfd_python_available"] is True
        assert info["cfd_python_version"] is not None
        assert isinstance(info["backends"], list)
        assert len(info["backends"]) > 0
