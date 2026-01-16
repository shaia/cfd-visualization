"""Tests for cfd_viz.stats module."""

import numpy as np
import pytest

from cfd_viz import cfd_python_integration
from cfd_viz.common import VTKData
from cfd_viz.convert import from_cfd_python
from cfd_viz.stats import (
    calculate_field_stats,
    compute_flow_statistics,
    compute_velocity_magnitude,
)


class TestCalculateFieldStats:
    """Tests for calculate_field_stats function."""

    def test_returns_dict_with_expected_keys(self):
        """Should return dict with min, max, avg, sum."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = calculate_field_stats(data)

        assert "min" in result
        assert "max" in result
        assert "avg" in result
        assert "sum" in result

    def test_calculates_correct_values_for_list(self):
        """Should calculate correct statistics for list input."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = calculate_field_stats(data)

        assert result["min"] == pytest.approx(1.0)
        assert result["max"] == pytest.approx(5.0)
        assert result["avg"] == pytest.approx(3.0)
        assert result["sum"] == pytest.approx(15.0)

    def test_calculates_correct_values_for_array(self):
        """Should calculate correct statistics for numpy array."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = calculate_field_stats(data)

        assert result["min"] == pytest.approx(1.0)
        assert result["max"] == pytest.approx(4.0)
        assert result["avg"] == pytest.approx(2.5)
        assert result["sum"] == pytest.approx(10.0)

    def test_numpy_fallback_when_disabled(self):
        """Should use NumPy when use_cfd_python=False."""
        data = [1.0, 2.0, 3.0]
        result = calculate_field_stats(data, use_cfd_python=False)

        assert result["min"] == pytest.approx(1.0)
        assert result["max"] == pytest.approx(3.0)
        assert result["avg"] == pytest.approx(2.0)
        assert result["sum"] == pytest.approx(6.0)

    def test_handles_negative_values(self):
        """Should handle negative values correctly."""
        data = [-5.0, -2.0, 0.0, 3.0, 7.0]
        result = calculate_field_stats(data)

        assert result["min"] == pytest.approx(-5.0)
        assert result["max"] == pytest.approx(7.0)
        assert result["avg"] == pytest.approx(0.6)
        assert result["sum"] == pytest.approx(3.0)

    def test_single_value(self):
        """Should handle single value correctly."""
        data = [42.0]
        result = calculate_field_stats(data)

        assert result["min"] == pytest.approx(42.0)
        assert result["max"] == pytest.approx(42.0)
        assert result["avg"] == pytest.approx(42.0)
        assert result["sum"] == pytest.approx(42.0)

    @pytest.mark.skipif(
        not cfd_python_integration.has_cfd_python(), reason="cfd-python not installed"
    )
    def test_cfd_python_matches_numpy(self):
        """Results should match between cfd-python and NumPy implementations."""
        data = list(range(100))

        cfd_result = calculate_field_stats(data, use_cfd_python=True)
        numpy_result = calculate_field_stats(data, use_cfd_python=False)

        assert cfd_result["min"] == pytest.approx(numpy_result["min"])
        assert cfd_result["max"] == pytest.approx(numpy_result["max"])
        assert cfd_result["avg"] == pytest.approx(numpy_result["avg"])
        assert cfd_result["sum"] == pytest.approx(numpy_result["sum"])


class TestComputeFlowStatistics:
    """Tests for compute_flow_statistics function."""

    def test_returns_expected_keys(self):
        """Should return dict with u, v, velocity_magnitude keys."""
        data = from_cfd_python([1.0] * 100, [2.0] * 100, nx=10, ny=10)
        result = compute_flow_statistics(data)

        assert "u" in result
        assert "v" in result
        assert "velocity_magnitude" in result

    def test_includes_pressure_when_present(self):
        """Should include p stats when pressure is available."""
        data = from_cfd_python(
            [1.0] * 100, [2.0] * 100, nx=10, ny=10, p=[101325.0] * 100
        )
        result = compute_flow_statistics(data)

        assert "p" in result
        assert result["p"]["avg"] == pytest.approx(101325.0)

    def test_excludes_pressure_when_absent(self):
        """Should not include p stats when pressure is not available."""
        data = from_cfd_python([1.0] * 100, [2.0] * 100, nx=10, ny=10)
        result = compute_flow_statistics(data)

        assert "p" not in result

    def test_velocity_magnitude_calculated(self):
        """Should calculate velocity magnitude correctly."""
        # u=3, v=4 -> magnitude=5
        data = from_cfd_python([3.0] * 100, [4.0] * 100, nx=10, ny=10)
        result = compute_flow_statistics(data)

        assert result["velocity_magnitude"]["avg"] == pytest.approx(5.0)

    def test_numpy_fallback_when_disabled(self):
        """Should use NumPy when use_cfd_python=False."""
        data = from_cfd_python([1.0] * 100, [2.0] * 100, nx=10, ny=10)
        result = compute_flow_statistics(data, use_cfd_python=False)

        assert result["u"]["avg"] == pytest.approx(1.0)
        assert result["v"]["avg"] == pytest.approx(2.0)

    def test_raises_for_missing_u(self):
        """Should raise ValueError when u field is missing."""
        data = VTKData(
            x=np.linspace(0, 1, 10),
            y=np.linspace(0, 1, 10),
            X=np.zeros((10, 10)),
            Y=np.zeros((10, 10)),
            fields={"v": np.ones((10, 10))},
            nx=10,
            ny=10,
            dx=0.1,
            dy=0.1,
        )

        with pytest.raises(ValueError, match="must have both u and v"):
            compute_flow_statistics(data)

    def test_raises_for_missing_v(self):
        """Should raise ValueError when v field is missing."""
        data = VTKData(
            x=np.linspace(0, 1, 10),
            y=np.linspace(0, 1, 10),
            X=np.zeros((10, 10)),
            Y=np.zeros((10, 10)),
            fields={"u": np.ones((10, 10))},
            nx=10,
            ny=10,
            dx=0.1,
            dy=0.1,
        )

        with pytest.raises(ValueError, match="must have both u and v"):
            compute_flow_statistics(data)

    @pytest.mark.skipif(
        not cfd_python_integration.has_cfd_python(), reason="cfd-python not installed"
    )
    def test_cfd_python_matches_numpy(self):
        """Results should match between cfd-python and NumPy implementations."""
        u = list(range(100))
        v = list(range(100, 200))
        data = from_cfd_python(u, v, nx=10, ny=10)

        cfd_result = compute_flow_statistics(data, use_cfd_python=True)
        numpy_result = compute_flow_statistics(data, use_cfd_python=False)

        # Compare u stats
        assert cfd_result["u"]["avg"] == pytest.approx(numpy_result["u"]["avg"])
        assert cfd_result["v"]["avg"] == pytest.approx(numpy_result["v"]["avg"])
        # Velocity magnitude
        assert cfd_result["velocity_magnitude"]["avg"] == pytest.approx(
            numpy_result["velocity_magnitude"]["avg"], rel=1e-5
        )


class TestComputeVelocityMagnitude:
    """Tests for compute_velocity_magnitude function."""

    def test_returns_2d_array(self):
        """Should return 2D array with correct shape."""
        data = from_cfd_python([1.0] * 100, [2.0] * 100, nx=10, ny=10)
        result = compute_velocity_magnitude(data)

        assert isinstance(result, np.ndarray)
        assert result.shape == (10, 10)

    def test_calculates_correct_magnitude(self):
        """Should calculate sqrt(u^2 + v^2)."""
        # u=3, v=4 -> magnitude=5
        data = from_cfd_python([3.0] * 100, [4.0] * 100, nx=10, ny=10)
        result = compute_velocity_magnitude(data)

        assert np.allclose(result, 5.0)

    def test_varying_values(self):
        """Should handle varying field values."""
        u = [float(i) for i in range(100)]
        v = [float(i) for i in range(100)]
        data = from_cfd_python(u, v, nx=10, ny=10)
        result = compute_velocity_magnitude(data)

        # Check a few values: magnitude = sqrt(i^2 + i^2) = i * sqrt(2)
        assert result[0, 0] == pytest.approx(0.0)
        assert result[0, 1] == pytest.approx(np.sqrt(2))
        assert result[9, 9] == pytest.approx(99 * np.sqrt(2))

    def test_numpy_fallback_when_disabled(self):
        """Should use NumPy when use_cfd_python=False."""
        data = from_cfd_python([3.0] * 100, [4.0] * 100, nx=10, ny=10)
        result = compute_velocity_magnitude(data, use_cfd_python=False)

        assert np.allclose(result, 5.0)

    def test_raises_for_missing_u(self):
        """Should raise ValueError when u field is missing."""
        data = VTKData(
            x=np.linspace(0, 1, 10),
            y=np.linspace(0, 1, 10),
            X=np.zeros((10, 10)),
            Y=np.zeros((10, 10)),
            fields={"v": np.ones((10, 10))},
            nx=10,
            ny=10,
            dx=0.1,
            dy=0.1,
        )

        with pytest.raises(ValueError, match="must have both u and v"):
            compute_velocity_magnitude(data)

    def test_raises_for_missing_v(self):
        """Should raise ValueError when v field is missing."""
        data = VTKData(
            x=np.linspace(0, 1, 10),
            y=np.linspace(0, 1, 10),
            X=np.zeros((10, 10)),
            Y=np.zeros((10, 10)),
            fields={"u": np.ones((10, 10))},
            nx=10,
            ny=10,
            dx=0.1,
            dy=0.1,
        )

        with pytest.raises(ValueError, match="must have both u and v"):
            compute_velocity_magnitude(data)

    @pytest.mark.skipif(
        not cfd_python_integration.has_cfd_python(), reason="cfd-python not installed"
    )
    def test_cfd_python_matches_numpy(self):
        """Results should match between cfd-python and NumPy implementations."""
        u = [float(i) for i in range(100)]
        v = [float(100 - i) for i in range(100)]
        data = from_cfd_python(u, v, nx=10, ny=10)

        cfd_result = compute_velocity_magnitude(data, use_cfd_python=True)
        numpy_result = compute_velocity_magnitude(data, use_cfd_python=False)

        assert np.allclose(cfd_result, numpy_result)


class TestModuleLevelExports:
    """Test that stats functions are available from cfd_viz package."""

    def test_calculate_field_stats_exported(self):
        """calculate_field_stats should be importable from cfd_viz."""
        from cfd_viz import calculate_field_stats as fn

        assert callable(fn)

    def test_compute_flow_statistics_exported(self):
        """compute_flow_statistics should be importable from cfd_viz."""
        from cfd_viz import compute_flow_statistics as fn

        assert callable(fn)

    def test_compute_velocity_magnitude_exported(self):
        """compute_velocity_magnitude should be importable from cfd_viz."""
        from cfd_viz import compute_velocity_magnitude as fn

        assert callable(fn)
