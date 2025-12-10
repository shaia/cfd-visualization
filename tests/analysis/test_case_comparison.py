"""Tests for cfd_viz.analysis.case_comparison module."""

import numpy as np
import pytest

from cfd_viz.analysis.case_comparison import (
    CaseComparison,
    FieldDifference,
    ParameterSweepResult,
    compare_fields,
    compute_convergence_metrics,
    compute_error_norms,
    compute_field_difference,
    parameter_sweep_analysis,
)


class TestComputeFieldDifference:
    """Tests for compute_field_difference function."""

    def test_identical_fields_zero_diff(self):
        """Identical fields should have zero difference."""
        field = np.random.rand(10, 10)
        diff = compute_field_difference(field, field)
        assert diff.max_diff == 0
        assert diff.min_diff == 0
        assert diff.mean_diff == 0
        assert diff.rms_diff == 0

    def test_constant_offset(self):
        """Constant offset should be detected correctly."""
        field1 = np.ones((10, 10))
        field2 = np.ones((10, 10)) + 0.5
        diff = compute_field_difference(field1, field2)
        np.testing.assert_almost_equal(diff.max_diff, 0.5)
        np.testing.assert_almost_equal(diff.min_diff, 0.5)
        np.testing.assert_almost_equal(diff.mean_diff, 0.5)

    def test_returns_field_difference(self):
        """Should return a FieldDifference dataclass."""
        field1 = np.random.rand(10, 10)
        field2 = np.random.rand(10, 10)
        diff = compute_field_difference(field1, field2)
        assert isinstance(diff, FieldDifference)

    def test_diff_array_shape(self):
        """Difference arrays should have same shape as input."""
        field1 = np.random.rand(20, 30)
        field2 = np.random.rand(20, 30)
        diff = compute_field_difference(field1, field2)
        assert diff.diff.shape == field1.shape
        assert diff.abs_diff.shape == field1.shape
        assert diff.relative_diff.shape == field1.shape

    def test_to_dict_excludes_arrays(self):
        """to_dict should return scalar metrics only."""
        field1 = np.random.rand(10, 10)
        field2 = np.random.rand(10, 10)
        diff = compute_field_difference(field1, field2)
        d = diff.to_dict()
        assert "max_diff" in d
        assert "rms_diff" in d
        assert "diff" not in d  # Array should not be in dict


class TestCompareFields:
    """Tests for compare_fields function."""

    def test_returns_case_comparison(self, random_flow):
        """Should return a CaseComparison dataclass."""
        u, v, p = random_flow["u"], random_flow["v"], random_flow["p"]
        dx, dy = random_flow["dx"], random_flow["dy"]
        comparison = compare_fields(u, v, p, u, v, p, dx, dy)
        assert isinstance(comparison, CaseComparison)

    def test_identical_fields_zero_comparison(self, random_flow):
        """Comparing field with itself should give zero differences."""
        u, v, p = random_flow["u"], random_flow["v"], random_flow["p"]
        dx, dy = random_flow["dx"], random_flow["dy"]
        comparison = compare_fields(u, v, p, u, v, p, dx, dy)
        assert comparison.velocity_diff.rms_diff == 0
        assert comparison.u_diff.rms_diff == 0
        assert comparison.v_diff.rms_diff == 0

    def test_mismatched_shapes_raises(self, random_flow):
        """Different shaped fields should raise ValueError."""
        u, v, p = random_flow["u"], random_flow["v"], random_flow["p"]
        dx, dy = random_flow["dx"], random_flow["dy"]
        u_small = u[:-1, :-1]
        v_small = v[:-1, :-1]
        with pytest.raises(ValueError):
            compare_fields(u, v, p, u_small, v_small, None, dx, dy)

    def test_handles_none_pressure(self, random_flow):
        """Should handle None pressure fields."""
        u, v = random_flow["u"], random_flow["v"]
        dx, dy = random_flow["dx"], random_flow["dy"]
        comparison = compare_fields(u, v, None, u * 1.1, v * 1.1, None, dx, dy)
        assert comparison.pressure_diff is None

    def test_metrics_comparison_has_entries(self, random_flow):
        """metrics_comparison should contain metric comparisons."""
        u, v, p = random_flow["u"], random_flow["v"], random_flow["p"]
        dx, dy = random_flow["dx"], random_flow["dy"]
        comparison = compare_fields(u, v, p, u * 1.1, v * 1.1, p * 1.1, dx, dy)
        assert "max_velocity" in comparison.metrics_comparison
        assert "mean_velocity" in comparison.metrics_comparison

    def test_summary_method(self, random_flow):
        """summary() should return key metrics."""
        u, v, p = random_flow["u"], random_flow["v"], random_flow["p"]
        dx, dy = random_flow["dx"], random_flow["dy"]
        comparison = compare_fields(u, v, p, u * 1.1, v * 1.1, p * 1.1, dx, dy)
        summary = comparison.summary()
        assert "velocity_rms_diff" in summary
        assert "vorticity_rms_diff" in summary


class TestParameterSweepAnalysis:
    """Tests for parameter_sweep_analysis function."""

    def test_returns_parameter_sweep_result(self, uniform_grid):
        """Should return ParameterSweepResult dataclass."""
        nx, ny = uniform_grid["nx"], uniform_grid["ny"]
        dx, dy = uniform_grid["dx"], uniform_grid["dy"]

        # Create cases with different velocities
        cases = [
            (np.ones((ny, nx)) * v, np.zeros((ny, nx)), None) for v in [1.0, 2.0, 3.0]
        ]
        param_values = [100, 200, 300]  # e.g., Reynolds numbers

        result = parameter_sweep_analysis(cases, param_values, "Re", dx, dy)
        assert isinstance(result, ParameterSweepResult)

    def test_sorted_by_parameter(self, uniform_grid):
        """Results should be sorted by parameter value."""
        nx, ny = uniform_grid["nx"], uniform_grid["ny"]
        dx, dy = uniform_grid["dx"], uniform_grid["dy"]

        cases = [
            (np.ones((ny, nx)) * v, np.zeros((ny, nx)), None) for v in [3.0, 1.0, 2.0]
        ]
        param_values = [300, 100, 200]  # Unsorted

        result = parameter_sweep_analysis(cases, param_values, "Re", dx, dy)
        assert result.parameter_values == [100, 200, 300]

    def test_trends_computed(self, uniform_grid):
        """Linear trends should be computed."""
        nx, ny = uniform_grid["nx"], uniform_grid["ny"]
        dx, dy = uniform_grid["dx"], uniform_grid["dy"]

        cases = [
            (np.ones((ny, nx)) * v, np.zeros((ny, nx)), None) for v in [1.0, 2.0, 3.0]
        ]
        param_values = [1, 2, 3]

        result = parameter_sweep_analysis(cases, param_values, "param", dx, dy)
        assert "max_velocity" in result.trends

    def test_mismatched_lengths_raises(self, uniform_grid):
        """Mismatched cases and parameter_values should raise."""
        nx, ny = uniform_grid["nx"], uniform_grid["ny"]
        dx, dy = uniform_grid["dx"], uniform_grid["dy"]

        cases = [(np.ones((ny, nx)), np.zeros((ny, nx)), None) for _ in range(3)]
        param_values = [1, 2]  # Only 2 values for 3 cases

        with pytest.raises(ValueError):
            parameter_sweep_analysis(cases, param_values, "param", dx, dy)


class TestComputeConvergenceMetrics:
    """Tests for compute_convergence_metrics function."""

    def test_empty_sequence(self):
        """Empty sequence should return converged."""
        result = compute_convergence_metrics([])
        assert result["is_converged"] is True

    def test_single_field(self):
        """Single field should return converged."""
        u = np.ones((10, 10))
        v = np.zeros((10, 10))
        result = compute_convergence_metrics([(u, v)])
        assert result["is_converged"] is True

    def test_converging_sequence(self):
        """Converging sequence should be detected."""
        fields = []
        for i in range(5):
            scale = 1.0 + 0.1 * (0.5**i)  # Decaying changes
            fields.append((np.ones((10, 10)) * scale, np.zeros((10, 10))))

        result = compute_convergence_metrics(fields)
        assert len(result["velocity_change"]) == 4  # n-1 changes


class TestComputeErrorNorms:
    """Tests for compute_error_norms function."""

    def test_identical_fields_zero_error(self):
        """Identical fields should have zero error."""
        u = np.random.rand(10, 10)
        v = np.random.rand(10, 10)
        norms = compute_error_norms(u, v, u, v)
        assert norms["u_l1"] == 0
        assert norms["u_l2"] == 0
        assert norms["u_linf"] == 0

    def test_constant_error(self):
        """Constant error should be captured correctly."""
        u1 = np.ones((10, 10))
        v1 = np.zeros((10, 10))
        u2 = np.ones((10, 10)) + 0.1
        v2 = np.zeros((10, 10))

        norms = compute_error_norms(u2, v2, u1, v1)
        np.testing.assert_almost_equal(norms["u_l1"], 0.1)
        np.testing.assert_almost_equal(norms["u_l2"], 0.1)
        np.testing.assert_almost_equal(norms["u_linf"], 0.1)

    def test_returns_all_norms(self):
        """Should return all expected error norms."""
        u1 = np.random.rand(10, 10)
        v1 = np.random.rand(10, 10)
        u2 = np.random.rand(10, 10)
        v2 = np.random.rand(10, 10)

        norms = compute_error_norms(u1, v1, u2, v2)
        assert "u_l1" in norms
        assert "u_l2" in norms
        assert "u_linf" in norms
        assert "v_l1" in norms
        assert "velocity_l2" in norms
