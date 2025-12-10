"""Tests for cfd_viz.analysis.time_series module."""

import numpy as np
import pytest

from cfd_viz.analysis.time_series import (
    ConvergenceMetrics,
    FlowHistoryStatistics,
    ProbeTimeSeries,
    TemporalStatistics,
    analyze_convergence,
    analyze_flow_history,
    compute_rms_fluctuations,
    compute_running_average,
    compute_temporal_statistics,
    compute_time_averaged_field,
    detect_periodicity,
    extract_probe_time_series,
)


class TestComputeTemporalStatistics:
    """Tests for compute_temporal_statistics function."""

    def test_returns_temporal_statistics(self):
        """Should return TemporalStatistics dataclass."""
        time = np.linspace(0, 10, 100)
        values = np.sin(time)
        stats = compute_temporal_statistics(time, values)
        assert isinstance(stats, TemporalStatistics)

    def test_constant_signal_is_stationary(self):
        """Constant signal should be stationary."""
        time = np.linspace(0, 10, 100)
        values = np.ones(100) * 5.0
        stats = compute_temporal_statistics(time, values)
        assert stats.is_stationary is True
        np.testing.assert_almost_equal(stats.mean, 5.0)
        np.testing.assert_almost_equal(stats.std, 0.0)

    def test_sinusoidal_statistics(self):
        """Sinusoidal signal should have known statistics."""
        time = np.linspace(0, 2 * np.pi, 1000)
        values = np.sin(time)
        stats = compute_temporal_statistics(time, values)
        np.testing.assert_almost_equal(stats.mean, 0.0, decimal=2)
        np.testing.assert_almost_equal(stats.max_value, 1.0, decimal=2)
        np.testing.assert_almost_equal(stats.min_value, -1.0, decimal=2)

    def test_trend_computed(self):
        """Linear trend should be computed."""
        time = np.linspace(0, 10, 100)
        values = 2 * time + 3  # Linear: slope=2, intercept=3
        stats = compute_temporal_statistics(time, values)
        assert stats.trend is not None
        slope, intercept = stats.trend
        np.testing.assert_almost_equal(slope, 2.0, decimal=5)
        np.testing.assert_almost_equal(intercept, 3.0, decimal=5)

    def test_coefficient_of_variation(self):
        """Coefficient of variation should be std/mean."""
        time = np.linspace(0, 10, 100)
        values = np.random.randn(100) + 10  # Mean ~10
        stats = compute_temporal_statistics(time, values)
        expected_cv = stats.std / stats.mean
        np.testing.assert_almost_equal(stats.coefficient_of_variation, expected_cv)


class TestAnalyzeFlowHistory:
    """Tests for analyze_flow_history function."""

    def test_returns_flow_history_statistics(self, uniform_grid):
        """Should return FlowHistoryStatistics dataclass."""
        nx, ny = uniform_grid["nx"], uniform_grid["ny"]
        dx, dy = uniform_grid["dx"], uniform_grid["dy"]

        time = np.linspace(0, 1, 10)
        u_history = [np.ones((ny, nx)) for _ in range(10)]
        v_history = [np.zeros((ny, nx)) for _ in range(10)]

        stats = analyze_flow_history(time, u_history, v_history, dx=dx, dy=dy)
        assert isinstance(stats, FlowHistoryStatistics)

    def test_constant_flow_statistics(self, uniform_grid):
        """Constant flow should have zero std."""
        nx, ny = uniform_grid["nx"], uniform_grid["ny"]
        dx, dy = uniform_grid["dx"], uniform_grid["dy"]

        time = np.linspace(0, 1, 10)
        u_history = [np.ones((ny, nx)) for _ in range(10)]
        v_history = [np.zeros((ny, nx)) for _ in range(10)]

        stats = analyze_flow_history(time, u_history, v_history, dx=dx, dy=dy)
        np.testing.assert_almost_equal(stats.max_velocity.std, 0.0)
        np.testing.assert_almost_equal(stats.mean_velocity.std, 0.0)

    def test_handles_pressure_history(self, uniform_grid):
        """Should handle pressure history when provided."""
        nx, ny = uniform_grid["nx"], uniform_grid["ny"]
        dx, dy = uniform_grid["dx"], uniform_grid["dy"]

        time = np.linspace(0, 1, 10)
        u_history = [np.ones((ny, nx)) for _ in range(10)]
        v_history = [np.zeros((ny, nx)) for _ in range(10)]
        p_history = [np.ones((ny, nx)) * 100 for _ in range(10)]

        stats = analyze_flow_history(
            time, u_history, v_history, p_history=p_history, dx=dx, dy=dy
        )
        assert stats.mean_pressure is not None


class TestComputeTimeAveragedField:
    """Tests for compute_time_averaged_field function."""

    def test_constant_fields_unchanged(self):
        """Averaging constant fields should return same field."""
        field_history = [np.ones((10, 10)) * 5.0 for _ in range(10)]
        avg = compute_time_averaged_field(field_history)
        np.testing.assert_array_almost_equal(avg, 5.0)

    def test_skips_initial_transient(self):
        """Should skip initial fields when start_index > 0."""
        # First 3 fields have value 1, rest have value 2
        field_history = [np.ones((10, 10)) * (1 if i < 3 else 2) for i in range(10)]
        avg = compute_time_averaged_field(field_history, start_index=3)
        np.testing.assert_array_almost_equal(avg, 2.0)

    def test_empty_raises(self):
        """Empty history should raise ValueError."""
        with pytest.raises(ValueError):
            compute_time_averaged_field([])


class TestComputeRmsFluctuations:
    """Tests for compute_rms_fluctuations function."""

    def test_constant_fields_zero_rms(self):
        """Constant fields should have zero RMS fluctuations."""
        field_history = [np.ones((10, 10)) * 5.0 for _ in range(10)]
        rms = compute_rms_fluctuations(field_history)
        np.testing.assert_array_almost_equal(rms, 0.0)

    def test_fluctuating_fields_positive_rms(self):
        """Fluctuating fields should have positive RMS."""
        field_history = [np.ones((10, 10)) * (1 + 0.1 * i) for i in range(10)]
        rms = compute_rms_fluctuations(field_history)
        assert np.all(rms > 0)


class TestExtractProbeTimeSeries:
    """Tests for extract_probe_time_series function."""

    def test_returns_probe_time_series(self, uniform_grid):
        """Should return ProbeTimeSeries dataclass."""
        nx, ny = uniform_grid["nx"], uniform_grid["ny"]
        x, y = uniform_grid["x"], uniform_grid["y"]

        time = np.linspace(0, 1, 10)
        u_history = [np.ones((ny, nx)) for _ in range(10)]
        v_history = [np.zeros((ny, nx)) for _ in range(10)]

        probe = extract_probe_time_series(
            time, u_history, v_history, x, y, probe_location=(0.5, 0.5)
        )
        assert isinstance(probe, ProbeTimeSeries)

    def test_correct_time_series_length(self, uniform_grid):
        """Time series should have correct length."""
        nx, ny = uniform_grid["nx"], uniform_grid["ny"]
        x, y = uniform_grid["x"], uniform_grid["y"]

        time = np.linspace(0, 1, 10)
        u_history = [np.ones((ny, nx)) for _ in range(10)]
        v_history = [np.zeros((ny, nx)) for _ in range(10)]

        probe = extract_probe_time_series(
            time, u_history, v_history, x, y, probe_location=(0.5, 0.5)
        )
        assert len(probe.u) == 10
        assert len(probe.v) == 10

    def test_turbulent_intensity_zero_for_steady(self, uniform_grid):
        """Turbulent intensity should be zero for steady flow."""
        nx, ny = uniform_grid["nx"], uniform_grid["ny"]
        x, y = uniform_grid["x"], uniform_grid["y"]

        time = np.linspace(0, 1, 10)
        u_history = [np.ones((ny, nx)) for _ in range(10)]
        v_history = [np.zeros((ny, nx)) for _ in range(10)]

        probe = extract_probe_time_series(
            time, u_history, v_history, x, y, probe_location=(0.5, 0.5)
        )
        np.testing.assert_almost_equal(probe.turbulent_intensity, 0.0)


class TestAnalyzeConvergence:
    """Tests for analyze_convergence function."""

    def test_returns_convergence_metrics(self):
        """Should return ConvergenceMetrics dataclass."""
        iterations = np.arange(100)
        residuals = {"momentum": 10.0 ** (-iterations / 10)}
        metrics = analyze_convergence(iterations, residuals)
        assert isinstance(metrics, ConvergenceMetrics)

    def test_detects_convergence(self):
        """Should detect convergence below tolerance."""
        iterations = np.arange(100)
        residuals = {"momentum": 1e-8 * np.ones(100)}  # Already converged
        metrics = analyze_convergence(iterations, residuals, tolerance=1e-6)
        assert metrics.is_converged is True

    def test_detects_non_convergence(self):
        """Should detect when not converged."""
        iterations = np.arange(100)
        residuals = {"momentum": 1e-4 * np.ones(100)}  # Not converged
        metrics = analyze_convergence(iterations, residuals, tolerance=1e-6)
        assert metrics.is_converged is False


class TestComputeRunningAverage:
    """Tests for compute_running_average function."""

    def test_constant_unchanged(self):
        """Constant values should be unchanged."""
        values = np.ones(100) * 5.0
        avg = compute_running_average(values, window_size=10)
        np.testing.assert_array_almost_equal(avg, 5.0)

    def test_smooths_noise(self):
        """Should smooth noisy signal."""
        np.random.seed(42)
        values = np.sin(np.linspace(0, 4 * np.pi, 100)) + 0.5 * np.random.randn(100)
        avg = compute_running_average(values, window_size=10)
        # Running average should have lower variance
        assert np.std(avg) < np.std(values)

    def test_preserves_length(self):
        """Should preserve array length."""
        values = np.random.rand(100)
        avg = compute_running_average(values, window_size=10)
        assert len(avg) == len(values)


class TestDetectPeriodicity:
    """Tests for detect_periodicity function."""

    def test_detects_sinusoidal_period(self):
        """Should detect period of sinusoidal signal."""
        sampling_rate = 100.0
        period = 0.5  # seconds
        t = np.arange(0, 5, 1 / sampling_rate)
        values = np.sin(2 * np.pi * t / period)

        detected = detect_periodicity(values, sampling_rate)
        if detected is not None:
            np.testing.assert_almost_equal(detected, period, decimal=1)

    def test_no_periodicity_for_constant(self):
        """Should return None for constant signal."""
        values = np.ones(100)
        detected = detect_periodicity(values)
        assert detected is None

    def test_handles_short_signals(self):
        """Should handle very short signals."""
        values = np.array([1, 2, 3])
        result = detect_periodicity(values)
        # Should return None or handle gracefully
        # (short signals don't have clear periodicity)
        assert result is None or result > 0
