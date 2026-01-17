"""Tests for cfd_viz.info module."""

import pytest

from cfd_viz.info import get_recommended_settings, get_system_info, print_system_info


class TestGetSystemInfo:
    """Tests for get_system_info function."""

    def test_returns_dict(self):
        """Should return a dictionary."""
        info = get_system_info()
        assert isinstance(info, dict)

    def test_has_required_keys(self):
        """Should have all required keys."""
        info = get_system_info()
        required_keys = [
            "cfd_python_available",
            "cfd_python_version",
            "backends",
            "simd",
            "has_simd",
            "has_avx2",
            "has_neon",
            "gpu_available",
        ]
        for key in required_keys:
            assert key in info, f"Missing key: {key}"

    def test_cfd_python_available_is_bool(self):
        """cfd_python_available should be a boolean."""
        info = get_system_info()
        assert isinstance(info["cfd_python_available"], bool)

    def test_backends_is_list(self):
        """backends should be a list."""
        info = get_system_info()
        assert isinstance(info["backends"], list)

    def test_with_cfd_python_available(self):
        """When cfd-python is available, should have version and backends."""
        info = get_system_info()
        if info["cfd_python_available"]:
            assert info["cfd_python_version"] is not None
            assert len(info["backends"]) > 0
            assert info["simd"] != "unknown"

    def test_simd_flags_are_bool(self):
        """SIMD flags should be booleans."""
        info = get_system_info()
        assert isinstance(info["has_simd"], bool)
        assert isinstance(info["has_avx2"], bool)
        assert isinstance(info["has_neon"], bool)

    def test_gpu_available_is_bool(self):
        """gpu_available should be a boolean."""
        info = get_system_info()
        assert isinstance(info["gpu_available"], bool)


class TestGetRecommendedSettings:
    """Tests for get_recommended_settings function."""

    def test_returns_dict(self):
        """Should return a dictionary."""
        settings = get_recommended_settings()
        assert isinstance(settings, dict)

    def test_has_required_keys(self):
        """Should have all required keys."""
        settings = get_recommended_settings()
        required_keys = [
            "parallel_frame_rendering",
            "recommended_workers",
            "use_gpu_acceleration",
            "chunk_size",
        ]
        for key in required_keys:
            assert key in settings, f"Missing key: {key}"

    def test_parallel_frame_rendering_is_bool(self):
        """parallel_frame_rendering should be a boolean."""
        settings = get_recommended_settings()
        assert isinstance(settings["parallel_frame_rendering"], bool)

    def test_recommended_workers_is_positive_int(self):
        """recommended_workers should be a positive integer."""
        settings = get_recommended_settings()
        assert isinstance(settings["recommended_workers"], int)
        assert settings["recommended_workers"] >= 1

    def test_chunk_size_is_positive_int(self):
        """chunk_size should be a positive integer."""
        settings = get_recommended_settings()
        assert isinstance(settings["chunk_size"], int)
        assert settings["chunk_size"] > 0

    def test_use_gpu_acceleration_is_bool(self):
        """use_gpu_acceleration should be a boolean."""
        settings = get_recommended_settings()
        assert isinstance(settings["use_gpu_acceleration"], bool)

    def test_chunk_size_reasonable_values(self):
        """chunk_size should be one of the expected values."""
        settings = get_recommended_settings()
        # Expected values: 256 (default), 512 (NEON), 1024 (AVX2)
        assert settings["chunk_size"] in [256, 512, 1024]

    def test_workers_capped_at_four(self):
        """recommended_workers should be at most 4."""
        settings = get_recommended_settings()
        assert settings["recommended_workers"] <= 4


class TestPrintSystemInfo:
    """Tests for print_system_info function."""

    def test_runs_without_error(self, capsys):
        """Should run without raising exceptions."""
        print_system_info()
        captured = capsys.readouterr()
        assert "cfd-visualization System Info" in captured.out

    def test_shows_cfd_python_status(self, capsys):
        """Should show cfd-python availability."""
        print_system_info()
        captured = capsys.readouterr()
        assert "cfd-python available:" in captured.out


class TestModuleLevelExports:
    """Test that info functions are available from cfd_viz package."""

    def test_get_system_info_exported(self):
        """get_system_info should be importable from cfd_viz."""
        from cfd_viz import get_system_info as fn

        assert callable(fn)

    def test_get_recommended_settings_exported(self):
        """get_recommended_settings should be importable from cfd_viz."""
        from cfd_viz import get_recommended_settings as fn

        assert callable(fn)

    def test_print_system_info_exported(self):
        """print_system_info should be importable from cfd_viz."""
        from cfd_viz import print_system_info as fn

        assert callable(fn)
