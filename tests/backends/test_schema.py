"""Tests for backend schema validation."""

import pytest

from cfd_viz.backends._schema import validate_field_stats, validate_system_info


class TestValidateFieldStats:
    """Tests for validate_field_stats."""

    def test_valid_result_passes(self):
        result = {"min": 0.0, "max": 1.0, "avg": 0.5, "sum": 5.0}
        assert validate_field_stats(result) is result

    def test_missing_key_raises(self):
        result = {"min": 0.0, "max": 1.0, "avg": 0.5}
        with pytest.raises(ValueError, match="missing keys"):
            validate_field_stats(result)

    def test_extra_keys_allowed(self):
        result = {"min": 0.0, "max": 1.0, "avg": 0.5, "sum": 5.0, "extra": 1}
        assert validate_field_stats(result) is result


class TestValidateSystemInfo:
    """Tests for validate_system_info."""

    def test_valid_result_passes(self):
        result = {
            "cfd_python_available": False,
            "cfd_python_version": None,
            "backends": [],
            "simd": "unknown",
            "has_simd": False,
            "has_avx2": False,
            "has_neon": False,
            "gpu_available": False,
        }
        assert validate_system_info(result) is result

    def test_missing_key_raises(self):
        result = {"cfd_python_available": True}
        with pytest.raises(ValueError, match="missing keys"):
            validate_system_info(result)
