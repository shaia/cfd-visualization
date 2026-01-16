"""Tests for cfd_viz.cfd_python_integration module."""

import pytest

from cfd_viz import cfd_python_integration


class TestHasCfdPython:
    """Tests for has_cfd_python function."""

    def test_returns_bool(self):
        """Should return a boolean value."""
        result = cfd_python_integration.has_cfd_python()
        assert isinstance(result, bool)

    def test_consistent_across_calls(self):
        """Should return same value on multiple calls."""
        result1 = cfd_python_integration.has_cfd_python()
        result2 = cfd_python_integration.has_cfd_python()
        assert result1 == result2


class TestGetCfdPythonVersion:
    """Tests for get_cfd_python_version function."""

    def test_returns_string_or_none(self):
        """Should return string if available, None otherwise."""
        result = cfd_python_integration.get_cfd_python_version()
        assert result is None or isinstance(result, str)

    def test_consistent_with_has_cfd_python(self):
        """Version should be non-None when has_cfd_python is True."""
        if cfd_python_integration.has_cfd_python():
            assert cfd_python_integration.get_cfd_python_version() is not None
        else:
            assert cfd_python_integration.get_cfd_python_version() is None


class TestRequireCfdPython:
    """Tests for require_cfd_python function."""

    def test_does_not_raise_when_available(self):
        """Should not raise if cfd-python is available."""
        if cfd_python_integration.has_cfd_python():
            # Should not raise
            cfd_python_integration.require_cfd_python()

    def test_raises_import_error_when_unavailable(self):
        """Should raise ImportError if cfd-python is not available."""
        if not cfd_python_integration.has_cfd_python():
            with pytest.raises(ImportError) as exc_info:
                cfd_python_integration.require_cfd_python()
            assert "cfd-python is required" in str(exc_info.value)

    def test_includes_feature_in_error(self):
        """Should include feature name in error message."""
        if not cfd_python_integration.has_cfd_python():
            with pytest.raises(ImportError) as exc_info:
                cfd_python_integration.require_cfd_python("live simulation")
            assert "live simulation" in str(exc_info.value)


class TestGetCfdPython:
    """Tests for get_cfd_python function."""

    def test_returns_module_when_available(self):
        """Should return cfd_python module if available."""
        if cfd_python_integration.has_cfd_python():
            module = cfd_python_integration.get_cfd_python()
            assert module is not None
            assert hasattr(module, "create_grid")

    def test_raises_when_unavailable(self):
        """Should raise ImportError if cfd-python is not available."""
        if not cfd_python_integration.has_cfd_python():
            with pytest.raises(ImportError):
                cfd_python_integration.get_cfd_python()


class TestCheckCfdPythonVersion:
    """Tests for check_cfd_python_version function."""

    def test_returns_bool(self):
        """Should return boolean."""
        result = cfd_python_integration.check_cfd_python_version("0.1.0")
        assert isinstance(result, bool)

    def test_false_when_not_installed(self):
        """Should return False if cfd-python is not installed."""
        if not cfd_python_integration.has_cfd_python():
            assert cfd_python_integration.check_cfd_python_version("0.1.0") is False

    def test_version_comparison(self):
        """Should correctly compare versions when available."""
        if cfd_python_integration.has_cfd_python():
            # Should pass for older version requirement
            assert cfd_python_integration.check_cfd_python_version("0.0.1") is True
            # Should pass for current or older version
            assert cfd_python_integration.check_cfd_python_version("0.1.6") is True
            # Should fail for impossibly high version
            assert cfd_python_integration.check_cfd_python_version("99.0.0") is False


class TestRequireCfdPythonVersion:
    """Tests for require_cfd_python_version function."""

    def test_does_not_raise_when_version_met(self):
        """Should not raise if version requirement is met."""
        if cfd_python_integration.has_cfd_python():
            # Should not raise for low version requirement
            cfd_python_integration.require_cfd_python_version("0.0.1")

    def test_raises_when_not_installed(self):
        """Should raise ImportError if cfd-python is not installed."""
        if not cfd_python_integration.has_cfd_python():
            with pytest.raises(ImportError) as exc_info:
                cfd_python_integration.require_cfd_python_version("0.1.6")
            assert "cfd-python >= 0.1.6 is required" in str(exc_info.value)

    def test_raises_when_version_too_old(self):
        """Should raise ImportError if version is too old."""
        if cfd_python_integration.has_cfd_python():
            with pytest.raises(ImportError) as exc_info:
                cfd_python_integration.require_cfd_python_version("99.0.0", "future feature")
            assert "99.0.0" in str(exc_info.value)
            assert "future feature" in str(exc_info.value)


class TestModuleLevelExports:
    """Test that integration functions are exported from cfd_viz package."""

    def test_has_cfd_python_exported(self):
        """has_cfd_python should be importable from cfd_viz."""
        from cfd_viz import has_cfd_python

        assert callable(has_cfd_python)

    def test_get_cfd_python_exported(self):
        """get_cfd_python should be importable from cfd_viz."""
        from cfd_viz import get_cfd_python

        assert callable(get_cfd_python)

    def test_get_cfd_python_version_exported(self):
        """get_cfd_python_version should be importable from cfd_viz."""
        from cfd_viz import get_cfd_python_version

        assert callable(get_cfd_python_version)

    def test_require_cfd_python_exported(self):
        """require_cfd_python should be importable from cfd_viz."""
        from cfd_viz import require_cfd_python

        assert callable(require_cfd_python)

    def test_check_cfd_python_version_exported(self):
        """check_cfd_python_version should be importable from cfd_viz."""
        from cfd_viz import check_cfd_python_version

        assert callable(check_cfd_python_version)

    def test_require_cfd_python_version_exported(self):
        """require_cfd_python_version should be importable from cfd_viz."""
        from cfd_viz import require_cfd_python_version

        assert callable(require_cfd_python_version)


class TestIntegrationWithCfdPython:
    """Integration tests that run only when cfd-python is available."""

    @pytest.mark.skipif(
        not cfd_python_integration.has_cfd_python(), reason="cfd-python not installed"
    )
    def test_can_create_grid(self):
        """Should be able to use cfd-python to create a grid."""
        cfd = cfd_python_integration.get_cfd_python()
        grid = cfd.create_grid(10, 10, 0.0, 1.0, 0.0, 1.0)
        assert grid["nx"] == 10
        assert grid["ny"] == 10

    @pytest.mark.skipif(
        not cfd_python_integration.has_cfd_python(), reason="cfd-python not installed"
    )
    def test_backend_availability_api(self):
        """Should be able to query backend availability."""
        cfd = cfd_python_integration.get_cfd_python()
        backends = cfd.get_available_backends()
        assert isinstance(backends, list)
        # Backend names are lowercase in cfd-python
        backend_names_lower = [b.lower() for b in backends]
        assert "scalar" in backend_names_lower  # Scalar should always be available

    @pytest.mark.skipif(
        not cfd_python_integration.has_cfd_python(), reason="cfd-python not installed"
    )
    def test_simd_detection(self):
        """Should be able to detect SIMD capabilities."""
        cfd = cfd_python_integration.get_cfd_python()
        simd_name = cfd.get_simd_name()
        assert simd_name in ("avx2", "neon", "none")

