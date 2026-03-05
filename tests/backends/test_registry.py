"""Tests for backend registry."""

from cfd_viz import cfd_python_integration
from cfd_viz.backends import get_info_backend, get_stats_backend
from cfd_viz.backends._numpy import NumPyStatsBackend, NumPySystemInfoBackend


class TestGetStatsBackend:
    """Tests for get_stats_backend."""

    def test_returns_numpy_when_disabled(self):
        backend = get_stats_backend(use_cfd_python=False)
        assert isinstance(backend, NumPyStatsBackend)

    def test_returns_correct_type_when_enabled(self):
        backend = get_stats_backend(use_cfd_python=True)
        if cfd_python_integration.has_cfd_python():
            from cfd_viz.backends._cfd_python import CfdPythonStatsBackend

            assert isinstance(backend, CfdPythonStatsBackend)
        else:
            assert isinstance(backend, NumPyStatsBackend)


class TestGetInfoBackend:
    """Tests for get_info_backend."""

    def test_returns_correct_type(self):
        backend = get_info_backend()
        if cfd_python_integration.has_cfd_python():
            from cfd_viz.backends._cfd_python import CfdPythonSystemInfoBackend

            assert isinstance(backend, CfdPythonSystemInfoBackend)
        else:
            assert isinstance(backend, NumPySystemInfoBackend)
