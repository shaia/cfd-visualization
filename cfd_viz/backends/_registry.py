"""Backend auto-detection and selection."""

from ..cfd_python_integration import has_cfd_python
from ._base import StatsBackend, SystemInfoBackend


def get_stats_backend(use_cfd_python: bool = True) -> StatsBackend:
    """Return the appropriate stats backend.

    Args:
        use_cfd_python: If True and cfd-python is available, use it.
            If False, always use NumPy.
    """
    if use_cfd_python and has_cfd_python():
        from ._cfd_python import CfdPythonStatsBackend

        return CfdPythonStatsBackend()

    from ._numpy import NumPyStatsBackend

    return NumPyStatsBackend()


def get_info_backend() -> SystemInfoBackend:
    """Return the appropriate system info backend."""
    if has_cfd_python():
        from ._cfd_python import CfdPythonSystemInfoBackend

        return CfdPythonSystemInfoBackend()

    from ._numpy import NumPySystemInfoBackend

    return NumPySystemInfoBackend()
