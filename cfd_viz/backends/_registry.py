"""Backend auto-detection and selection."""

import functools

from ..cfd_python_integration import (
    MIN_CFD_PYTHON_VERSION,
    check_cfd_python_version,
    has_cfd_python,
)
from ._base import StatsBackend, SystemInfoBackend


def _cfd_python_usable() -> bool:
    """Check if cfd-python is available and meets the minimum version."""
    return has_cfd_python() and check_cfd_python_version(MIN_CFD_PYTHON_VERSION)


@functools.lru_cache(maxsize=4)
def get_stats_backend(use_cfd_python: bool = True) -> StatsBackend:
    """Return the appropriate stats backend.

    Args:
        use_cfd_python: If True and cfd-python is available, use it.
            If False, always use NumPy.
    """
    if use_cfd_python and _cfd_python_usable():
        from ._cfd_python import CfdPythonStatsBackend

        return CfdPythonStatsBackend()

    from ._numpy import NumPyStatsBackend

    return NumPyStatsBackend()


@functools.lru_cache(maxsize=2)
def get_info_backend() -> SystemInfoBackend:
    """Return the appropriate system info backend."""
    if _cfd_python_usable():
        from ._cfd_python import CfdPythonSystemInfoBackend

        return CfdPythonSystemInfoBackend()

    from ._numpy import NumPySystemInfoBackend

    return NumPySystemInfoBackend()
