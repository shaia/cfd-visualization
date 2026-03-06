"""Pluggable computation backends for cfd-visualization.

Two backends are available:
- NumPy (always available, the fallback)
- cfd-python (preferred when installed, SIMD/GPU accelerated)
"""

from ._base import StatsBackend, SystemInfoBackend
from ._registry import get_info_backend, get_stats_backend
from ._schema import (
    validate_field_stats,
    validate_flow_statistics,
    validate_system_info,
)

__all__ = [
    "StatsBackend",
    "SystemInfoBackend",
    "get_stats_backend",
    "get_info_backend",
    "validate_field_stats",
    "validate_flow_statistics",
    "validate_system_info",
]
