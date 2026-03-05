"""Lightweight result schema validation for catching API drift."""

from typing import Any, Dict, Set

FIELD_STATS_REQUIRED_KEYS: Set[str] = {"min", "max", "avg", "sum"}

SYSTEM_INFO_REQUIRED_KEYS: Set[str] = {
    "cfd_python_available",
    "cfd_python_version",
    "backends",
    "simd",
    "has_simd",
    "has_avx2",
    "has_neon",
    "gpu_available",
}


def validate_field_stats(result: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and return field stats dict.

    Raises:
        ValueError: If required keys are missing.
    """
    missing = FIELD_STATS_REQUIRED_KEYS - result.keys()
    if missing:
        raise ValueError(
            f"Backend returned incomplete field stats, missing keys: {missing}"
        )
    return result


def validate_system_info(result: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and return system info dict.

    Raises:
        ValueError: If required keys are missing.
    """
    missing = SYSTEM_INFO_REQUIRED_KEYS - result.keys()
    if missing:
        raise ValueError(
            f"Backend returned incomplete system info, missing keys: {missing}"
        )
    return result
