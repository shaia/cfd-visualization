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
        ValueError: If result is not a dict or required keys are missing.
    """
    if not isinstance(result, dict):
        raise ValueError(
            f"Backend returned invalid field stats: "
            f"expected a dict, got {type(result).__name__}"
        )
    missing = FIELD_STATS_REQUIRED_KEYS - result.keys()
    if missing:
        raise ValueError(
            f"Backend returned incomplete field stats, missing keys: {sorted(missing)}"
        )
    return result


FLOW_STATS_REQUIRED_KEYS: Set[str] = {"u", "v", "velocity_magnitude"}


def validate_flow_statistics(result: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and return flow statistics dict.

    Checks top-level keys and validates each nested stats dict.

    Raises:
        ValueError: If required keys are missing.
    """
    missing = FLOW_STATS_REQUIRED_KEYS - result.keys()
    if missing:
        raise ValueError(
            f"Backend returned incomplete flow statistics, missing keys: {sorted(missing)}"
        )
    for value in result.values():
        validate_field_stats(value)
    return result


def validate_system_info(result: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and return system info dict.

    Raises:
        ValueError: If required keys are missing.
    """
    missing = SYSTEM_INFO_REQUIRED_KEYS - result.keys()
    if missing:
        raise ValueError(
            f"Backend returned incomplete system info, missing keys: {sorted(missing)}"
        )
    return result
