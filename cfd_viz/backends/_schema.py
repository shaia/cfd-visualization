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
        ValueError: If required keys are missing or have invalid structure/types.
    """
    if not isinstance(result, dict):
        raise ValueError(
            f"Backend returned invalid flow statistics: "
            f"expected a dict, got {type(result).__name__}"
        )
    missing = FLOW_STATS_REQUIRED_KEYS - result.keys()
    if missing:
        raise ValueError(
            f"Backend returned incomplete flow statistics, missing keys: {sorted(missing)}"
        )

    # Validate required flow statistic fields.
    for field_name in FLOW_STATS_REQUIRED_KEYS:
        value = result[field_name]
        if not isinstance(value, dict):
            raise ValueError(
                f"Backend returned invalid stats for '{field_name}': "
                f"expected a dict, got {type(value).__name__}"
            )
        validate_field_stats(value)

    # Optionally validate pressure statistics if present.
    if "p" in result:
        p_value = result["p"]
        if not isinstance(p_value, dict):
            raise ValueError(
                f"Backend returned invalid stats for 'p': "
                f"expected a dict, got {type(p_value).__name__}"
            )
        validate_field_stats(p_value)

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
