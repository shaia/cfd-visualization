"""System information and performance hints.

Query cfd-python backend availability and provide optimization recommendations.
"""

import os
from typing import Any

from .backends import get_info_backend, validate_system_info


def get_system_info() -> dict[str, Any]:
    """Get system information relevant to visualization performance.

    Returns:
        Dict with cfd-python availability, backends, SIMD info, and GPU status.

    Example:
        >>> info = get_system_info()
        >>> print(f"cfd-python available: {info['cfd_python_available']}")
        >>> print(f"SIMD: {info['simd']}")
        >>> print(f"Backends: {info['backends']}")
    """
    backend = get_info_backend()
    return validate_system_info(backend.get_system_info())


def get_recommended_settings() -> dict[str, Any]:
    """Get recommended visualization settings based on available backends.

    Returns:
        Dict with recommended chunk sizes, parallel rendering hints, etc.

    Example:
        >>> settings = get_recommended_settings()
        >>> if settings['parallel_frame_rendering']:
        ...     print(f"Use {settings['recommended_workers']} workers")
    """
    info = get_system_info()

    settings: dict[str, Any] = {
        "parallel_frame_rendering": False,
        "recommended_workers": 1,
        "use_gpu_acceleration": False,
        "chunk_size": 256,
    }

    if info["cfd_python_available"]:
        # Enable parallel rendering if OpenMP is available
        # Use case-insensitive check for robustness
        backends_lower = [b.lower() for b in info["backends"]]
        if "openmp" in backends_lower:
            settings["parallel_frame_rendering"] = True
            settings["recommended_workers"] = min(4, os.cpu_count() or 1)

        # Adjust chunk size based on SIMD capabilities
        if info["simd"] == "avx2":
            settings["chunk_size"] = 1024
        elif info["simd"] == "neon":
            settings["chunk_size"] = 512

        settings["use_gpu_acceleration"] = info["gpu_available"]

    return settings


def print_system_info() -> None:
    """Print system information to console.

    Useful for debugging and understanding available optimizations.

    Example:
        >>> print_system_info()
        cfd-visualization System Info
        ========================================
        cfd-python available: True
        cfd-python version: 0.1.6
        ...
    """
    info = get_system_info()
    settings = get_recommended_settings()

    print("cfd-visualization System Info")
    print("=" * 40)
    print(f"cfd-python available: {info['cfd_python_available']}")

    if info["cfd_python_available"]:
        print(f"cfd-python version: {info['cfd_python_version']}")
        print(f"Available backends: {', '.join(info['backends'])}")
        print(f"SIMD: {info['simd']}")
        print(f"  Has AVX2: {info['has_avx2']}")
        print(f"  Has NEON: {info['has_neon']}")
        print(f"GPU available: {info['gpu_available']}")
        print()
        print("Recommended Settings:")
        print(f"  Parallel rendering: {settings['parallel_frame_rendering']}")
        print(f"  Workers: {settings['recommended_workers']}")
        print(f"  Chunk size: {settings['chunk_size']}")
        print(f"  GPU acceleration: {settings['use_gpu_acceleration']}")
    else:
        print("Install cfd-python for optimized computations:")
        print("  pip install cfd-python")
