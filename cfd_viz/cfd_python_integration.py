"""cfd-python integration utilities.

This module provides detection and version checking for cfd-python,
enabling graceful fallback when it's not installed.

Example:
    >>> from cfd_viz.cfd_python_integration import has_cfd_python, get_cfd_python
    >>> if has_cfd_python():
    ...     cfd = get_cfd_python()
    ...     print(f"Using cfd-python {cfd.__version__}")
    ... else:
    ...     print("cfd-python not available, using NumPy fallback")
"""

from typing import Any, Optional

# Module-level state for cfd-python detection
_CFD_PYTHON_AVAILABLE: bool = False
_CFD_PYTHON_VERSION: Optional[str] = None
_cfd_python_module: Any = None

# Try to import cfd-python at module load time
try:
    import cfd_python as _cfd_python_module

    _CFD_PYTHON_AVAILABLE = True
    _CFD_PYTHON_VERSION = getattr(_cfd_python_module, "__version__", "unknown")
except ImportError:
    _cfd_python_module = None


def has_cfd_python() -> bool:
    """Check if cfd-python is available.

    Returns:
        True if cfd-python is installed and importable, False otherwise.

    Example:
        >>> if has_cfd_python():
        ...     # Use cfd-python features
        ...     pass
        ... else:
        ...     # Use NumPy fallback
        ...     pass
    """
    return _CFD_PYTHON_AVAILABLE


def get_cfd_python_version() -> Optional[str]:
    """Get the installed cfd-python version.

    Returns:
        Version string if cfd-python is installed, None otherwise.

    Example:
        >>> version = get_cfd_python_version()
        >>> if version:
        ...     print(f"cfd-python version: {version}")
    """
    return _CFD_PYTHON_VERSION


def require_cfd_python(feature: str = "this feature") -> None:
    """Raise ImportError if cfd-python is not available.

    Args:
        feature: Description of the feature requiring cfd-python,
            used in the error message.

    Raises:
        ImportError: If cfd-python is not installed.

    Example:
        >>> require_cfd_python("live simulation")  # Raises if not installed
    """
    if not _CFD_PYTHON_AVAILABLE:
        raise ImportError(
            f"cfd-python is required for {feature}. "
            f"Install with: pip install cfd-python>=0.1.6"
        )


def get_cfd_python() -> Any:
    """Get the cfd_python module, raising ImportError if unavailable.

    Returns:
        The cfd_python module.

    Raises:
        ImportError: If cfd-python is not installed.

    Example:
        >>> cfd = get_cfd_python()
        >>> result = cfd.run_simulation_with_params(nx=50, ny=50, steps=100)
    """
    require_cfd_python()
    return _cfd_python_module


def check_cfd_python_version(min_version: str = "0.1.6") -> bool:
    """Check if cfd-python meets minimum version requirement.

    Args:
        min_version: Minimum required version string (e.g., "0.1.6").

    Returns:
        True if cfd-python is installed and meets the version requirement,
        False if not installed or version is too old.

    Example:
        >>> if check_cfd_python_version("0.1.6"):
        ...     # Use features from 0.1.6+
        ...     pass
    """
    if not _CFD_PYTHON_AVAILABLE:
        return False

    if _CFD_PYTHON_VERSION == "unknown":
        # Assume compatible if version unknown
        return True

    try:
        from packaging import version

        return version.parse(_CFD_PYTHON_VERSION) >= version.parse(min_version)
    except ImportError:
        # packaging not available, do simple string comparison
        try:
            # Parse version tuples for comparison
            installed = tuple(int(x) for x in _CFD_PYTHON_VERSION.split(".")[:3])
            required = tuple(int(x) for x in min_version.split(".")[:3])
            return installed >= required
        except (ValueError, AttributeError):
            # Can't parse versions, assume compatible
            return True


def require_cfd_python_version(min_version: str = "0.1.6", feature: str = "") -> None:
    """Raise ImportError if cfd-python doesn't meet version requirement.

    Args:
        min_version: Minimum required version string.
        feature: Description of the feature requiring this version.

    Raises:
        ImportError: If cfd-python is not installed or version is too old.

    Example:
        >>> require_cfd_python_version("0.1.6", "backend availability API")
    """
    if not _CFD_PYTHON_AVAILABLE:
        feature_msg = f" for {feature}" if feature else ""
        raise ImportError(
            f"cfd-python >= {min_version} is required{feature_msg}. "
            f"Install with: pip install cfd-python>={min_version}"
        )

    if not check_cfd_python_version(min_version):
        feature_msg = f" for {feature}" if feature else ""
        raise ImportError(
            f"cfd-python >= {min_version} is required{feature_msg}, "
            f"but {_CFD_PYTHON_VERSION} is installed. "
            f"Upgrade with: pip install --upgrade cfd-python>={min_version}"
        )
