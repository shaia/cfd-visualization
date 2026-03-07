"""Global plotting defaults for cfd-visualization.

Provides a centralized PlotDefaults dataclass so researchers can
set preferences once instead of passing them to every function.

Usage:
    >>> import cfd_viz
    >>> cfd_viz.set_defaults(cmap="coolwarm", dpi=200)
    >>> cfd_viz.get_defaults().cmap
    'coolwarm'

    >>> with cfd_viz.plot_context(cmap="hot", dpi=300):
    ...     # all plots inside use "hot" colormap at 300 dpi
    ...     pass
    >>> # defaults restored automatically
"""

from __future__ import annotations

import contextvars
import copy
import threading
from contextlib import contextmanager
from dataclasses import dataclass, fields as dc_fields
from pathlib import Path
from typing import Any, Iterator, Tuple


class _UnsetType:
    """Sentinel for 'no value provided' (distinct from None)."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return "UNSET"

    def __bool__(self):
        return False


UNSET = _UnsetType()


@dataclass
class PlotDefaults:
    """Global plotting default values.

    Attributes:
        cmap: Default matplotlib colormap for sequential data.
        diverging_cmap: Default matplotlib colormap for diverging data
            (e.g., vorticity, field differences).
        sequential_cmap: Secondary sequential colormap (e.g., pressure).
        figsize: Default figure size (width, height) in inches.
        dpi: Default dots per inch for saved figures.
        levels: Default number of contour levels.
        font_size: Default font size for titles and labels.
        colorscale: Default Plotly colorscale for sequential data.
        diverging_colorscale: Default Plotly colorscale for diverging data.
    """

    cmap: str = "viridis"
    diverging_cmap: str = "RdBu_r"
    sequential_cmap: str = "plasma"
    figsize: Tuple[float, float] = (10, 8)
    dpi: int = 150
    levels: int = 20
    font_size: int = 12
    colorscale: str = "Viridis"
    diverging_colorscale: str = "RdBu"


_lock = threading.RLock()
_defaults = PlotDefaults()
_context_override: contextvars.ContextVar[PlotDefaults | None] = contextvars.ContextVar(
    "_context_override", default=None
)


def get_defaults() -> PlotDefaults:
    """Return a copy of the current global defaults.

    If called inside a ``plot_context()`` block, returns the
    context-local overrides instead of the global defaults.
    """
    override = _context_override.get()
    if override is not None:
        return copy.copy(override)
    with _lock:
        return copy.copy(_defaults)


def set_defaults(**kwargs: Any) -> None:
    """Update global defaults.

    Args:
        **kwargs: Field names from PlotDefaults and their new values.

    Raises:
        TypeError: If an unknown field name is passed.

    Example:
        >>> set_defaults(cmap="coolwarm", dpi=200)
    """
    valid_fields = {f.name for f in dc_fields(PlotDefaults)}
    unknown = set(kwargs) - valid_fields
    if unknown:
        raise TypeError(
            f"Unknown defaults: {', '.join(sorted(unknown))}. "
            f"Valid fields: {', '.join(sorted(valid_fields))}"
        )
    with _lock:
        for key, value in kwargs.items():
            setattr(_defaults, key, value)


def reset_defaults() -> None:
    """Reset all defaults to their original values."""
    global _defaults  # noqa: PLW0603
    with _lock:
        _defaults = PlotDefaults()


def resolve(value: Any, field_name: str) -> Any:
    """Resolve a parameter: return value if set, else the global default.

    Used inside plotting functions to resolve UNSET sentinels::

        actual_cmap = resolve(cmap, "cmap")

    Args:
        value: The caller-provided value (or UNSET).
        field_name: The PlotDefaults field to fall back to.

    Returns:
        The resolved value.
    """
    if isinstance(value, _UnsetType):
        return getattr(get_defaults(), field_name)
    return value


@contextmanager
def plot_context(**kwargs: Any) -> Iterator[PlotDefaults]:
    """Temporarily override defaults within a with-block.

    Args:
        **kwargs: Fields to override temporarily.

    Yields:
        The temporary PlotDefaults object.

    Example:
        >>> with plot_context(cmap="coolwarm", dpi=300):
        ...     quick_plot(u, v, nx, ny)  # uses coolwarm
        >>> # defaults restored automatically
    """
    valid_fields = {f.name for f in dc_fields(PlotDefaults)}
    unknown = set(kwargs) - valid_fields
    if unknown:
        raise TypeError(
            f"Unknown defaults: {', '.join(sorted(unknown))}. "
            f"Valid fields: {', '.join(sorted(valid_fields))}"
        )
    # Start from current effective defaults (context-local or global)
    base = get_defaults()
    for key, value in kwargs.items():
        setattr(base, key, value)
    token = _context_override.set(base)
    try:
        yield copy.copy(base)
    finally:
        _context_override.reset(token)


def load_config_file(path: str | None = None) -> bool:
    """Load defaults from a TOML config file.

    Search order (when *path* is None):
        1. ``cfd_viz.toml`` in the current directory
        2. ``pyproject.toml`` ``[tool.cfd_viz.defaults]`` in the current directory

    Args:
        path: Explicit path to a TOML file.

    Returns:
        True if config was loaded, False otherwise.
    """
    try:
        import tomllib  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        try:
            import tomli as tomllib  # type: ignore[import-not-found,no-redef]
        except ModuleNotFoundError:
            return False

    if path is not None:
        candidates = [Path(path)]
    else:
        candidates = [Path("cfd_viz.toml"), Path("pyproject.toml")]

    for filepath in candidates:
        if not filepath.exists():
            continue
        with open(filepath, "rb") as f:
            data = tomllib.load(f)

        if filepath.name == "pyproject.toml":
            defaults_data = data.get("tool", {}).get("cfd_viz", {}).get("defaults", {})
        else:
            defaults_data = data.get("defaults", data)

        if defaults_data:
            if "figsize" in defaults_data:
                defaults_data["figsize"] = tuple(defaults_data["figsize"])
            set_defaults(**defaults_data)
            return True

    return False
