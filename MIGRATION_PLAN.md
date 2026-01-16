# CFD-Visualization Migration Plan

This document outlines the required changes to update cfd-visualization to work with cfd-python 0.1.6.

## Current State

- **cfd-visualization version:** 0.1.0
- **Target cfd-python:** 0.1.6
- **Status:** Phase 1 Complete

## What's New in cfd-python 0.1.6

cfd-python 0.1.6 introduces several new APIs that cfd-visualization can leverage:

### New Features Available

| Feature | Functions | Benefit for cfd-viz |
|---------|-----------|---------------------|
| **Backend Availability** | `get_available_backends()`, `backend_is_available()`, `backend_get_name()` | Performance hints, GPU detection |
| **Derived Fields** | `calculate_field_stats()`, `compute_velocity_magnitude()`, `compute_flow_statistics()` | Consistent statistics, faster computation |
| **Error Handling** | `CFDError` hierarchy, `raise_for_status()` | Better error messages from simulations |
| **CPU Features** | `has_avx2()`, `has_neon()`, `get_simd_name()` | Optimization hints |
| **Boundary Conditions** | `bc_apply_*` functions, `BC_TYPE_*`, `BC_EDGE_*` constants | Direct BC control from Python |
| **Grid Stretching** | `create_grid_stretched()` | Non-uniform grid support |

### cfd-python 0.1.6 API Summary

```python
import cfd_python

# Backend availability
cfd_python.get_available_backends()  # -> ['Scalar', 'SIMD', 'OpenMP', 'CUDA']
cfd_python.backend_is_available(cfd_python.BACKEND_CUDA)  # -> bool
cfd_python.backend_get_name(cfd_python.BACKEND_SIMD)  # -> 'SIMD'

# Derived fields
cfd_python.calculate_field_stats(field_list)  # -> {'min': ..., 'max': ..., 'avg': ..., 'sum': ...}
cfd_python.compute_velocity_magnitude(u, v, nx, ny)  # -> list
cfd_python.compute_flow_statistics(u, v, p, nx, ny)  # -> {'u': {...}, 'v': {...}, 'p': {...}, 'velocity_magnitude': {...}}

# CPU features
cfd_python.has_avx2()  # -> bool
cfd_python.has_neon()  # -> bool
cfd_python.get_simd_name()  # -> 'avx2' | 'neon' | 'none'

# Error handling
from cfd_python import CFDError, CFDMemoryError, CFDInvalidError, raise_for_status
raise_for_status(status_code, context="operation")

# Grid
cfd_python.create_grid(nx, ny, xmin, xmax, ymin, ymax)  # uniform
cfd_python.create_grid_stretched(nx, ny, xmin, xmax, ymin, ymax, beta)  # stretched

# Simulation
result = cfd_python.run_simulation_with_params(nx=100, ny=100, steps=500, ...)
```

---

## Migration Plan

### Phase 1: Add cfd-python Dependency (Critical)

**Priority:** P0 - Required foundation

**Estimated Effort:** 0.5 days

**Tasks:**

- [ ] **1.1 Update pyproject.toml**
  ```toml
  [project]
  dependencies = [
      "numpy",
      "matplotlib",
      "scipy",
      "pandas",
      "watchdog",
  ]

  [project.optional-dependencies]
  interactive = ["plotly", "dash"]
  simulation = ["cfd-python>=0.1.6"]  # NEW
  full = ["plotly", "dash", "cfd-python>=0.1.6"]  # NEW
  ```

- [ ] **1.2 Add cfd-python detection helper**
  ```python
  # cfd_viz/compat.py
  """Compatibility layer for optional cfd-python integration."""

  _CFD_PYTHON_AVAILABLE = False
  _CFD_PYTHON_VERSION = None

  try:
      import cfd_python
      _CFD_PYTHON_AVAILABLE = True
      _CFD_PYTHON_VERSION = getattr(cfd_python, '__version__', 'unknown')
  except ImportError:
      cfd_python = None

  def has_cfd_python() -> bool:
      """Check if cfd-python is available."""
      return _CFD_PYTHON_AVAILABLE

  def require_cfd_python(feature: str = "this feature"):
      """Raise ImportError if cfd-python is not available."""
      if not _CFD_PYTHON_AVAILABLE:
          raise ImportError(
              f"cfd-python is required for {feature}. "
              f"Install with: pip install cfd-python>=0.1.6"
          )

  def get_cfd_python():
      """Get cfd_python module, raising ImportError if unavailable."""
      require_cfd_python()
      return cfd_python
  ```

- [ ] **1.3 Add version check**
  ```python
  # In cfd_viz/compat.py
  from packaging import version

  def check_cfd_python_version(min_version: str = "0.1.6") -> bool:
      """Check if cfd-python meets minimum version requirement."""
      if not _CFD_PYTHON_AVAILABLE:
          return False
      if _CFD_PYTHON_VERSION == 'unknown':
          return True  # Assume compatible
      try:
          return version.parse(_CFD_PYTHON_VERSION) >= version.parse(min_version)
      except Exception:
          return True
  ```

**Success Criteria:**
- cfd-python is an optional dependency
- Code gracefully handles missing cfd-python
- Version requirements are enforced

---

### Phase 2: Create Conversion Utilities (Important)

**Priority:** P1 - Core integration

**Estimated Effort:** 1 day

**Tasks:**

- [ ] **2.1 Create conversion module**
  ```python
  # cfd_viz/convert.py
  """Conversion utilities for cfd-python integration."""

  import numpy as np
  from typing import Optional, Dict, Any, List

  from .common import VTKData
  from .compat import get_cfd_python, has_cfd_python


  def from_cfd_python(
      u: List[float],
      v: List[float],
      p: Optional[List[float]] = None,
      nx: int = 0,
      ny: int = 0,
      xmin: float = 0.0,
      xmax: float = 1.0,
      ymin: float = 0.0,
      ymax: float = 1.0,
  ) -> VTKData:
      """Convert cfd_python simulation results to VTKData for visualization.

      Args:
          u: Flat list of u-velocity values (row-major order)
          v: Flat list of v-velocity values
          p: Flat list of pressure values (optional)
          nx: Number of grid points in x
          ny: Number of grid points in y
          xmin, xmax, ymin, ymax: Domain bounds

      Returns:
          VTKData object ready for visualization

      Example:
          >>> result = cfd_python.run_simulation_with_params(nx=50, ny=50, ...)
          >>> data = from_cfd_python(
          ...     result['u'], result['v'], result['p'],
          ...     result['nx'], result['ny']
          ... )
          >>> plot_velocity_field(data.X, data.Y, data.u, data.v)
      """
      if nx <= 0 or ny <= 0:
          raise ValueError(f"Invalid grid dimensions: {nx}x{ny}")

      expected_size = nx * ny
      if len(u) != expected_size:
          raise ValueError(f"u has {len(u)} elements, expected {expected_size}")
      if len(v) != expected_size:
          raise ValueError(f"v has {len(v)} elements, expected {expected_size}")

      dx = (xmax - xmin) / (nx - 1) if nx > 1 else 1.0
      dy = (ymax - ymin) / (ny - 1) if ny > 1 else 1.0

      x = np.linspace(xmin, xmax, nx)
      y = np.linspace(ymin, ymax, ny)
      X, Y = np.meshgrid(x, y)

      fields = {
          'u': np.array(u).reshape((ny, nx)),
          'v': np.array(v).reshape((ny, nx)),
      }
      if p is not None:
          if len(p) != expected_size:
              raise ValueError(f"p has {len(p)} elements, expected {expected_size}")
          fields['p'] = np.array(p).reshape((ny, nx))

      return VTKData(
          x=x,
          y=y,
          X=X,
          Y=Y,
          fields=fields,
          nx=nx,
          ny=ny,
          dx=dx,
          dy=dy,
      )


  def from_simulation_result(result: Dict[str, Any]) -> VTKData:
      """Convert cfd_python simulation result dict to VTKData.

      Args:
          result: Dictionary returned by run_simulation_with_params()

      Returns:
          VTKData object ready for visualization
      """
      return from_cfd_python(
          u=result['u'],
          v=result['v'],
          p=result.get('p'),
          nx=result['nx'],
          ny=result['ny'],
          xmin=result.get('xmin', 0.0),
          xmax=result.get('xmax', 1.0),
          ymin=result.get('ymin', 0.0),
          ymax=result.get('ymax', 1.0),
      )


  def to_cfd_python(data: VTKData) -> Dict[str, Any]:
      """Convert VTKData to cfd_python-compatible dictionary.

      Args:
          data: VTKData object

      Returns:
          Dictionary with flat lists compatible with cfd_python functions
      """
      result = {
          'u': data.u.flatten().tolist() if data.u is not None else [],
          'v': data.v.flatten().tolist() if data.v is not None else [],
          'p': data.get('p').flatten().tolist() if data.get('p') is not None else None,
          'nx': data.nx,
          'ny': data.ny,
          'xmin': float(data.x.min()) if len(data.x) > 0 else 0.0,
          'xmax': float(data.x.max()) if len(data.x) > 0 else 1.0,
          'ymin': float(data.y.min()) if len(data.y) > 0 else 0.0,
          'ymax': float(data.y.max()) if len(data.y) > 0 else 1.0,
      }
      return result
  ```

- [ ] **2.2 Add tests for conversion**
  ```python
  # tests/test_convert.py
  import pytest
  import numpy as np
  from cfd_viz.convert import from_cfd_python, to_cfd_python, from_simulation_result

  def test_from_cfd_python_basic():
      u = [1.0] * 100
      v = [0.5] * 100
      data = from_cfd_python(u, v, nx=10, ny=10)
      assert data.nx == 10
      assert data.ny == 10
      assert data.u.shape == (10, 10)

  def test_roundtrip():
      u = list(range(100))
      v = list(range(100, 200))
      data = from_cfd_python(u, v, nx=10, ny=10)
      result = to_cfd_python(data)
      assert result['nx'] == 10
      assert result['ny'] == 10
      assert result['u'] == u
  ```

**Success Criteria:**
- VTKData can be created from cfd-python results
- VTKData can be converted back to cfd-python format
- Round-trip conversion preserves data

---

### Phase 3: Use cfd-python Statistics (Enhancement)

**Priority:** P1 - Consistency with simulation

**Estimated Effort:** 0.5 days

**Tasks:**

- [ ] **3.1 Add statistics wrapper that uses cfd-python when available**
  ```python
  # cfd_viz/stats.py
  """Field statistics with optional cfd-python acceleration."""

  import numpy as np
  from typing import Dict, Any, Union
  from numpy.typing import NDArray

  from .common import VTKData
  from .compat import has_cfd_python, get_cfd_python


  def calculate_field_stats(
      data: Union[NDArray, list],
      use_cfd_python: bool = True
  ) -> Dict[str, float]:
      """Calculate field statistics (min, max, avg, sum).

      Uses cfd-python's optimized implementation when available,
      falls back to NumPy otherwise.

      Args:
          data: Field data as array or flat list
          use_cfd_python: Whether to use cfd-python (if available)

      Returns:
          Dict with 'min', 'max', 'avg', 'sum' keys
      """
      if use_cfd_python and has_cfd_python():
          cfd = get_cfd_python()
          flat = data.flatten().tolist() if hasattr(data, 'flatten') else list(data)
          return cfd.calculate_field_stats(flat)

      # NumPy fallback
      arr = np.asarray(data)
      return {
          'min': float(np.nanmin(arr)),
          'max': float(np.nanmax(arr)),
          'avg': float(np.nanmean(arr)),
          'sum': float(np.nansum(arr)),
      }


  def compute_flow_statistics(
      data: VTKData,
      use_cfd_python: bool = True
  ) -> Dict[str, Dict[str, float]]:
      """Compute comprehensive flow statistics.

      Args:
          data: VTKData with u, v, and optionally p fields
          use_cfd_python: Whether to use cfd-python (if available)

      Returns:
          Dict with 'u', 'v', 'p', 'velocity_magnitude' sub-dicts
      """
      if use_cfd_python and has_cfd_python():
          cfd = get_cfd_python()
          u_flat = data.u.flatten().tolist()
          v_flat = data.v.flatten().tolist()
          p_flat = data.get('p')
          if p_flat is not None:
              p_flat = p_flat.flatten().tolist()
          else:
              p_flat = [0.0] * len(u_flat)
          return cfd.compute_flow_statistics(u_flat, v_flat, p_flat, data.nx, data.ny)

      # NumPy fallback
      vel_mag = np.sqrt(data.u**2 + data.v**2)
      result = {
          'u': calculate_field_stats(data.u, use_cfd_python=False),
          'v': calculate_field_stats(data.v, use_cfd_python=False),
          'velocity_magnitude': calculate_field_stats(vel_mag, use_cfd_python=False),
      }
      p = data.get('p')
      if p is not None:
          result['p'] = calculate_field_stats(p, use_cfd_python=False)
      return result
  ```

- [ ] **3.2 Add velocity magnitude using cfd-python**
  ```python
  # In cfd_viz/stats.py
  def compute_velocity_magnitude(
      data: VTKData,
      use_cfd_python: bool = True
  ) -> NDArray:
      """Compute velocity magnitude field.

      Args:
          data: VTKData with u, v fields
          use_cfd_python: Whether to use cfd-python (if available)

      Returns:
          2D array of velocity magnitudes
      """
      if use_cfd_python and has_cfd_python():
          cfd = get_cfd_python()
          u_flat = data.u.flatten().tolist()
          v_flat = data.v.flatten().tolist()
          mag_flat = cfd.compute_velocity_magnitude(u_flat, v_flat, data.nx, data.ny)
          return np.array(mag_flat).reshape((data.ny, data.nx))

      # NumPy fallback
      return np.sqrt(data.u**2 + data.v**2)
  ```

**Success Criteria:**
- Statistics match between cfd-python and NumPy implementations
- Graceful fallback when cfd-python unavailable
- Consistent results across visualization pipeline

---

### Phase 4: Quick Plotting Functions (Enhancement)

**Priority:** P2 - Convenience

**Estimated Effort:** 0.5 days

**Tasks:**

- [ ] **4.1 Create quick plotting module**
  ```python
  # cfd_viz/quick.py
  """Quick visualization functions for cfd-python results."""

  import numpy as np
  import matplotlib.pyplot as plt
  from typing import Optional, List, Union, Literal

  from .convert import from_cfd_python, from_simulation_result
  from .fields import magnitude, vorticity
  from .plotting import plot_contour_field, plot_velocity_field, plot_streamlines


  FieldType = Literal["velocity_magnitude", "vorticity", "u", "v", "p"]


  def quick_plot(
      u: List[float],
      v: List[float],
      nx: int,
      ny: int,
      field: FieldType = "velocity_magnitude",
      p: Optional[List[float]] = None,
      xmin: float = 0.0,
      xmax: float = 1.0,
      ymin: float = 0.0,
      ymax: float = 1.0,
      ax: Optional[plt.Axes] = None,
      **kwargs
  ):
      """Quick visualization of cfd-python simulation results.

      Args:
          u, v: Velocity component lists from cfd-python
          nx, ny: Grid dimensions
          field: Field to plot - "velocity_magnitude", "vorticity", "u", "v", "p"
          p: Pressure list (required if field="p")
          xmin, xmax, ymin, ymax: Domain bounds
          ax: Matplotlib axes (created if None)
          **kwargs: Additional arguments passed to plotting function

      Returns:
          matplotlib figure

      Example:
          >>> result = cfd_python.run_simulation_with_params(nx=50, ny=50, steps=500)
          >>> quick_plot(result['u'], result['v'], result['nx'], result['ny'])
      """
      data = from_cfd_python(u, v, p, nx, ny, xmin, xmax, ymin, ymax)

      if ax is None:
          fig, ax = plt.subplots(figsize=(8, 6))
      else:
          fig = ax.get_figure()

      if field == "velocity_magnitude":
          field_data = magnitude(data.u, data.v)
          title = kwargs.pop('title', 'Velocity Magnitude')
      elif field == "vorticity":
          field_data = vorticity(data.u, data.v, data.dx, data.dy)
          title = kwargs.pop('title', 'Vorticity')
      elif field == "u":
          field_data = data.u
          title = kwargs.pop('title', 'U Velocity')
      elif field == "v":
          field_data = data.v
          title = kwargs.pop('title', 'V Velocity')
      elif field == "p":
          if data.get('p') is None:
              raise ValueError("Pressure field required but not provided")
          field_data = data.get('p')
          title = kwargs.pop('title', 'Pressure')
      else:
          raise ValueError(f"Unknown field: {field}")

      return plot_contour_field(data.X, data.Y, field_data, title=title, ax=ax, **kwargs)


  def quick_plot_result(
      result: dict,
      field: FieldType = "velocity_magnitude",
      **kwargs
  ):
      """Quick visualization of run_simulation_with_params() result.

      Args:
          result: Dictionary returned by cfd_python.run_simulation_with_params()
          field: Field to plot
          **kwargs: Additional arguments passed to quick_plot

      Example:
          >>> result = cfd_python.run_simulation_with_params(nx=50, ny=50, steps=500)
          >>> quick_plot_result(result, field="vorticity")
      """
      return quick_plot(
          u=result['u'],
          v=result['v'],
          nx=result['nx'],
          ny=result['ny'],
          p=result.get('p'),
          xmin=result.get('xmin', 0.0),
          xmax=result.get('xmax', 1.0),
          ymin=result.get('ymin', 0.0),
          ymax=result.get('ymax', 1.0),
          **kwargs
      )
  ```

- [ ] **4.2 Export in package**
  ```python
  # Update cfd_viz/__init__.py
  from .convert import from_cfd_python, to_cfd_python, from_simulation_result
  from .quick import quick_plot, quick_plot_result
  from .stats import calculate_field_stats, compute_flow_statistics, compute_velocity_magnitude
  ```

**Success Criteria:**
- One-liner visualization from cfd-python results
- All field types supported
- Clean integration with existing plotting

---

### Phase 5: Backend-Aware Features (Enhancement)

**Priority:** P2 - Nice to have

**Estimated Effort:** 0.5 days

**Tasks:**

- [ ] **5.1 Add backend info helper**
  ```python
  # cfd_viz/info.py
  """System information and performance hints."""

  from typing import Dict, Any, List

  from .compat import has_cfd_python, get_cfd_python


  def get_system_info() -> Dict[str, Any]:
      """Get system information relevant to visualization performance.

      Returns:
          Dict with cfd-python availability, backends, SIMD info
      """
      info = {
          'cfd_python_available': has_cfd_python(),
          'cfd_python_version': None,
          'backends': [],
          'simd': 'unknown',
          'gpu_available': False,
      }

      if has_cfd_python():
          cfd = get_cfd_python()
          info['cfd_python_version'] = getattr(cfd, '__version__', 'unknown')
          info['backends'] = cfd.get_available_backends()
          info['simd'] = cfd.get_simd_name()
          info['gpu_available'] = cfd.backend_is_available(cfd.BACKEND_CUDA)

      return info


  def get_recommended_settings() -> Dict[str, Any]:
      """Get recommended visualization settings based on available backends.

      Returns:
          Dict with recommended chunk sizes, parallel rendering hints, etc.
      """
      info = get_system_info()

      settings = {
          'parallel_frame_rendering': False,
          'recommended_workers': 1,
          'use_gpu_acceleration': False,
          'chunk_size': 256,
      }

      if info['cfd_python_available']:
          if 'OpenMP' in info['backends']:
              settings['parallel_frame_rendering'] = True
              import os
              settings['recommended_workers'] = min(4, os.cpu_count() or 1)

          if info['simd'] == 'avx2':
              settings['chunk_size'] = 1024
          elif info['simd'] == 'neon':
              settings['chunk_size'] = 512

          settings['use_gpu_acceleration'] = info['gpu_available']

      return settings
  ```

- [ ] **5.2 Add info command/display**
  ```python
  # In cfd_viz/info.py
  def print_system_info():
      """Print system information to console."""
      info = get_system_info()

      print("CFD-Visualization System Info")
      print("=" * 40)
      print(f"cfd-python available: {info['cfd_python_available']}")
      if info['cfd_python_available']:
          print(f"cfd-python version: {info['cfd_python_version']}")
          print(f"Available backends: {', '.join(info['backends'])}")
          print(f"SIMD: {info['simd']}")
          print(f"GPU available: {info['gpu_available']}")
      else:
          print("  (Install cfd-python for enhanced features)")
  ```

**Success Criteria:**
- System info accurately reports capabilities
- Recommendations are useful for performance tuning

---

### Phase 6: Update Examples (Required)

**Priority:** P1 - Documentation

**Estimated Effort:** 0.5 days

**Tasks:**

- [ ] **6.1 Update basic_simulation.py**
  - Use `from_simulation_result()` for cleaner conversion
  - Add error handling with `CFDError`
  - Show `compute_flow_statistics()` usage

- [ ] **6.2 Create new integration example**
  ```python
  # examples/cfd_python_integration.py
  """Example: Direct cfd-python integration with cfd-visualization."""

  import cfd_python
  from cfd_viz import quick_plot_result, from_simulation_result
  from cfd_viz.stats import compute_flow_statistics
  from cfd_viz.info import print_system_info

  # Show system capabilities
  print_system_info()
  print()

  # Run simulation
  result = cfd_python.run_simulation_with_params(
      nx=64, ny=64,
      steps=500,
  )

  # Quick visualization (one-liner)
  quick_plot_result(result, field="velocity_magnitude")

  # Full statistics using cfd-python backend
  data = from_simulation_result(result)
  stats = compute_flow_statistics(data)
  print(f"Max velocity: {stats['velocity_magnitude']['max']:.4f}")
  ```

- [ ] **6.3 Add error handling example**
  ```python
  # examples/error_handling.py
  """Example: Error handling with cfd-python integration."""

  from cfd_viz.compat import require_cfd_python, has_cfd_python

  if not has_cfd_python():
      print("cfd-python not installed. Install with: pip install cfd-python>=0.1.6")
      exit(1)

  import cfd_python
  from cfd_python import CFDError, raise_for_status

  try:
      result = cfd_python.run_simulation_with_params(nx=10, ny=10, steps=100)
      status = result.get('status', 0)
      raise_for_status(status, "simulation")
  except CFDError as e:
      print(f"Simulation failed: {e}")
  ```

**Success Criteria:**
- Examples run successfully with cfd-python 0.1.6
- Examples demonstrate new features
- Examples show proper error handling

---

### Phase 7: Update Documentation (Required)

**Priority:** P1 - User guidance

**Estimated Effort:** 0.5 days

**Tasks:**

- [ ] **7.1 Update README.md**
  - Add cfd-python integration section
  - Show quick_plot usage
  - Document optional dependencies

- [ ] **7.2 Update ROADMAP.md**
  - Mark Phase 1 tasks as complete
  - Mark Phase 8 tasks as complete
  - Update version planning

- [ ] **7.3 Add docstrings**
  - All new functions documented
  - Examples in docstrings

**Success Criteria:**
- Clear installation instructions
- API documentation complete
- Examples in documentation

---

## File Changes Summary

### Files to Create

| File | Purpose |
|------|---------|
| `cfd_viz/compat.py` | cfd-python detection and version checking |
| `cfd_viz/convert.py` | Data conversion utilities |
| `cfd_viz/quick.py` | Quick plotting functions |
| `cfd_viz/stats.py` | Statistics with cfd-python acceleration |
| `cfd_viz/info.py` | System info and recommendations |
| `tests/test_convert.py` | Conversion tests |
| `tests/test_stats.py` | Statistics tests |
| `tests/test_compat.py` | Compatibility layer tests |
| `examples/cfd_python_integration.py` | Integration example |
| `examples/error_handling.py` | Error handling example |

### Files to Modify

| File | Changes |
|------|---------|
| `pyproject.toml` | Add cfd-python optional dependency |
| `cfd_viz/__init__.py` | Export new functions |
| `examples/basic_simulation.py` | Use new conversion utilities |
| `README.md` | Add integration documentation |
| `ROADMAP.md` | Update status |

---

## Timeline Estimate

| Phase | Duration | Cumulative |
|-------|----------|------------|
| Phase 1: Add Dependency | 0.5 days | 0.5 days |
| Phase 2: Conversion Utils | 1 day | 1.5 days |
| Phase 3: Statistics | 0.5 days | 2 days |
| Phase 4: Quick Plotting | 0.5 days | 2.5 days |
| Phase 5: Backend Info | 0.5 days | 3 days |
| Phase 6: Examples | 0.5 days | 3.5 days |
| Phase 7: Documentation | 0.5 days | 4 days |

**Total estimated effort:** ~4 days

---

## Success Criteria

1. cfd-python 0.1.6 is an optional dependency
2. Visualization works without cfd-python (NumPy fallback)
3. Quick plotting from simulation results in one line
4. Statistics computed via cfd-python when available
5. System info shows backend capabilities
6. All existing tests pass
7. New integration tests pass
8. Examples demonstrate new features

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| cfd-python not installed | Low | Graceful fallback to NumPy |
| API changes in future cfd-python | Medium | Version pinning, compat layer |
| Different results NumPy vs cfd-python | Low | Test both paths, document differences |
| Performance regression | Low | Benchmark both implementations |

---

## Version Planning

| Version | Phases | Focus |
|---------|--------|-------|
| 0.2.0 | 1-4, 6-7 | cfd-python 0.1.6 integration |
| 0.3.0 | 5 + Jupyter | Backend info, Jupyter widgets |

---

## Related Documents

- [cfd-python MIGRATION_PLAN.md](../cfd-python/MIGRATION_PLAN.md) - cfd-python v0.1.6 migration details
- [cfd-python ROADMAP.md](../cfd-python/ROADMAP.md) - cfd-python future plans
- [ROADMAP.md](ROADMAP.md) - cfd-visualization roadmap
