# CFD-Visualization Future Roadmap

This document outlines planned enhancements and new features for cfd-visualization.

**Last Updated:** 2026-01-22
**Current Version:** 0.1.0
**Target Version:** 0.2.0+

---

## Overview

`cfd-visualization` provides visualization and analysis tools for CFD simulation results. This roadmap focuses on enhancing visualization capabilities, improving integration with simulation libraries, and adding interactive features.

### Current Capabilities

- **VTK I/O**: `read_vtk_file()`, `VTKData` class
- **Field computations**: Vorticity, gradients, magnitude, derived fields
- **Plotting**: Contours, vectors, streamlines, quiver plots
- **Analysis**: Boundary layer, line extraction, case comparison, flow features
- **Animation**: Frame generation, GIF/MP4 export
- **Interactive**: Plotly dashboards

---

## Phase 1: Integration with cfd-python ✅ COMPLETE

**Priority:** P1 - Enables seamless simulation-to-visualization workflow
**Estimated Effort:** 2-3 days
**Status:** ✅ Completed (2026-01-22)

### Goals

- Provide seamless data flow from cfd-python simulations to visualization
- Enable direct visualization of simulation results
- Bi-directional data conversion utilities

### Tasks

- [x] **1.1 Create conversion utilities**
  ```python
  # cfd_viz/convert.py
  """Conversion utilities for cfd-python integration."""

  import numpy as np
  from .common import VTKData

  def from_cfd_python(
      u: list, v: list, p: list,
      nx: int, ny: int,
      xmin: float = 0.0, xmax: float = 1.0,
      ymin: float = 0.0, ymax: float = 1.0
  ) -> VTKData:
      """Convert cfd_python simulation results to VTKData for visualization.

      Args:
          u: Flat list of u-velocity values
          v: Flat list of v-velocity values
          p: Flat list of pressure values
          nx: Number of grid points in x
          ny: Number of grid points in y
          xmin, xmax, ymin, ymax: Domain bounds

      Returns:
          VTKData object ready for visualization
      """
      dx = (xmax - xmin) / (nx - 1) if nx > 1 else 1.0
      dy = (ymax - ymin) / (ny - 1) if ny > 1 else 1.0

      return VTKData(
          u=np.array(u).reshape((ny, nx)),
          v=np.array(v).reshape((ny, nx)),
          p=np.array(p).reshape((ny, nx)) if p else None,
          x=np.linspace(xmin, xmax, nx),
          y=np.linspace(ymin, ymax, ny),
          dx=dx,
          dy=dy
      )

  def to_cfd_python(data: VTKData) -> dict:
      """Convert VTKData to cfd_python-compatible dictionary.

      Args:
          data: VTKData object

      Returns:
          Dictionary with flat lists compatible with cfd_python functions
      """
      return {
          'u': data.u.flatten().tolist(),
          'v': data.v.flatten().tolist(),
          'p': data.p.flatten().tolist() if data.p is not None else None,
          'nx': data.u.shape[1],
          'ny': data.u.shape[0],
          'x': data.x.tolist() if isinstance(data.x, np.ndarray) else data.x,
          'y': data.y.tolist() if isinstance(data.y, np.ndarray) else data.y,
      }
  ```

- [x] **1.2 Add quick plotting wrapper**
  ```python
  # cfd_viz/quick.py
  """Quick visualization functions for simulation results."""

  from .convert import from_cfd_python
  from .fields import magnitude, vorticity
  from .plotting import plot_contour_field, plot_velocity_field

  def quick_plot(
      u: list, v: list,
      nx: int, ny: int,
      field: str = "velocity_magnitude",
      **kwargs
  ):
      """Quick visualization of simulation results.

      Args:
          u, v: Velocity component lists
          nx, ny: Grid dimensions
          field: Field to plot - "velocity_magnitude", "vorticity", "u", "v"
          **kwargs: Additional arguments passed to plot_contour_field

      Returns:
          matplotlib figure
      """
      data = from_cfd_python(u, v, None, nx, ny)
      X, Y = np.meshgrid(data.x, data.y)

      if field == "velocity_magnitude":
          field_data = magnitude(data.u, data.v)
          title = "Velocity Magnitude"
      elif field == "vorticity":
          field_data = vorticity(data.u, data.v, data.dx, data.dy)
          title = "Vorticity"
      elif field == "u":
          field_data = data.u
          title = "U Velocity"
      elif field == "v":
          field_data = data.v
          title = "V Velocity"
      else:
          raise ValueError(f"Unknown field: {field}")

      return plot_contour_field(X, Y, field_data, title=title, **kwargs)
  ```

- [x] **1.3 Export integration utilities in package**
  ```python
  # Update cfd_viz/__init__.py
  from .convert import from_cfd_python, to_cfd_python
  from .quick import quick_plot
  ```

- [x] **1.4 Add integration tests**

- [x] **1.5 Document integration in examples**

### Success Criteria

- Simulation results can be directly visualized with one function call
- VTK files can be loaded and converted to cfd_python format
- Integration is documented with examples

---

## Phase 2: Enhanced Jupyter Integration

**Priority:** P2 - Enhanced interactivity
**Estimated Effort:** 2-3 days

### Goals

- Rich display in Jupyter notebooks
- Interactive widgets for visualization parameters
- Live simulation visualization

### Tasks

- [ ] **2.1 Add `_repr_html_` for VTKData**
  ```python
  class VTKData:
      def _repr_html_(self) -> str:
          import io
          import base64
          import matplotlib.pyplot as plt

          fig, ax = plt.subplots(figsize=(6, 4))
          speed = np.sqrt(self.u**2 + self.v**2)
          im = ax.contourf(self.x, self.y, speed, levels=20, cmap='viridis')
          ax.set_xlabel('x')
          ax.set_ylabel('y')
          ax.set_title('Velocity Magnitude')
          plt.colorbar(im, ax=ax)

          buf = io.BytesIO()
          fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
          plt.close(fig)
          buf.seek(0)
          img_str = base64.b64encode(buf.read()).decode()

          return f"""
          <div style="border: 1px solid #ccc; padding: 10px; display: inline-block;">
              <b>VTKData</b>: {self.u.shape[1]} × {self.u.shape[0]}<br>
              <img src="data:image/png;base64,{img_str}" />
          </div>
          """
  ```

- [ ] **2.2 Interactive visualization widgets**
  ```python
  # cfd_viz/jupyter.py
  """Jupyter notebook integration."""

  def interactive_contour(data: VTKData):
      """Create interactive contour plot with field selector."""
      from ipywidgets import interact, Dropdown, FloatSlider
      import matplotlib.pyplot as plt

      @interact(
          field=Dropdown(options=['velocity', 'vorticity', 'u', 'v', 'p']),
          levels=FloatSlider(min=10, max=50, value=20),
          colormap=Dropdown(options=['viridis', 'jet', 'coolwarm', 'plasma'])
      )
      def plot(field, levels, colormap):
          # Plot logic here
          ...
  ```

- [ ] **2.3 Live animation display**
  ```python
  def animate_in_notebook(
      vtk_files: list,
      field: str = "velocity_magnitude",
      interval: int = 100
  ):
      """Display animation inline in Jupyter notebook."""
      from IPython.display import HTML
      from matplotlib.animation import FuncAnimation

      # Create animation
      ...
      return HTML(anim.to_jshtml())
  ```

- [ ] **2.4 Add ipywidgets optional dependency**
  ```toml
  [project.optional-dependencies]
  jupyter = ["ipywidgets>=8.0"]
  ```

### Success Criteria

- VTKData displays preview in Jupyter automatically
- Interactive widgets work for parameter exploration
- Animations render smoothly inline

---

## Phase 3: Advanced Plot Types

**Priority:** P2 - Enhanced visualization options
**Estimated Effort:** 3-4 days

### Goals

- Add more visualization types for CFD data
- Support publication-quality figures
- Add comparison and overlay plots

### Tasks

- [ ] **3.1 Add pressure-velocity coupling visualization**
  ```python
  def plot_pressure_velocity(data: VTKData, **kwargs):
      """Plot pressure contours with velocity vectors overlaid."""
      ...
  ```

- [ ] **3.2 Add boundary layer profile plots**
  ```python
  def plot_boundary_layer_profile(
      data: VTKData,
      x_location: float,
      wall: str = "bottom"
  ):
      """Plot velocity profile at specified x-location."""
      ...
  ```

- [ ] **3.3 Add Reynolds stress visualization**
  ```python
  def plot_reynolds_stress(
      u_frames: list,
      v_frames: list,
      nx: int, ny: int
  ):
      """Compute and plot Reynolds stress from time series."""
      ...
  ```

- [ ] **3.4 Add Q-criterion isosurface (for 3D)**
  ```python
  def plot_q_criterion(data: VTKData3D, threshold: float = 0.1):
      """Plot Q-criterion isosurface for vortex identification."""
      ...
  ```

- [ ] **3.5 Add comparison subplots**
  ```python
  def plot_comparison(
      datasets: list[VTKData],
      labels: list[str],
      field: str = "velocity_magnitude",
      layout: str = "horizontal"  # or "grid"
  ):
      """Create side-by-side comparison plots."""
      ...
  ```

### Success Criteria

- New plot types are documented with examples
- Plots are publication-quality with proper labels
- Comparison plots work for multiple datasets

---

## Phase 4: 3D Visualization Support

**Priority:** P2 - Extends to 3D simulations
**Estimated Effort:** 4-5 days

### Goals

- Support 3D VTK data
- Add 3D plotting capabilities
- Enable slice and isosurface visualization

### Tasks

- [ ] **4.1 Create VTKData3D class**
  ```python
  @dataclass
  class VTKData3D:
      """Container for 3D CFD field data."""
      u: np.ndarray  # Shape: (nz, ny, nx)
      v: np.ndarray
      w: np.ndarray
      p: np.ndarray | None
      x: np.ndarray
      y: np.ndarray
      z: np.ndarray
      dx: float
      dy: float
      dz: float
  ```

- [ ] **4.2 Add 3D VTK file reading**
  ```python
  def read_vtk_3d(filename: str) -> VTKData3D:
      """Read 3D structured VTK file."""
      ...
  ```

- [ ] **4.3 Add slice extraction**
  ```python
  def extract_slice(
      data: VTKData3D,
      plane: str = "xy",  # "xy", "xz", "yz"
      location: float = 0.5  # Normalized position
  ) -> VTKData:
      """Extract 2D slice from 3D data."""
      ...
  ```

- [ ] **4.4 Add PyVista/VTK 3D visualization**
  ```python
  def plot_3d_isosurface(
      data: VTKData3D,
      field: str,
      isovalues: list[float]
  ):
      """Plot 3D isosurfaces using PyVista."""
      import pyvista as pv
      ...
  ```

- [ ] **4.5 Add 3D streamline tracing**
  ```python
  def plot_3d_streamlines(
      data: VTKData3D,
      seed_points: np.ndarray
  ):
      """Plot 3D streamlines."""
      ...
  ```

### Success Criteria

- 3D VTK files can be loaded
- Slices can be extracted for 2D visualization
- 3D isosurfaces and streamlines render correctly

---

## Phase 5: Animation Enhancements

**Priority:** P3 - Improved animation capabilities
**Estimated Effort:** 2-3 days

### Goals

- Improve animation quality and performance
- Add more export formats
- Support animation scripting

### Tasks

- [ ] **5.1 Add MP4 export with codec options**
  ```python
  def export_animation(
      frames: list[VTKData],
      output: str,
      fps: int = 30,
      codec: str = "h264",  # or "hevc", "prores"
      quality: str = "high"  # "low", "medium", "high"
  ):
      """Export animation with codec control."""
      ...
  ```

- [ ] **5.2 Add parallel frame rendering**
  ```python
  def render_frames_parallel(
      vtk_files: list[str],
      output_dir: str,
      n_workers: int = 4,
      **plot_kwargs
  ):
      """Render frames in parallel for faster animation creation."""
      from concurrent.futures import ProcessPoolExecutor
      ...
  ```

- [ ] **5.3 Add animation scripting**
  ```python
  class AnimationScript:
      """Script for complex animations with camera moves, etc."""

      def __init__(self, frames: list[VTKData]):
          self.frames = frames
          self.keyframes = []

      def add_camera_move(self, start_frame: int, end_frame: int,
                          start_view: dict, end_view: dict):
          """Add camera movement keyframe."""
          ...

      def add_annotation(self, frame: int, text: str, position: tuple):
          """Add text annotation at specific frame."""
          ...

      def render(self, output: str):
          """Render the scripted animation."""
          ...
  ```

- [ ] **5.4 Add time annotations and scale bars**

### Success Criteria

- Animations export quickly with parallel rendering
- Multiple codecs supported
- Scripted animations work correctly

---

## Phase 6: Interactive Dashboards

**Priority:** P3 - Web-based visualization
**Estimated Effort:** 3-4 days

### Goals

- Enhance Plotly dashboard capabilities
- Add real-time simulation monitoring
- Support web deployment

### Tasks

- [ ] **6.1 Enhance interactive dashboard**
  ```python
  def create_dashboard(
      data: VTKData | list[VTKData],
      port: int = 8050
  ):
      """Create comprehensive Dash-based visualization dashboard."""
      import dash
      from dash import dcc, html
      ...
  ```

- [ ] **6.2 Add real-time monitoring**
  ```python
  class SimulationMonitor:
      """Real-time dashboard for running simulations."""

      def __init__(self, output_dir: str, refresh_interval: int = 1000):
          self.output_dir = output_dir
          self.refresh_interval = refresh_interval

      def start(self):
          """Start monitoring and display dashboard."""
          ...
  ```

- [ ] **6.3 Add comparison dashboard**
  ```python
  def create_comparison_dashboard(
      datasets: dict[str, list[VTKData]],
      labels: list[str]
  ):
      """Dashboard for comparing multiple simulation cases."""
      ...
  ```

- [ ] **6.4 Add export to standalone HTML**
  ```python
  def export_interactive_html(
      data: VTKData,
      output: str,
      include_controls: bool = True
  ):
      """Export interactive visualization as standalone HTML file."""
      ...
  ```

### Success Criteria

- Dashboard runs locally with live updates
- Multiple cases can be compared interactively
- HTML export works standalone without server

---

## Phase 7: Analysis Enhancements

**Priority:** P2 - Enhanced analysis tools
**Estimated Effort:** 3-4 days

### Goals

- Add more analysis capabilities
- Support statistical analysis of time series
- Add validation tools

### Tasks

- [ ] **7.1 Add time-averaging**
  ```python
  def time_average(frames: list[VTKData]) -> VTKData:
      """Compute time-averaged fields."""
      ...

  def fluctuation_fields(
      frames: list[VTKData],
      mean: VTKData | None = None
  ) -> list[VTKData]:
      """Compute fluctuation fields u' = u - <u>."""
      ...
  ```

- [ ] **7.2 Add spectral analysis**
  ```python
  def compute_spectrum(
      frames: list[VTKData],
      probe_location: tuple[float, float]
  ) -> dict:
      """Compute frequency spectrum at probe location."""
      from scipy.fft import fft, fftfreq
      ...
  ```

- [ ] **7.3 Add drag/lift coefficient calculation**
  ```python
  def compute_force_coefficients(
      data: VTKData,
      body_contour: np.ndarray,
      rho: float = 1.0,
      u_inf: float = 1.0
  ) -> dict:
      """Compute drag and lift coefficients."""
      ...
  ```

- [ ] **7.4 Add Ghia et al. reference data**
  ```python
  # cfd_viz/validation/__init__.py
  """Reference data for validation."""

  GHIA_LID_DRIVEN_CAVITY = {
      100: {
          'y': [...],
          'u_centerline': [...],
          'x': [...],
          'v_centerline': [...]
      },
      400: {...},
      1000: {...}
  }

  def compare_to_ghia(data: VTKData, Re: int) -> dict:
      """Compare simulation to Ghia et al. (1982) reference."""
      ...
  ```

- [ ] **7.5 Add error metrics**
  ```python
  def compute_error_metrics(
      computed: VTKData,
      reference: VTKData | np.ndarray
  ) -> dict:
      """Compute L1, L2, Linf error norms."""
      ...
  ```

### Success Criteria

- Time-series analysis works correctly
- Reference data comparisons are easy
- Error metrics computed accurately

---

## Version Planning

| Version | Phases | Focus | Status |
| ------- | ------ | ----- | ------ |
| 0.2.0 | 1, 8 | cfd-python integration (basic + v0.1.6 features) | Ready for release |
| 0.3.0 | 2, 3 | Jupyter & advanced plots | Planned |
| 0.4.0 | 4 | 3D visualization | Planned |
| 0.5.0 | 5, 6 | Animation & dashboards | Planned |
| 0.6.0 | 7 | Analysis enhancements | Planned |

---

## Phase 8: Enhanced cfd-python v0.1.6 Integration ✅ COMPLETE

**Priority:** P1 - Leverage new cfd-python features
**Estimated Effort:** 2-3 days
**Status:** ✅ Completed (2026-01-22)

### Background

cfd-python v0.1.6 added significant new capabilities that cfd-visualization can leverage:

- **Backend Availability API**: Query SIMD/OpenMP/CUDA backends at runtime
- **Derived Fields**: `calculate_field_stats()`, `compute_velocity_magnitude()`, `compute_flow_statistics()`
- **Error Handling**: Python exception hierarchy with `raise_for_status()`
- **CPU Features Detection**: `has_avx2()`, `has_neon()`, `get_simd_name()`

### Tasks

- [x] **8.1 Use cfd-python's `calculate_field_stats()` for consistency**
  ```python
  # Instead of recomputing statistics, use cfd-python's optimized version
  def get_field_stats(data: VTKData, field: str = "velocity_magnitude") -> dict:
      """Get field statistics using cfd-python's optimized function."""
      try:
          import cfd_python
          flat_field = get_field_data(data, field).flatten().tolist()
          return cfd_python.calculate_field_stats(flat_field)
      except ImportError:
          # Fallback to NumPy if cfd-python not available
          return _compute_stats_numpy(data, field)
  ```

- [x] **8.2 Leverage `compute_flow_statistics()` for comprehensive analysis**
  ```python
  def analyze_flow(data: VTKData) -> dict:
      """Comprehensive flow analysis using cfd-python backend."""
      import cfd_python

      u_flat = data.u.flatten().tolist()
      v_flat = data.v.flatten().tolist()
      p_flat = data.p.flatten().tolist() if data.p is not None else [0.0] * len(u_flat)

      return cfd_python.compute_flow_statistics(
          u_flat, v_flat, p_flat,
          data.u.shape[1], data.u.shape[0]
      )
  ```

- [x] **8.3 Add backend-aware performance hints**
  ```python
  def get_recommended_settings() -> dict:
      """Get recommended visualization settings based on available backends."""
      import cfd_python

      backends = cfd_python.get_available_backends()
      simd = cfd_python.get_simd_name()

      return {
          'parallel_rendering': 'OpenMP' in backends,
          'simd_available': simd != 'none',
          'gpu_acceleration': 'CUDA' in backends,
          'recommended_chunk_size': 1024 if 'avx2' in simd else 256,
      }
  ```

- [ ] **8.4 Add Rankine vortex visualization (from cfd-python examples)**
  ```python
  def create_rankine_vortex(
      nx: int, ny: int,
      core_radius: float = 0.3,
      strength: float = 1.0,
      xmin: float = 0.0, xmax: float = 1.0,
      ymin: float = 0.0, ymax: float = 1.0
  ) -> VTKData:
      """Create synthetic Rankine vortex for testing and demonstration.

      The Rankine vortex combines:
      - Solid body rotation inside core (v ~ r)
      - Irrotational flow outside core (v ~ 1/r)

      This produces the characteristic "volcano" shape in 3D velocity plots.
      """
      ...

  def plot_rankine_vortex_3d(
      data: VTKData,
      field: str = "velocity_magnitude"
  ):
      """Create 3D surface plot of velocity field (Rankine vortex visualization)."""
      from mpl_toolkits.mplot3d import Axes3D
      ...
  ```

- [ ] **8.5 Add physics-based plot annotations**
  ```python
  def add_physics_annotations(
      ax,
      data: VTKData,
      flow_type: str = "cavity"  # "cavity", "channel", "vortex"
  ):
      """Add physics-relevant annotations to plots.

      For cavity flow: Mark primary vortex center, corner vortices
      For channel flow: Add analytical Poiseuille profile comparison
      For vortex: Mark core radius, maximum velocity ring
      """
      ...
  ```

- [ ] **8.6 Add convergence visualization utilities**
  ```python
  def plot_convergence_study(
      step_counts: list[int],
      max_velocities: list[float],
      avg_velocities: list[float],
      threshold: float = 0.01
  ):
      """Plot convergence analysis from cfd-python simulation runs.

      Shows:
      - Velocity vs simulation steps
      - Relative change (convergence indicator)
      - Threshold line for convergence criterion
      """
      ...
  ```

### Success Criteria

- Statistics computed via cfd-python when available (faster, consistent)
- Rankine vortex can be created and visualized with 3D surface plots
- Convergence plots match cfd-python example output
- Backend availability influences visualization recommendations

---

## Ideas Backlog

Items not yet planned but worth considering:

### Visualization
- [ ] ParaView export support
- [ ] VDB volume format for 3D
- [ ] GPU-accelerated rendering (leverage cfd-python CUDA backend)
- [ ] WebGL-based web viewer
- [ ] Virtual reality (VR) visualization
- [ ] 3D velocity surface plots (volcano/Rankine vortex style)

### Analysis
- [ ] Proper Orthogonal Decomposition (POD)
- [ ] Dynamic Mode Decomposition (DMD)
- [ ] Machine learning feature extraction
- [ ] Automatic vortex detection
- [ ] Coherent structure identification
- [ ] Analytical solution comparisons (Poiseuille, Blasius, etc.)

### Integration
- [ ] OpenFOAM result import
- [ ] CGNS format support
- [ ] HDF5 parallel I/O
- [ ] ParaView Catalyst live coupling
- [ ] Direct cfd-python simulation callback hooks

### Tooling
- [ ] VS Code preview extension
- [ ] CLI for batch visualization
- [ ] Docker image with all dependencies
- [ ] Backend detection and optimization hints

---

## Related Documents

- [README.md](README.md) - User documentation
- [cfd-python ROADMAP](../cfd-python/ROADMAP.md) - Simulation library roadmap

---

## Contributing

Contributions are welcome! Priority areas:

1. **3D visualization** - PyVista integration
2. **Animation** - Performance improvements
3. **Analysis** - Additional flow metrics
4. **Documentation** - Examples and tutorials
5. **Testing** - Visualization regression tests
