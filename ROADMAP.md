# CFD-Visualization Roadmap

This document outlines planned enhancements for cfd-visualization, organized by priority and release target.

**Last Updated:** 2026-03-05
**Current Version:** 0.1.0

---

## Current Capabilities

- **VTK I/O**: `read_vtk_file()`, `VTKData` container class
- **Field computations**: Vorticity, gradients, magnitude, Q-criterion, lambda2, derived fields
- **Plotting**: Contours, vectors, streamlines, quiver plots, convergence history
- **Analysis**: Boundary layer, line extraction, case comparison, flow features, time series
- **Animation**: Frame generation, GIF/MP4 export
- **Interactive**: Plotly dashboards and heatmaps
- **cfd-python integration**: Conversion utilities, quick plotting, statistics, system info

---

## Phase 1: cfd-python Integration - COMPLETE

**Status:** Completed (2026-01-22)

- [x] Conversion utilities (`cfd_viz/convert.py`) - `from_cfd_python()`, `from_simulation_result()`, `to_cfd_python()`
- [x] Quick plotting wrapper (`cfd_viz/quick.py`) - `quick_plot()`, `quick_plot_result()`, `quick_plot_data()`
- [x] Statistics with cfd-python acceleration (`cfd_viz/stats.py`)
- [x] Backend-aware system info (`cfd_viz/info.py`)
- [x] Integration tests and examples

---

## Phase 2: Foundation & Code Quality

**Priority:** P0 - Must-have before adding features
**Target:** v0.2.0

The VTK reader (`cfd_viz/common/vtk_reader.py`) is 240 lines with zero dedicated tests. VTKData accepts any data without validation. The version string is duplicated between `pyproject.toml` and `__init__.py`. These are trust issues for a library used in research.

### Tasks

- [ ] **2.1 Add pytest configuration** to `pyproject.toml` - testpaths, markers, filterwarnings
- [ ] **2.2 Write VTK reader tests** - STRUCTURED_POINTS/RECTILINEAR_GRID parsing, malformed files, edge cases (empty fields, single-row grids)
- [ ] **2.3 Add VTKData validation** - verify field shapes match `(ny, nx)` on construction, warn on NaN/inf, add `__repr__` for debugging
- [ ] **2.4 Normalize field naming** - add `FIELD_ALIASES` mapping so `data["pressure"]` and `data["p"]` both work; standardize across all modules
- [ ] **2.5 Single-source version** - use hatchling dynamic version or `importlib.metadata`; remove the duplication
- [ ] **2.6 Add integration test** - end-to-end: create VTK data in memory, read it, compute vorticity, plot to figure, assert no exceptions

### Success Criteria

- VTK reader has dedicated test coverage
- VTKData raises `ValueError` for mismatched field shapes
- Version appears in exactly one source file

---

## Phase 3: Global Configuration System

**Priority:** P1 - Biggest daily-use friction point
**Target:** v0.2.0

Every plotting function has hardcoded defaults (`cmap="viridis"`, `figsize=(8,6)`, `dpi=150`). A researcher who prefers a specific colormap for publications must pass it to every single call. There are 18+ hardcoded colormap strings across plotting modules.

### Tasks

- [ ] **3.1 Create `PlotDefaults` dataclass** - `cmap`, `diverging_cmap`, `figsize`, `dpi`, `levels`, `font_size`. Provide `get_defaults()` / `set_defaults()` functions
- [ ] **3.2 Add context manager** - `with cfd_viz.plot_context(cmap="coolwarm", dpi=300):` for temporary overrides
- [ ] **3.3 Support config file** - read `[tool.cfd_viz]` from pyproject.toml or standalone `cfd_viz.toml` for project-level defaults
- [ ] **3.4 Migrate plotting functions** - replace hardcoded defaults with `get_defaults()` calls across all plotting modules
- [ ] **3.5 Add configuration tests** - default override, context manager nesting, config file loading

### Success Criteria

- `cfd_viz.set_defaults(cmap="coolwarm")` changes all subsequent plots
- Context manager correctly restores previous defaults
- All existing examples and tests pass without changes

---

## Phase 4: CLI Tooling & Entry Points

**Priority:** P1 - Registers existing orphaned code
**Target:** v0.3.0

Five scripts exist in `scripts/` but are not registered as package entry points. Users must run `python scripts/create_animation.py` instead of a proper CLI command.

### Tasks

- [ ] **4.1 Register entry points** in `pyproject.toml` - `[project.scripts]` section
- [ ] **4.2 Create unified `cfd-viz` CLI** - single entry point with argparse subcommands dispatching to existing scripts
- [ ] **4.3 Add batch processing** - `cfd-viz batch --config batch.toml` for processing multiple VTK files in one run
- [ ] **4.4 Add `cfd-viz info`** - wraps existing `print_system_info()`, shows backends and optional deps
- [ ] **4.5 Add progress indicators** - simple stderr output for batch operations and animation rendering (no extra dependency)

### Success Criteria

- `pip install -e .` makes `cfd-viz` command available
- `cfd-viz info` prints system capabilities
- `cfd-viz batch` processes multiple files from config

---

## Phase 5: Jupyter Integration

**Priority:** P2 - Enhanced interactivity
**Target:** v0.3.0

### Tasks

- [ ] **5.1 Add `_repr_html_` to VTKData** - grid dimensions, available fields, field statistics in a formatted HTML table (no extra dependency)
- [ ] **5.2 Create `cfd_viz/jupyter.py`** - `explore_field(data)` with ipywidgets sliders for colormap, levels, field selection
- [ ] **5.3 Inline animation display** - return HTML5 video via `IPython.display.HTML`
- [ ] **5.4 Add ipywidgets optional dependency** - `jupyter = ["ipywidgets>=8.0"]`

### Success Criteria

- VTKData shows a summary table when displayed in Jupyter
- `explore_field(data)` creates interactive widgets
- Animations play inline in notebooks

---

## Phase 6: Advanced Plots & Publication Quality

**Priority:** P2 - Enhanced visualization for researchers
**Target:** v0.4.0

### Tasks

- [ ] **6.1 Comparison subplot layouts** - `plot_comparison(data_list, fields, layout="2x2")` for side-by-side multi-case parameter studies
- [ ] **6.2 Convergence visualization** - plot velocity vs iteration steps with relative change and threshold lines; accept cfd-python convergence data directly
- [ ] **6.3 Rankine vortex visualization** - synthetic Rankine vortex for testing/demos with 3D surface plot
- [ ] **6.4 Physics-based auto-annotations** - detect flow type (cavity, channel, vortex) and annotate with Re, max velocity, recirculation zones
- [ ] **6.5 Publication export helper** - `save_publication(fig, "figure1", formats=["pdf", "svg", "png"], dpi=600)` with journal column-width presets
- [ ] **6.6 Pressure-velocity coupling plot** - overlaid pressure contours with velocity vectors

### Success Criteria

- Comparison layouts work for 2-6 cases
- Publication helper produces journal-ready figures
- Auto-annotations correctly identify flow features

---

## Phase 7: Analysis Enhancements & Validation Data

**Priority:** P2 - Solver validation and data manipulation
**Target:** v0.4.0

### Tasks

- [ ] **7.1 Ghia et al. reference data** - centerline velocity profiles for Re=100, 400, 1000, 3200, 5000, 10000
- [ ] **7.2 Analytical solutions** - Poiseuille profile, Blasius boundary layer, Taylor-Green vortex decay for comparison plotting
- [ ] **7.3 Error norm computation** - L1, L2, Linf between computed and reference data
- [ ] **7.4 Data slicing API** - `data.slice(x=0.5)` returns LineProfile, `data.subset(xmin=0.2, xmax=0.8)` returns cropped VTKData
- [ ] **7.5 Field caching** - derived fields (vorticity, Q-criterion) cached on VTKData `_cache` dict so repeated calls skip recomputation

### Success Criteria

- `ghia_data(Re=1000)` returns arrays matching the published paper
- Error norms can be plotted against grid resolution for convergence studies
- `data.slice(x=0.5)` integrates with existing line plotting functions

---

## Phase 8: cfd-python v0.2.0+ Alignment

**Priority:** P1 - Leverage upcoming cfd-python features
**Target:** v0.5.0

cfd-python Phase 10 introduces high-level classes (`Grid`, `Simulation`, `SimulationResult`). Phase 11 adds NumPy array support (eliminating list-to-array conversion overhead). Phase 12 explicitly targets cfd-visualization integration.

### Tasks

- [ ] **8.1 Adapter for `SimulationResult` class** - duck-typed conversion accepting any object with `.u`, `.v`, `.p`, `.grid` attributes
- [ ] **8.2 Adapter for `Grid` class** - convert to cfd-viz coordinate arrays
- [ ] **8.3 Zero-copy NumPy path** - detect ndarray input in `convert.py` and use directly instead of `np.array(list)`
- [ ] **8.4 Validation visualization** - plotting functions for Ghia comparison, Poiseuille profile overlay, Taylor-Green energy decay (uses Phase 7 reference data)
- [ ] **8.5 Performance benchmark visualization** - plot backend comparison results: grid-size vs time, speedup ratios
- [ ] **8.6 Bump minimum cfd-python version** - update optional dependency to `>=0.2.0` with 0.1.6 fallback

### Success Criteria

- `from_simulation(result)` works for both dict-based (v0.1.6) and class-based (v0.2.0+) results
- NumPy arrays pass through without list conversion
- Validation plots match expected reference comparisons

---

## Phase 9: Animation & Interactive Enhancements

**Priority:** P3 - Dynamic visualization
**Target:** v0.6.0

### Tasks

- [ ] **9.1 Time annotations on animation frames** - step number, physical time, scale bar overlay
- [ ] **9.2 Parallel frame rendering** - `ProcessPoolExecutor` for independent frame generation
- [ ] **9.3 MP4 export** - ffmpeg-based output with codec and quality options
- [ ] **9.4 Enhanced Plotly dashboard** - field selector dropdown, time slider, case comparison tabs
- [ ] **9.5 Standalone HTML export** - self-contained HTML file with embedded data, no running server required
- [ ] **9.6 Real-time simulation monitoring** - integrate with cfd-python callbacks; build on existing `scripts/create_monitor.py`

### Success Criteria

- Animations include configurable time annotations
- Frame rendering is 2-4x faster with parallelism
- HTML export works standalone without a server

---

## Phase 10: 3D Visualization

**Priority:** P3 - Depends on cfd-python Phase 15 (3D grid support)
**Target:** v1.0.0

3D visualization adds a heavy optional dependency (PyVista) that most 2D users don't need. Deliberately last, aligned with cfd-python's 3D grid support timeline.

### Tasks

- [ ] **10.1 Create VTKData3D class** - extends data model with Z coordinates and w-velocity
- [ ] **10.2 Add 3D VTK file reading** - STRUCTURED_POINTS/RECTILINEAR_GRID with nz > 1
- [ ] **10.3 Slice extraction** - `data3d.slice(z=0.5)` returns 2D VTKData compatible with all existing functions
- [ ] **10.4 PyVista integration** - optional dependency `viz3d = ["pyvista>=0.40"]`, `to_pyvista(data3d)` conversion
- [ ] **10.5 3D streamline visualization** - using PyVista

### Success Criteria

- 3D VTK files load correctly
- 2D slices work with all existing plotting functions
- PyVista renders isosurfaces and streamlines

---

## Version Planning

| Version | Phases | Focus |
| ------- | ------ | ----- |
| 0.2.0 | 2, 3 | Foundation, configuration, testing |
| 0.3.0 | 4, 5 | CLI tooling, Jupyter integration |
| 0.4.0 | 6, 7 | Advanced plots, validation data, data slicing |
| 0.5.0 | 8 | cfd-python v0.2.0 alignment |
| 0.6.0 | 9 | Animation & interactive enhancements |
| 1.0.0 | 10 | 3D visualization, API stabilization |

---

## Ideas Backlog

Items worth considering but not yet scheduled:

### Visualization

- [ ] ParaView export (`.pvd` series files)
- [ ] WebGL-based standalone viewer
- [ ] GPU-accelerated rendering via CUDA interop

### Analysis

- [ ] Proper Orthogonal Decomposition (POD)
- [ ] Dynamic Mode Decomposition (DMD)
- [ ] Automatic vortex tracking across time series
- [ ] Analytical solution library expansion (Couette, Stokes, Lamb-Oseen)

### I/O & Interoperability

- [ ] OpenFOAM result import
- [ ] HDF5 field storage for large datasets
- [ ] CGNS format support

### Tooling

- [ ] Sphinx documentation site with API reference
- [ ] Docker image with all dependencies

---

## Related Documents

- [README.md](README.md) - User documentation
- [cfd-python ROADMAP](../cfd-python/ROADMAP.md) - Simulation library roadmap (Phase 12 targets cfd-viz integration)

---

## Contributing

Contributions welcome! Priority areas:

1. **Testing** - VTK reader and integration tests (Phase 2)
2. **Configuration** - Global defaults system (Phase 3)
3. **Analysis** - Validation data and error metrics (Phase 7)
4. **3D visualization** - PyVista integration (Phase 10)
5. **Documentation** - Examples and tutorials
