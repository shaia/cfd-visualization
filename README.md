# CFD Visualization Library

A Python library for visualizing CFD (Computational Fluid Dynamics) simulation results. Designed to work with VTK output files from the CFD framework.

## Installation

### Prerequisites

- [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- [cfd-python](../cfd-python) package for running simulations

### Setup

```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate cfd-visualization

# Install package in development mode
pip install -e .
```

## Project Structure

```
cfd-visualization/
├── cfd_viz/                    # Main library package
│   ├── common/                 # VTK I/O and utilities
│   ├── fields/                 # Field computations (vorticity, magnitude, etc.)
│   ├── analysis/               # Flow analysis (profiles, metrics, features)
│   ├── plotting/               # Matplotlib plotting functions
│   ├── animation/              # Animation creation and export
│   └── interactive/            # Plotly interactive visualizations
├── examples/                   # Example scripts
├── scripts/                    # CLI tools
├── tests/                      # Test suite
└── data/                       # Sample data
```

## Examples

Run example scripts to see the library in action:

```bash
# Basic lid-driven cavity simulation with visualization
python examples/basic_simulation.py

# Channel flow (Poiseuille flow) with velocity profiles
python examples/channel_flow.py

# Cavity variations - compare different aspect ratios
python examples/cavity_variations.py

# Transient animation - flow evolution over time
python examples/transient_animation.py

# Interactive Plotly visualizations (outputs HTML files)
python examples/interactive_plotly.py
```

### Example Outputs

| Example | Description | Output |
|---------|-------------|--------|
| `basic_simulation.py` | Lid-driven cavity flow with 6-panel visualization | PNG |
| `channel_flow.py` | Poiseuille flow with analytical comparison | PNG |
| `cavity_variations.py` | Square, tall, and wide cavity comparison | PNG |
| `transient_animation.py` | Time evolution animations | PNG, GIF |
| `interactive_plotly.py` | Interactive web visualizations | HTML |

## CLI Scripts

Command-line tools for processing VTK files:

```bash
# Create animations from VTK files
python scripts/create_animation.py --type velocity output/*.vtk

# Create interactive Plotly dashboard
python scripts/create_dashboard.py --vtk-pattern "output/*.vtk"

# Create line profile analysis
python scripts/create_line_profiles.py --latest

# Monitor simulation in real-time
python scripts/create_monitor.py --watch_dir output/

# Create vorticity analysis
python scripts/create_vorticity_analysis.py --latest
```

## cfd-python Integration

cfd-visualization integrates seamlessly with [cfd-python](https://github.com/shaia/cfd-python) v0.1.6+ for direct visualization of simulation results.

### Installation with cfd-python

```bash
# Install with simulation support
pip install cfd-visualization[simulation]

# Or install everything
pip install cfd-visualization[full]
```

### Quick Visualization

```python
import cfd_python
from cfd_viz import quick_plot_result

# Run simulation
result = cfd_python.run_simulation_with_params(nx=64, ny=64, steps=500)

# One-liner visualization
fig, ax = quick_plot_result(result, field="velocity_magnitude")
```

### Converting Simulation Results

```python
from cfd_viz import from_simulation_result, from_cfd_python

# From result dictionary
data = from_simulation_result(result)

# Or from individual arrays
data = from_cfd_python(
    u=result['u'], v=result['v'], p=result['p'],
    nx=result['nx'], ny=result['ny']
)

# Now use any cfd_viz function
from cfd_viz.fields import vorticity
omega = vorticity(data.u, data.v, data.dx, data.dy)
```

### Statistics with cfd-python Backend

```python
from cfd_viz import calculate_field_stats, compute_flow_statistics

# Uses cfd-python's optimized implementation when available
stats = compute_flow_statistics(data)
print(f"Max velocity: {stats['velocity_magnitude']['max']:.4f}")
```

### System Information

```python
from cfd_viz import print_system_info

print_system_info()
# Output:
# cfd-visualization System Info
# ========================================
# cfd-python available: True
# cfd-python version: 0.1.6
# Available backends: Scalar, SIMD, OpenMP
# Has AVX2: True
# Has NEON: False
# SIMD backend: avx2
# GPU available: False
#
# Recommended Settings
# --------------------
# backend: SIMD
# simd_backend: avx2
# use_gpu: False
```

---

## Library API

### Reading VTK Files

```python
from cfd_viz.common import read_vtk_file

data = read_vtk_file("simulation.vtk")
X, Y = data.X, data.Y  # Meshgrid coordinates
u, v = data.u, data.v  # Velocity components
p = data.get("p")      # Pressure (if available)
```

### Computing Derived Fields

```python
from cfd_viz.fields import magnitude, vorticity, divergence

vel_mag = magnitude(u, v)
omega = vorticity(u, v, dx, dy)
div = divergence(u, v, dx, dy)
```

### Plotting

```python
from cfd_viz.plotting import (
    plot_velocity_field,
    plot_streamlines,
    plot_vorticity_field,
)

fig, ax = plt.subplots()
plot_velocity_field(X, Y, u, v, ax=ax, title="Velocity")
plot_streamlines(X, Y, u, v, ax=ax)
```

### Creating Animations

```python
from cfd_viz.animation import (
    create_animation_frames,
    create_field_animation,
    save_animation,
)

frames = create_animation_frames(frames_data, time_indices=times)
fig, anim = create_field_animation(frames, field_name="velocity_mag")
save_animation(anim, "flow.gif", fps=5)
```

### Interactive Visualizations

```python
from cfd_viz.interactive import (
    create_interactive_frame,
    create_dashboard_figure,
    create_heatmap_figure,
)

frame = create_interactive_frame(x, y, u, v, p=p)
fig = create_dashboard_figure(frame, title="CFD Dashboard")
fig.write_html("dashboard.html")
```

## Running Tests

```bash
pytest tests/ -v
```

## Dependencies

- numpy
- matplotlib
- scipy
- plotly
- pandas
- imageio (for GIF export)

## License

MIT License
