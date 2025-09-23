# CFD Visualization Framework - Source Code Organization

This directory contains the organized source code for the CFD visualization framework, categorized by functionality for better maintainability and ease of use.

## Directory Structure

```
src/
├── analysis/        # High-priority CFD analysis tools
├── animation/       # Animation and flow visualization tools
├── interactive/     # Interactive and advanced tools
├── utilities/       # Core utilities and support functions
└── __init__.py     # Main package initialization
```

## 📊 Analysis Tools (`src/analysis/`)

**Purpose:** Essential CFD flow analysis and quantitative assessment tools.

### High-Priority Analysis Tools:

| Tool | Purpose | Key Features |
|------|---------|--------------|
| `vorticity_visualizer.py` | Vorticity and circulation analysis | Vorticity fields, Q-criterion, vortex core detection, circulation calculations |
| `cross_section_analyzer.py` | Line plots and boundary layer analysis | Multi-line analysis, boundary layer profiling, interactive selection |
| `parameter_study.py` | Parameter comparison and optimization | Side-by-side comparisons, parameter sweeps, statistical analysis |
| `realtime_monitor.py` | Live simulation monitoring | Real-time dashboards, convergence tracking, file system monitoring |

**Usage Patterns:**
```bash
# Fundamental flow physics analysis
python src/analysis/vorticity_visualizer.py --latest

# Quantitative flow profiling
python src/analysis/cross_section_analyzer.py --latest --interactive

# Parameter optimization studies
python src/analysis/parameter_study.py --sweep Re --pattern "*Re*.vtk"

# Real-time monitoring
python src/analysis/realtime_monitor.py --watch_dir ../../output/vtk_files
```

## 🎬 Animation Tools (`src/animation/`)

**Purpose:** Creating dynamic visualizations and animations from CFD data.

### Animation and Flow Visualization:

| Tool | Purpose | Key Features |
|------|---------|--------------|
| `animate_flow.py` | Advanced flow animation system | Multi-frame processing, customizable rendering, multiple export formats |
| `create_cfd_animation.py` | Specialized CFD animation generator | VTK time series processing, multi-variable animation, synchronized playback |
| `create_simple_animation.py` | Quick animation utilities | Rapid prototyping, basic flow patterns, lightweight processing |
| `velocity_flow_viz.py` | Velocity field visualization | Vector field rendering, streamline generation, magnitude contours |
| `test_animation.py` | Animation testing and validation | Frame rate optimization, quality assessment, format compatibility |

**Usage Patterns:**
```bash
# Create comprehensive CFD animations
python src/animation/create_cfd_animation.py --input_dir ../../output/vtk_files --fps 10

# Advanced flow visualizations
python src/animation/velocity_flow_viz.py --latest --output flow_patterns

# Quick animation prototyping
python src/animation/create_simple_animation.py --pattern "*.vtk" --output quick_anim
```

## 🔄 Interactive Tools (`src/interactive/`)

**Purpose:** Advanced interactive analysis and web-based visualization platforms.

### Interactive and Advanced Tools:

| Tool | Purpose | Key Features |
|------|---------|--------------|
| `interactive_dashboard.py` | Web-based CFD analysis dashboard | Plotly integration, real-time interaction, multi-dataset support |
| `advanced_visualize.py` | Sophisticated flow analysis platform | Multi-physics visualization, statistical overlays, publication graphics |
| `enhanced_visualize.py` | Extended visualization capabilities | 3D rendering support, particle tracking, temporal analysis |

**Usage Patterns:**
```bash
# Launch interactive web dashboard
python src/interactive/interactive_dashboard.py --port 8080

# Advanced multi-physics visualization
python src/interactive/advanced_visualize.py --latest --mode publication

# 3D and enhanced visualizations
python src/interactive/enhanced_visualize.py --latest --render-3d
```

## 🔧 Utility Tools (`src/utilities/`)

**Purpose:** Core functionality, workflow automation, and supporting utilities.

### Core Utilities and Support:

| Tool | Purpose | Key Features |
|------|---------|--------------|
| `visualize_cfd.py` | Core CFD visualization library | VTK file processing, data structure management, coordinate systems |
| `simple_viz.py` | Rapid visualization utilities | One-line plotting, default configurations, batch processing |
| `run_visualization.py` | Master visualization controller | Workflow management, configuration system, parallel processing |

**Usage Patterns:**
```bash
# Core visualization functions
python src/utilities/visualize_cfd.py --file simulation.vtk --output basic_plots

# Simple batch visualization
python src/utilities/simple_viz.py --directory ../../output/vtk_files --batch

# Automated workflow execution
python src/utilities/run_visualization.py --config workflow.yaml
```

## Import Structure and Package Usage

### Python Package Imports
```python
# Import entire categories
from src.analysis import vorticity_visualizer, parameter_study
from src.animation import create_cfd_animation
from src.interactive import interactive_dashboard
from src.utilities import visualize_cfd

# Import specific functions
from src.analysis.vorticity_visualizer import calculate_vorticity
from src.animation.velocity_flow_viz import create_velocity_plot
from src.utilities.visualize_cfd import read_vtk_file
```

### Command Line Usage
```bash
# Direct script execution
python src/analysis/vorticity_visualizer.py --latest
python src/animation/create_cfd_animation.py --input_dir data/

# Module execution
python -m src.analysis.parameter_study --compare file1.vtk file2.vtk
python -m src.utilities.run_visualization --config analysis.yaml
```

## Development Guidelines

### Adding New Tools

**1. Choose the Appropriate Category:**
- **Analysis:** Tools for quantitative CFD analysis and flow physics
- **Animation:** Tools for creating dynamic visualizations
- **Interactive:** Tools with web interfaces or advanced user interaction
- **Utilities:** Core support functions and workflow automation

**2. Follow Naming Conventions:**
- Use descriptive names: `boundary_layer_analyzer.py`
- Include purpose in filename: `turbulence_statistics.py`
- Add appropriate docstrings and module documentation

**3. Update Package Files:**
- Add imports to category `__init__.py`
- Update this README with tool descriptions
- Add usage examples to main documentation

### Code Organization Best Practices

**1. Consistent Interface:**
```python
def main():
    parser = argparse.ArgumentParser(description='Tool description')
    parser.add_argument('--input', help='Input VTK file')
    parser.add_argument('--output', default='output', help='Output directory')
    # ... other arguments

    args = parser.parse_args()
    # Tool implementation
```

**2. Common Functions:**
- Use `utilities/visualize_cfd.py` for shared VTK reading functions
- Import common plotting functions from utilities
- Leverage existing data structures and validation

**3. Documentation Standards:**
- Include module-level docstrings with purpose and usage
- Document all functions with parameters and return values
- Provide command-line usage examples
- Include scientific background where relevant

## Performance and Dependencies

### Computational Requirements
- **Analysis tools:** Medium to high (complex calculations)
- **Animation tools:** High (multiple frame processing)
- **Interactive tools:** Medium (real-time updates)
- **Utilities:** Low to medium (basic processing)

### External Dependencies
- **Core:** NumPy, Matplotlib, SciPy
- **Advanced:** VTK, Plotly, Pandas
- **Interactive:** Plotly, Dash (for web interfaces)
- **Monitoring:** Watchdog (for file system events)

### Memory and Performance Optimization
- Use streaming for large datasets in animation tools
- Implement lazy loading in interactive tools
- Cache calculations in analysis tools
- Provide memory usage options for utilities

This organization ensures scalability, maintainability, and ease of use for the comprehensive CFD visualization framework.