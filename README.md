# CFD Visualization Tools

This project contains visualization tools and utilities for the CFD (Computational Fluid Dynamics) framework.

## Environment Setup

This project uses Anaconda/Conda for dependency management and virtual environment isolation.

### Prerequisites

- [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed
- Git (for cloning the repository)

### Quick Start

1. **Navigate to the project directory:**
   ```bash
   cd cfd-visualization
   ```

2. **Create the conda environment:**
   ```bash
   conda env create -f environment.yml
   ```

3. **Activate the environment:**
   ```bash
   conda activate cfd-visualization
   ```

4. **Install the package in development mode:**
   ```bash
   pip install -e .
   ```

### Alternative Manual Setup

If you prefer to create the environment manually:

```bash
# Create a new conda environment
conda create -n cfd-visualization python=3.10

# Activate the environment
conda activate cfd-visualization

# Install dependencies
conda install -c conda-forge numpy matplotlib scipy pandas seaborn imageio tqdm jupyter ipykernel
pip install vtk plotly

# Install the package
pip install -e .
```

## Project Structure

```
cfd-visualization/
├── environment.yml       # Conda environment specification
├── requirements.txt      # Pip requirements (backup)
├── setup.py             # Python package configuration
├── src/                 # Python visualization scripts
│   ├── analysis/        # High-priority CFD analysis tools
│   ├── animation/       # Animation and flow visualization
│   ├── interactive/     # Interactive and advanced tools
│   └── utilities/       # Core utilities and support tools
├── data/                # Test data and animation frames
└── docs/                # Documentation
```

## Python Visualization Scripts

### High-Priority Analysis Tools

#### 🌀 Vorticity Visualizer (`src/analysis/vorticity_visualizer.py`)

**Advanced flow physics analysis for understanding rotational flow structures.**

**Core Capabilities:**
- **Vorticity Field Calculation:** Computes ∇ × v (curl of velocity) using central differences
- **Q-Criterion Analysis:** Identifies vortex cores using Q = 0.5(||Ω||² - ||S||²) where Ω is rotation rate tensor and S is strain rate tensor
- **Vortex Core Detection:** Combines vorticity magnitude and Q-criterion thresholds for robust vortex identification
- **Circulation Analysis:** Calculates ∮ v⃗ · dl⃗ around multiple circular paths using line integrals
- **Streamline Integration:** Overlays flow patterns on vorticity fields for comprehensive visualization

**Scientific Output (6-Panel Analysis):**
1. **Vorticity Contours:** Color-coded rotation field with positive (red) and negative (blue) vorticity
2. **Q-Criterion Field:** Vortex identification map highlighting coherent rotating structures
3. **Detected Vortex Cores:** Velocity magnitude field with identified vortex cores outlined in red
4. **Vorticity + Streamlines:** Combined view showing rotation and flow direction
5. **Circulation Analysis:** Multiple integration paths with calculated circulation values
6. **Statistics Panel:** Quantitative metrics including max/mean vorticity, Q-criterion values, and vortex area

**Applications:**
- Turbulent flow analysis and eddy detection
- Mixing efficiency assessment in reactors
- Aerodynamic vortex shedding studies
- Flow separation and reattachment analysis

**Usage:** `python src/analysis/vorticity_visualizer.py --latest --output vorticity_results`

---

#### 📊 Cross-Section Analyzer (`src/analysis/cross_section_analyzer.py`)

**Detailed quantitative analysis of flow profiles and boundary layer characteristics.**

**Core Capabilities:**
- **Multi-Line Analysis:** Extracts data along horizontal, vertical, and diagonal sections through the flow domain
- **Boundary Layer Profiling:** Calculates boundary layer thickness (δ₉₉) where u = 0.99×u∞
- **Interactive Line Selection:** Click-to-define custom analysis paths with real-time profile extraction
- **Wake Detection:** Identifies low-velocity regions indicating flow separation or wake formation
- **Velocity Fluctuation Analysis:** Computes deviations from cross-sectional mean values

**Scientific Output (12-Panel Analysis):**
1. **Flow Field Overview:** Velocity magnitude with predefined analysis lines overlaid
2-5. **Profile Plots:** Velocity and pressure distributions along centerlines and quarter sections
6. **Boundary Layer Analysis:** Normalized velocity profiles (u/u_max vs y/δ) at multiple stations
7. **Velocity Station Comparison:** Vertical profiles at different streamwise locations
8. **Pressure Distribution:** Centerline pressure with pressure gradient (dp/dx) analysis
9. **Wake Region Detection:** Velocity field with detected low-velocity zones highlighted
10. **Velocity Fluctuations:** Spatial variations from mean flow patterns
11. **Statistics Summary:** Comprehensive flow metrics and boundary layer parameters
12. **Cross-sectional Averages:** Spanwise and streamwise averaged quantities

**Technical Features:**
- **Scipy Interpolation:** RegularGridInterpolator for accurate data extraction along arbitrary paths
- **Gradient Calculations:** Numpy gradient for pressure gradient and shear rate computations
- **Statistical Analysis:** Mean, standard deviation, and coefficient of variation calculations

**Applications:**
- Pipe and channel flow analysis
- Airfoil boundary layer studies
- Heat exchanger performance evaluation
- Validation against analytical solutions (Poiseuille, Blasius profiles)

**Usage:**
- Standard: `python src/analysis/cross_section_analyzer.py --latest`
- Interactive: `python src/analysis/cross_section_analyzer.py --latest --interactive`

---

#### 🔬 Parameter Study Tool (`src/analysis/parameter_study.py`)

**Comprehensive comparison and optimization analysis for CFD parameter variations.**

**Core Capabilities:**
- **Automated Parameter Extraction:** Parses simulation parameters from filenames using regex patterns
- **Statistical Comparison:** Calculates percentage changes and significance of parameter variations
- **Trend Analysis:** Polynomial fitting for parameter sweep relationships
- **Grid-Independent Comparison:** Handles different mesh sizes through interpolation
- **Solver Performance Analysis:** Quantitative comparison between different numerical methods

**Scientific Output:**
**Two-Case Comparison (12-Panel Analysis):**
1-2. **Velocity Fields:** Side-by-side velocity magnitude comparisons
3. **Difference Plot:** Quantitative velocity differences (Case2 - Case1)
4-5. **Pressure Fields:** Pressure distribution comparisons
6. **Pressure Difference:** Pressure change analysis
7. **Centerline Profiles:** u-velocity comparison along domain centerline
8. **Metrics Bar Chart:** Key performance indicators comparison
9. **Velocity Distributions:** Histogram comparison of velocity magnitude ranges
10-11. **Streamline Comparison:** Flow pattern visualization for both cases
12. **Statistics Table:** Comprehensive metrics with percentage changes color-coded by significance

**Parameter Sweep Analysis (6-Panel Trends):**
- Reynolds number effects on flow characteristics
- Viscosity influence on boundary layer development
- Time step sensitivity analysis
- Grid convergence studies
- Solver algorithm performance comparison

**Supported Parameters:**
- Reynolds number (Re), viscosity (μ), time step (dt)
- Grid resolution (nx×ny), solver type (basic/optimized)
- Custom parameters via filename patterns

**Applications:**
- Design optimization and sensitivity analysis
- Numerical method validation and verification
- Performance benchmarking between solver algorithms
- Academic research parameter studies

**Usage:**
- Compare: `python src/analysis/parameter_study.py --compare baseline.vtk optimized.vtk`
- Sweep: `python src/analysis/parameter_study.py --sweep Re --pattern "*Re*.vtk"`

---

#### 📡 Real-time Monitor (`src/analysis/realtime_monitor.py`)

**Live monitoring dashboard for tracking CFD simulation progress and convergence.**

**Core Capabilities:**
- **File System Monitoring:** Automatic detection of new VTK files using watchdog library
- **Real-time Visualization:** Live updates of velocity and pressure fields as simulation progresses
- **Convergence Tracking:** Time series analysis of key metrics with trend identification
- **Performance Metrics:** Kinetic energy, vorticity, and flow uniformity monitoring
- **Data Logging:** Automatic CSV export of all monitored quantities for post-processing

**Scientific Output (6-Panel Dashboard):**
1. **Live Velocity Field:** Real-time velocity magnitude contours from latest simulation output
2. **Live Pressure Field:** Current pressure distribution with automatic scaling
3. **Maximum Velocity Trend:** Time series showing peak velocity evolution
4. **Mean Velocity Trend:** Average velocity tracking for global flow assessment
5. **Kinetic Energy Evolution:** Total kinetic energy integration over the domain
6. **Statistics & Convergence Panel:** Current metrics with convergence trend analysis

**Monitoring Metrics:**
- **Flow Quantities:** Max/mean velocity, max/mean pressure, total kinetic energy
- **Flow Physics:** Maximum vorticity, velocity uniformity coefficient
- **Convergence Indicators:** 5-point trend analysis for determining steady-state approach
- **Performance Tracking:** Files processed, update frequency, monitoring duration

**Technical Features:**
- **Watchdog Integration:** Cross-platform file system event monitoring
- **Manual Polling Mode:** Fallback option for systems without watchdog support
- **Matplotlib Animation:** Efficient real-time plotting with automatic refresh
- **CSV Data Export:** Timestamped metrics for external analysis tools

**Applications:**
- Long-running simulation monitoring
- Convergence verification for steady-state problems
- Early termination detection for divergent solutions
- Performance monitoring for computational efficiency assessment

**Usage:**
- Auto-monitor: `python src/analysis/realtime_monitor.py --watch_dir ../../output/vtk_files`
- Manual mode: `python src/analysis/realtime_monitor.py --manual --interval 5.0`

### General Visualization Tools

#### Animation and Flow Visualization

**`animate_flow.py`** - **Advanced Flow Animation System**
- **Multi-frame Processing:** Batch processing of VTK time series for smooth animations
- **Customizable Rendering:** Vector fields, streamlines, and scalar field animations
- **Export Formats:** MP4, GIF, and frame sequence output options
- **Performance Optimization:** Memory-efficient processing of large datasets
- **Applications:** Flow evolution studies, unsteady flow visualization, presentation materials

**`create_cfd_animation.py`** - **Specialized CFD Animation Generator**
- **VTK Time Series Processing:** Automatic detection and sequencing of numbered VTK files
- **Multi-variable Animation:** Simultaneous animation of velocity, pressure, and vorticity
- **Synchronized Playback:** Coordinated visualization of multiple flow quantities
- **Custom Colormaps:** Scientific color schemes optimized for CFD data
- **Applications:** Research presentations, educational materials, simulation validation

**`create_simple_animation.py`** - **Quick Animation Utilities**
- **Rapid Prototyping:** Fast animation creation for preliminary analysis
- **Basic Flow Patterns:** Simple velocity and pressure animations
- **Educational Focus:** Clear, simplified visualizations for teaching
- **Lightweight Processing:** Minimal computational requirements
- **Applications:** Quick checks, educational demonstrations, rapid prototyping

**`velocity_flow_viz.py`** - **Comprehensive Velocity Field Visualization**
- **Vector Field Rendering:** Quiver plots with automatic scaling and density control
- **Streamline Generation:** Adaptive streamline placement with customizable integration
- **Magnitude Contours:** Velocity magnitude fields with scientific colormaps
- **Multi-panel Layouts:** Combined vector and scalar field presentations
- **Technical Features:** RegularGridInterpolator, adaptive streamline seeding, publication-quality rendering
- **Applications:** Flow pattern analysis, vector field validation, technical documentation

#### Interactive and Advanced Tools

**`interactive_dashboard.py`** - **Web-based CFD Analysis Dashboard**
- **Plotly Integration:** Interactive 3D visualizations with zoom, pan, and rotation capabilities
- **Multi-dataset Support:** Comparative analysis across multiple simulation runs
- **Real-time Interaction:** Dynamic parameter adjustment and instant visualization updates
- **Web Browser Interface:** Cross-platform accessibility without local software requirements
- **Export Capabilities:** High-resolution image and interactive HTML exports
- **Technical Features:** Plotly.graph_objects, subplot management, responsive design
- **Applications:** Collaborative analysis, remote visualization, interactive presentations

**`advanced_visualize.py`** - **Sophisticated Flow Analysis Platform**
- **Multi-physics Visualization:** Simultaneous rendering of flow, thermal, and species fields
- **Advanced Colormapping:** Custom scientific color schemes with perceptually uniform scaling
- **Statistical Overlays:** Mean, variance, and correlation field analysis
- **Publication Graphics:** Vector graphics output with precise typography and scaling
- **Custom Analysis Tools:** User-defined analysis regions and measurement tools
- **Applications:** Research publications, detailed flow analysis, multi-physics studies

**`enhanced_visualize.py`** - **Extended Visualization Capabilities**
- **3D Rendering Support:** Isosurface generation and volume rendering for 3D datasets
- **Particle Tracking:** Lagrangian particle path visualization and analysis
- **Temporal Analysis:** Time-averaged fields and fluctuation analysis
- **Boundary Condition Visualization:** Wall functions and inlet/outlet condition display
- **Grid Quality Assessment:** Mesh visualization and quality metrics
- **Applications:** Complex geometry flows, particle-laden flows, grid validation

#### Utilities and Core Functions

**`visualize_cfd.py`** - **Core CFD Visualization Library**
- **VTK File Processing:** Robust structured and unstructured VTK file parsing
- **Data Structure Management:** Efficient memory handling for large datasets
- **Coordinate System Support:** Cartesian, cylindrical, and curvilinear grid handling
- **Unit Conversion:** Automatic dimensional analysis and unit consistency checking
- **Error Handling:** Comprehensive file validation and error reporting
- **Technical Foundation:** NumPy arrays, SciPy interpolation, matplotlib integration
- **Applications:** Foundation library for all other visualization tools

**`simple_viz.py`** - **Rapid Visualization Utilities**
- **One-line Plotting:** Simple function calls for common visualization tasks
- **Default Configurations:** Pre-configured settings for standard CFD visualizations
- **Batch Processing:** Automated visualization of multiple files with consistent formatting
- **Template System:** Reusable visualization templates for common flow configurations
- **Applications:** Quick analysis, batch processing, standardized reporting

**`run_visualization.py`** - **Master Visualization Controller**
- **Workflow Management:** Orchestrates multiple visualization tools in sequence
- **Configuration System:** YAML/JSON configuration files for complex visualization workflows
- **Command-line Interface:** Complete CLI with parameter passing and output management
- **Parallel Processing:** Multi-core utilization for batch visualization tasks
- **Progress Tracking:** Real-time progress reporting for long visualization jobs
- **Applications:** Automated visualization pipelines, batch processing, workflow automation

**`test_animation.py`** - **Animation Testing and Validation**
- **Frame Rate Optimization:** Performance testing for smooth animation playback
- **Quality Assessment:** Visual quality validation and artifact detection
- **Format Compatibility:** Testing across different output formats and codecs
- **Memory Profiling:** Performance monitoring for large animation sequences
- **Regression Testing:** Validation of animation output consistency
- **Applications:** Quality assurance, performance optimization, format validation

#### Technical Specifications Summary

**Supported File Formats:**
- **Input:** VTK Structured Points, VTK Unstructured Grid, VTK Legacy format
- **Output:** PNG, JPG, SVG, PDF (static), MP4, GIF (animations), CSV (data export)

**Performance Capabilities:**
- **Grid Sizes:** Tested up to 1000×1000 structured grids
- **Memory Efficiency:** Streaming processing for large datasets
- **Multi-core Support:** Parallel processing where applicable
- **Interactive Response:** <1 second update times for typical visualizations

**Scientific Accuracy:**
- **Numerical Precision:** Float64 computation throughout analysis pipelines
- **Interpolation Methods:** Second-order accurate RegularGridInterpolator
- **Conservation Checking:** Mass and momentum conservation validation
- **Dimensional Analysis:** Automatic unit consistency verification


## Data

The `data/` directory contains:

- `test_frames/` - Test animation frames
- `animations/` - Generated animation files

## Dependencies

The Python scripts typically require:
- numpy
- matplotlib
- vtk (for VTK file processing)
- scipy (for advanced features)

## Usage

1. **Ensure your CFD simulation generates VTK output files**
2. **Activate the conda environment:**
   ```bash
   conda activate cfd-visualization
   ```
3. **Run visualization scripts:**
   ```bash
   # High-priority analysis tools
   python src/analysis/vorticity_visualizer.py --latest
   python src/analysis/cross_section_analyzer.py --latest --interactive
   python src/analysis/parameter_study.py --compare file1.vtk file2.vtk
   python src/analysis/realtime_monitor.py

   # Animation tools
   python src/animation/create_cfd_animation.py --input_dir ../../output/vtk_files

   # Interactive tools
   python src/interactive/interactive_dashboard.py

   # Core utilities
   python src/utilities/visualize_cfd.py
   # or use the console entry point:
   cfd-visualize
   ```

## Development

### Updating Dependencies

To add new dependencies:

1. **Add to environment.yml** for conda packages
2. **Add to requirements.txt** for pip packages
3. **Update the environment:**
   ```bash
   conda env update -f environment.yml
   ```

### Jupyter Notebooks

The environment includes Jupyter for interactive development:

```bash
conda activate cfd-visualization
jupyter notebook
```

## Quick Start Examples

**Analyze the latest simulation:**
```bash
conda activate cfd-visualization
python src/analysis/vorticity_visualizer.py --latest
python src/analysis/cross_section_analyzer.py --latest
```

**Compare two simulation runs:**
```bash
python src/analysis/parameter_study.py --compare sim1.vtk sim2.vtk
```

**Monitor a running simulation:**
```bash
python src/analysis/realtime_monitor.py --watch_dir ../../output/vtk_files
```

**📖 For detailed usage instructions, see [USAGE_GUIDE.md](USAGE_GUIDE.md)**

## Detailed Examples and Use Cases

### Example 1: Complete Flow Analysis Workflow
```bash
# Step 1: Generate CFD data
cd ../../
CMAKE_BUILD_TYPE=Release ./build.sh build
cd build/Release && ./animated_flow_simulation

# Step 2: Activate visualization environment
cd ../../../../cfd-visualization
conda activate cfd-visualization

# Step 3: Comprehensive analysis sequence
python src/analysis/vorticity_visualizer.py --latest --output flow_analysis
python src/analysis/cross_section_analyzer.py --latest --interactive --output flow_analysis
python src/analysis/parameter_study.py --input_dir ../../output/vtk_files --output flow_analysis
```

### Example 2: Solver Comparison Study
```bash
# Compare basic vs optimized solver performance
python src/analysis/parameter_study.py --compare \
    ../../output/vtk_files/output_100.vtk \
    ../../output/vtk_files/output_optimized_100.vtk \
    --output solver_comparison

# Analyze parameter sensitivity
python src/analysis/parameter_study.py --sweep mu --pattern "*mu*.vtk" --output parameter_study
```

### Example 3: Real-time Monitoring Setup
```bash
# Terminal 1: Start monitoring dashboard
python src/analysis/realtime_monitor.py \
    --watch_dir ../../output/vtk_files \
    --output monitoring_results

# Terminal 2: Run long simulation with periodic output
cd ../../
./build.sh run-long-simulation  # Custom long-running simulation
```

### Example 4: Boundary Layer Analysis
```bash
# Interactive boundary layer analysis
python src/analysis/cross_section_analyzer.py --latest --interactive

# Click points to define custom analysis lines:
# - Along wall-normal direction for boundary layer profiles
# - Along streamwise direction for pressure recovery
# - Across wake region for deficit analysis
```

### Example 5: Animation Creation Workflow
```bash
# Generate time series data
cd ../../
for i in {1..50}; do
    ./build.sh run-timestep $i
done

# Create comprehensive animation
cd ../cfd-visualization
python src/animation/create_cfd_animation.py \
    --input_dir ../../output/vtk_files \
    --output_dir animations \
    --fps 10 --format mp4
```

## Integration with CFD Framework

This visualization project is designed to work with VTK output files generated by the main CFD framework. The expected input format is structured VTK files containing velocity, pressure, and other flow field data.

### Supported Analysis Features

- **🌀 Vorticity Analysis:** Flow rotation, circulation, vortex core detection
- **📊 Cross-sectional Analysis:** Boundary layers, velocity profiles, wake analysis
- **🔬 Parameter Studies:** Solver comparison, parameter sweeps, optimization
- **📡 Real-time Monitoring:** Live convergence tracking, performance metrics

## Visualization Outputs Summary

### High-Priority Tools Output Files

| Tool | Output Files | Panels | Key Information |
|------|-------------|--------|----------------|
| **Vorticity Visualizer** | `vorticity_analysis.png` | 6 | Vorticity contours, Q-criterion, vortex cores, circulation values, statistics |
| **Cross-Section Analyzer** | `cross_section_analysis.png` | 12 | Line profiles, boundary layers, wake detection, pressure gradients, averages |
| **Parameter Study Tool** | `comparison_[cases].png`<br>`parameter_sweep_[param].png` | 12<br>6 | Side-by-side comparisons, difference plots, trend analysis, statistics tables |
| **Real-time Monitor** | Live dashboard<br>`monitoring_metrics.csv` | 6<br>Data | Real-time fields, convergence trends, performance metrics, logged data |

### Scientific Data Products

**Quantitative Analysis:**
- Boundary layer thickness (δ₉₉) calculations
- Circulation values (Γ) around multiple radii
- Vorticity magnitude and Q-criterion fields
- Pressure gradient distributions (dp/dx, dp/dy)
- Statistical metrics (mean, max, std, uniformity)

**Comparative Analysis:**
- Percentage changes between simulation cases
- Parameter sensitivity trends and correlations
- Solver performance benchmarking
- Grid convergence studies

**Time-Series Data:**
- Convergence monitoring metrics
- Real-time performance tracking
- CSV export for external analysis tools
- Automated data logging capabilities

### Research and Engineering Applications

**Academic Research:**
- Flow physics investigation and validation
- Numerical method development and testing
- Parameter optimization studies
- Publication-quality figure generation

**Industrial Applications:**
- Design optimization workflows
- Performance monitoring systems
- Quality assurance and validation
- Process optimization analysis

**Educational Use:**
- Flow visualization for teaching
- Interactive analysis demonstrations
- Concept illustration and explanation
- Student project support tools

## Environment Management

### List available environments:
```bash
conda env list
```

### Remove the environment:
```bash
conda env remove -n cfd-visualization
```

### Export current environment:
```bash
conda env export > environment.yml
```
