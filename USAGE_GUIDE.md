# CFD Visualization Tools - Usage Guide

This guide explains how to use the high-priority visualization tools implemented for your CFD framework.

## Prerequisites

1. **Activate the conda environment:**
   ```bash
   conda activate cfd-visualization
   ```

2. **Ensure your CFD simulation generates VTK output files** in `../../output/vtk_files/`

## High-Priority Tools

### 1. Vorticity Visualizer 🌀

**Purpose:** Analyze vorticity fields, circulation, and vortex cores

**Basic Usage:**
```bash
# Analyze latest VTK file
python src/analysis/vorticity_visualizer.py --latest

# Analyze specific file
python src/analysis/vorticity_visualizer.py path/to/file.vtk

# Custom output directory
python src/analysis/vorticity_visualizer.py --latest --output my_analysis
```

**What it provides:**
- Vorticity field contours with color-coded rotation
- Q-criterion for vortex identification
- Detected vortex cores highlighted in red
- Circulation analysis around multiple radii
- Comprehensive statistics and metrics

**Output:** `vorticity_analysis.png` with 6 detailed subplots

---

### 2. Cross-Section Analyzer 📊

**Purpose:** Detailed line plots, boundary layer analysis, and flow profiles

**Basic Usage:**
```bash
# Comprehensive analysis with predefined sections
python src/analysis/cross_section_analyzer.py --latest

# Interactive mode for custom line analysis
python src/analysis/cross_section_analyzer.py --latest --interactive

# Analyze specific file
python src/analysis/cross_section_analyzer.py path/to/file.vtk
```

**What it provides:**
- Velocity and pressure profiles along predefined lines
- Boundary layer thickness calculations
- Wake region detection
- Velocity fluctuation analysis
- Cross-sectional averages and statistics

**Interactive Features:**
- Click two points to define custom analysis lines
- Real-time profile extraction and plotting

**Output:** `cross_section_analysis.png` with 12 comprehensive subplots

---

### 3. Parameter Study Tool 🔬

**Purpose:** Compare simulation runs and analyze parameter variations

**Compare Two Cases:**
```bash
# Compare two specific files
python src/analysis/parameter_study.py --compare file1.vtk file2.vtk

# Compare latest two files in directory
python src/analysis/parameter_study.py --input_dir ../../output/vtk_files
```

**Parameter Sweep Analysis:**
```bash
# Analyze how Reynolds number affects flow
python src/analysis/parameter_study.py --sweep Re --pattern "*Re*.vtk"

# Analyze viscosity effects
python src/analysis/parameter_study.py --sweep mu --pattern "*mu*.vtk"

# Analyze time step effects
python src/analysis/parameter_study.py --sweep dt --pattern "*dt*.vtk"
```

**What it provides:**
- Side-by-side field comparisons
- Difference plots highlighting changes
- Metrics comparison bar charts
- Velocity distribution histograms
- Detailed statistics tables with percentage changes
- Parameter trend analysis

**Output:**
- `comparison_[case1]_vs_[case2].png` for two-case comparison
- `parameter_sweep_[parameter].png` for sweep analysis

---

### 4. Real-time Monitor 📡

**Purpose:** Live monitoring of running CFD simulations

**Basic Usage:**
```bash
# Auto-monitoring with file system events
python src/analysis/realtime_monitor.py --watch_dir ../../output/vtk_files

# Manual polling mode (if watchdog not available)
python src/analysis/realtime_monitor.py --manual --interval 5.0

# Custom monitoring setup
python src/analysis/realtime_monitor.py --watch_dir /custom/path --output monitoring_results
```

**What it provides:**
- Live velocity and pressure field updates
- Real-time convergence tracking
- Time series plots of key metrics
- Convergence trend analysis
- Automatic metrics logging to CSV

**Interactive Features:**
- Dashboard updates automatically when new VTK files appear
- Convergence status indicators
- Performance metrics tracking

**Output:**
- Live dashboard display
- `monitoring_metrics.csv` with time series data

## Example Workflows

### 1. Complete Flow Analysis
```bash
# Step 1: Run your CFD simulation
cd ../../
./build.sh run

# Step 2: Activate visualization environment
conda activate cfd-visualization
cd ../cfd-visualization

# Step 3: Comprehensive analysis
python src/analysis/vorticity_visualizer.py --latest
python src/analysis/cross_section_analyzer.py --latest --interactive
```

### 2. Parameter Study Workflow
```bash
# Run multiple simulations with different parameters
# (modify parameters in your CFD code)

# Analyze the parameter sweep
python src/analysis/parameter_study.py --sweep Re --pattern "*Re*.vtk"
python src/analysis/parameter_study.py --sweep mu --pattern "*mu*.vtk"
```

### 3. Real-time Monitoring Workflow
```bash
# Terminal 1: Start monitoring
python src/analysis/realtime_monitor.py

# Terminal 2: Run long CFD simulation
cd ../../
./build.sh run-long-simulation
```

### 4. Solver Comparison
```bash
# Compare basic vs optimized solver
python src/analysis/parameter_study.py --pattern "*output*.vtk" --pattern "*output_optimized*.vtk"
```

## Tips and Best Practices

### File Naming Conventions
For automatic parameter extraction, use descriptive filenames:
- `simulation_Re100_mu0.01_dt0.001.vtk`
- `output_optimized_500.vtk`
- `flow_grid50x25_Re1000.vtk`

### Performance Optimization
- Use `--latest` flag to analyze only the most recent file
- For large datasets, consider analyzing every nth file
- Real-time monitoring works best with moderate grid sizes

### Troubleshooting

**Common Issues:**

1. **"No VTK files found"**
   - Check that your CFD simulation is generating VTK output
   - Verify the path to output directory
   - Ensure files have `.vtk` extension

2. **"Could not find velocity data"**
   - Verify VTK files contain `u_velocity` and `v_velocity` fields
   - Check VTK file format matches expected structure

3. **Interactive plots not working**
   - Ensure you're not using a headless environment
   - Try adding `plt.show()` if plots don't appear

4. **Real-time monitor not updating**
   - Use `--manual` mode if file system events aren't working
   - Check file permissions in watch directory
   - Install `watchdog` package: `pip install watchdog`

### Advanced Usage

**Custom Analysis Scripts:**
```python
# Import and use functions directly
from src.vorticity_visualizer import calculate_vorticity, detect_vortex_cores
from src.cross_section_analyzer import extract_line_data, analyze_boundary_layer

# Create your own analysis workflows
```

**Batch Processing:**
```bash
# Analyze all files in a directory
for file in ../../output/vtk_files/*.vtk; do
    python src/vorticity_visualizer.py "$file"
done
```

## Output Directory Structure

```
visualization_output/
├── vorticity_analysis.png          # Vorticity analysis results
├── cross_section_analysis.png      # Cross-sectional analysis
├── comparison_case1_vs_case2.png   # Parameter comparisons
├── parameter_sweep_Re.png          # Parameter sweep results
├── monitoring_metrics.csv          # Real-time monitoring data
└── ...                             # Additional analysis files
```

## Integration with Main CFD Project

These tools work seamlessly with VTK files generated by your main CFD framework:

1. **Run CFD simulation:** `./build.sh run`
2. **Switch to visualization:** `cd ../cfd-visualization && conda activate cfd-visualization`
3. **Analyze results:** Use any of the visualization tools above

The tools automatically find and process VTK files from your CFD simulation output directory.