#!/usr/bin/env python3
"""
Vorticity Analysis Example
==========================

Demonstrates vorticity and circulation analysis using the vorticity_visualizer
module. Runs a simulation and analyzes the resulting flow field for:
- Vorticity field visualization
- Q-criterion for vortex identification
- Vortex core detection
- Circulation calculation

Requirements:
    cfd-python, numpy, matplotlib, scipy
"""

import sys
import os
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'analysis'))

from config import DATA_DIR, PLOTS_DIR, ensure_dirs

try:
    import cfd_python
    CFD_AVAILABLE = True
except ImportError:
    CFD_AVAILABLE = False

from vorticity_visualizer import (
    read_vtk_file,
    calculate_vorticity,
    calculate_q_criterion,
    detect_vortex_cores,
    create_vorticity_visualization
)


def run_simulation() -> Optional[str]:
    """Run a lid-driven cavity simulation to generate vortical flow."""
    if not CFD_AVAILABLE:
        print("Error: cfd-python not available")
        return None

    ensure_dirs()
    cfd_python.set_output_dir(str(DATA_DIR))

    output_file = str(DATA_DIR / "vorticity_example.vtk")

    print("Running lid-driven cavity simulation...")
    print("(This generates vortical flow patterns)")

    # Lid-driven cavity produces interesting vorticity patterns
    result = cfd_python.run_simulation_with_params(
        nx=80,
        ny=80,
        xmin=0.0,
        xmax=1.0,
        ymin=0.0,
        ymax=1.0,
        steps=500,
        solver_type='projection',
        output_file=output_file
    )

    print(f"Simulation complete: {output_file}")
    return output_file


def analyze_vorticity(vtk_file: str) -> None:
    """Analyze vorticity in the flow field."""
    print("\nReading VTK file...")
    data = read_vtk_file(vtk_file)

    if data is None:
        print("Error: Could not read VTK file")
        return

    print(f"Grid size: {data['nx']} x {data['ny']}")
    print(f"Grid spacing: dx={data['dx']:.4f}, dy={data['dy']:.4f}")

    # Calculate vorticity
    print("\nCalculating vorticity field...")
    vorticity = calculate_vorticity(data['u'], data['v'], data['dx'], data['dy'])

    # Calculate Q-criterion
    print("Calculating Q-criterion...")
    Q = calculate_q_criterion(data['u'], data['v'], data['dx'], data['dy'])

    # Detect vortex cores
    print("Detecting vortex cores...")
    vortex_cores = detect_vortex_cores(vorticity, Q)

    # Print statistics
    print("\n" + "=" * 50)
    print("Vorticity Analysis Results")
    print("=" * 50)
    print(f"Max vorticity:  {vorticity.max():+.4f}")
    print(f"Min vorticity:  {vorticity.min():+.4f}")
    print(f"Mean vorticity: {vorticity.mean():+.6f}")
    print(f"Max Q-criterion: {Q.max():.4f}")
    print(f"Vortex core cells: {vortex_cores.sum()}")
    print("=" * 50)

    # Create visualization
    output_dir = str(PLOTS_DIR / "vorticity_analysis")
    print(f"\nCreating visualizations in {output_dir}...")
    create_vorticity_visualization(data, output_dir)

    print("\nAnalysis complete!")
    print(f"Output files saved to: {output_dir}")


def main() -> None:
    """Main entry point."""
    ensure_dirs()

    print("Vorticity Analysis Example")
    print("=" * 50)
    print()

    # Run simulation
    vtk_file = run_simulation()
    if vtk_file is None:
        return

    # Analyze vorticity
    analyze_vorticity(vtk_file)


if __name__ == "__main__":
    main()
