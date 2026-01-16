#!/usr/bin/env python3
"""Interactive Plotly Visualization Example.

This example demonstrates how to create interactive visualizations
using Plotly with CFD simulation data.

Features demonstrated:
1. Interactive heatmaps with zoom/pan
2. 3D surface plots with rotation
3. Multi-panel dashboards
4. Animated dashboards with playback controls

Usage:
    python examples/interactive_plotly.py

Requirements:
    - cfd_python package (pip install -e ../cfd-python)
    - cfd_viz package
    - plotly package
"""

import sys
from pathlib import Path

import numpy as np

try:
    import cfd_python
except ImportError:
    print("Error: cfd_python package not installed.")
    print("Install with: pip install -e ../cfd-python")
    sys.exit(1)

from cfd_viz.common import read_vtk_file
from cfd_viz.fields import magnitude, vorticity
from cfd_viz.interactive import (
    create_contour_figure,
    create_convergence_figure,
    create_dashboard_figure,
    create_heatmap_figure,
    create_interactive_frame,
    create_interactive_frame_collection,
    create_surface_figure,
    create_vector_figure,
)


def run_simulation(
    nx: int = 80,
    ny: int = 80,
    steps: int = 500,
    output_file: str = "interactive_flow.vtk",
) -> str:
    """Run CFD simulation.

    Args:
        nx, ny: Grid dimensions.
        steps: Number of simulation steps.
        output_file: Output VTK file path.

    Returns:
        Path to output file.
    """
    print(f"Running simulation: {nx}x{ny} grid, {steps} steps...")

    result = cfd_python.run_simulation_with_params(
        nx=nx,
        ny=ny,
        xmin=0.0,
        xmax=1.0,
        ymin=0.0,
        ymax=1.0,
        steps=steps,
        output_file=output_file,
    )

    print("Simulation complete!")
    if isinstance(result, dict) and "stats" in result:
        stats = result["stats"]
        print(f"  Max velocity: {stats.get('max_velocity', 'N/A')}")

    return output_file


def run_transient_simulations(
    nx: int = 60,
    ny: int = 60,
    total_steps: int = 300,
    output_interval: int = 50,
    output_dir: str = "interactive_transient",
) -> list:
    """Run multiple simulations for transient visualization.

    Args:
        nx, ny: Grid dimensions.
        total_steps: Total simulation steps.
        output_interval: Steps between outputs.
        output_dir: Directory for output files.

    Returns:
        List of output VTK file paths.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print("Running transient simulations...")
    print(f"  Grid: {nx} x {ny}")
    print(f"  Total steps: {total_steps}")
    print(f"  Output interval: {output_interval}")

    vtk_files = []

    for step in range(output_interval, total_steps + 1, output_interval):
        output_file = str(output_path / f"flow_{step:04d}.vtk")

        cfd_python.run_simulation_with_params(
            nx=nx,
            ny=ny,
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
            steps=step,
            output_file=output_file,
        )

        print(f"  Step {step}: {output_file}")
        vtk_files.append(output_file)

    return vtk_files


def create_single_frame_visualizations(vtk_file: str):
    """Create various interactive visualizations from a single simulation.

    Args:
        vtk_file: Path to VTK file.
    """
    print(f"\nLoading results from: {vtk_file}")

    data = read_vtk_file(vtk_file)
    if data is None:
        print(f"Error: Could not read {vtk_file}")
        return

    # Extract fields
    x, y = data.x, data.y
    u, v = data.u, data.v
    p = data.get("p", np.zeros_like(u))
    dx, dy = data.dx, data.dy

    # Compute derived quantities
    vel_mag = magnitude(u, v)
    omega = vorticity(u, v, dx, dy)

    print(f"Domain: [{x.min():.2f}, {x.max():.2f}] x [{y.min():.2f}, {y.max():.2f}]")
    print(f"Grid: {data.nx} x {data.ny}")
    print(f"Max velocity: {np.nanmax(vel_mag):.4f}")

    # 1. Interactive Heatmap
    print("\nCreating interactive heatmap...")
    fig_heatmap = create_heatmap_figure(
        x,
        y,
        vel_mag,
        title="Velocity Magnitude (Interactive)",
        colorscale="Viridis",
    )
    fig_heatmap.write_html("interactive_heatmap.html")
    print("  Saved: interactive_heatmap.html")

    # 2. Interactive Contour
    print("Creating interactive contour plot...")
    fig_contour = create_contour_figure(
        x,
        y,
        omega,
        title="Vorticity Field (Interactive)",
        colorscale="RdBu",
        ncontours=30,
    )
    fig_contour.write_html("interactive_contour.html")
    print("  Saved: interactive_contour.html")

    # 3. Interactive Vector Field
    print("Creating interactive vector field...")
    fig_vector = create_vector_figure(
        x,
        y,
        u,
        v,
        subsample=4,
        title="Velocity Vectors (Interactive)",
    )
    fig_vector.write_html("interactive_vectors.html")
    print("  Saved: interactive_vectors.html")

    # 4. 3D Surface Plot
    print("Creating 3D surface plot...")
    fig_surface = create_surface_figure(
        x,
        y,
        vel_mag,
        title="Velocity Magnitude 3D Surface",
        colorscale="Viridis",
    )
    fig_surface.write_html("interactive_surface.html")
    print("  Saved: interactive_surface.html")

    # 5. Multi-panel Dashboard
    print("Creating multi-panel dashboard...")
    frame = create_interactive_frame(x, y, u, v, p=p, time_index=500)
    fig_dashboard = create_dashboard_figure(
        frame,
        title="CFD Flow Dashboard",
    )
    fig_dashboard.write_html("interactive_dashboard.html")
    print("  Saved: interactive_dashboard.html")


def create_transient_visualizations(vtk_files: list):
    """Create transient/animated interactive visualizations.

    Args:
        vtk_files: List of VTK file paths.
    """
    print("\nCreating transient visualizations...")

    # Load all frames
    frames_data = []
    time_indices = []

    for vtk_file in vtk_files:
        data = read_vtk_file(vtk_file)
        if data is None:
            print(f"Warning: Could not read {vtk_file}")
            continue

        x, y = data.x, data.y
        u, v = data.u, data.v
        p = data.get("p", np.zeros_like(u))

        frames_data.append((x, y, u, v, p))
        # Extract time index from filename
        time_idx = int(vtk_file.split("_")[-1].split(".")[0])
        time_indices.append(time_idx)

    print(f"Loaded {len(frames_data)} frames")

    # Create frame collection
    frame_collection = create_interactive_frame_collection(frames_data, time_indices)

    # Convergence history plot
    print("Creating convergence history plot...")
    fig_convergence = create_convergence_figure(
        frame_collection,
        title="Simulation Convergence History",
    )
    fig_convergence.write_html("interactive_convergence.html")
    print("  Saved: interactive_convergence.html")


def main():
    """Main function."""
    print("Interactive Plotly Visualization Example")
    print("=" * 45)
    print()

    # Run steady-state simulation
    vtk_file = run_simulation(
        nx=80,
        ny=80,
        steps=500,
        output_file="interactive_flow.vtk",
    )

    # Create single-frame visualizations
    create_single_frame_visualizations(vtk_file)

    # Run transient simulations
    vtk_files = run_transient_simulations(
        nx=60,
        ny=60,
        total_steps=300,
        output_interval=50,
        output_dir="interactive_transient",
    )

    # Create transient visualizations
    create_transient_visualizations(vtk_files)

    print("\n" + "=" * 45)
    print("Done!")
    print("\nOutput files (open in browser):")
    print("  - interactive_heatmap.html")
    print("  - interactive_contour.html")
    print("  - interactive_vectors.html")
    print("  - interactive_surface.html")
    print("  - interactive_dashboard.html")
    print("  - interactive_convergence.html")
    print("\nFeatures:")
    print("  - Zoom: scroll or pinch")
    print("  - Pan: drag")
    print("  - Hover: see values")
    print("  - 3D: rotate by dragging")


if __name__ == "__main__":
    main()
