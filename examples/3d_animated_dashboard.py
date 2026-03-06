#!/usr/bin/env python3
"""Animated Dashboard with 3D Views Example.

This example demonstrates how to create an animated Plotly dashboard
that includes 3D scatter plots and 3D surface views alongside 2D
visualizations, with Play/Pause playback controls.

Features demonstrated:
1. Animated multi-panel dashboard (3x3 grid)
2. 3D scatter plot of velocity vectors colored by magnitude
3. 3D surface plot of velocity magnitude
4. Playback controls (Play/Pause, timeline slider)
5. Time evolution of flow fields

Usage:
    python examples/3d_animated_dashboard.py

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
from cfd_viz.interactive import (
    create_animated_dashboard,
    create_interactive_frame_collection,
)


def run_transient_simulations() -> list:
    """Run simulations at multiple timesteps.

    Returns:
        List of output VTK file paths.
    """
    output_dir = Path("output/vtk/3d_animated_transient")
    output_dir.mkdir(parents=True, exist_ok=True)

    total_steps = 300
    output_interval = 50

    print("Running transient simulations...")
    print("  Grid: 60 x 60")
    print(f"  Total steps: {total_steps}")
    print(f"  Output interval: {output_interval}")

    vtk_files = []
    for step in range(output_interval, total_steps + 1, output_interval):
        output_file = str(output_dir / f"flow_{step:04d}.vtk")

        cfd_python.run_simulation_with_params(
            nx=60,
            ny=60,
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


def create_animated_3d_dashboard(vtk_files: list):
    """Create an animated dashboard with 3D views.

    Args:
        vtk_files: List of VTK file paths (one per timestep).
    """
    print("\nLoading simulation frames...")

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
        time_idx = int(vtk_file.split("_")[-1].split(".")[0])
        time_indices.append(time_idx)

    print(f"Loaded {len(frames_data)} frames")

    # Create frame collection for animated dashboard
    frame_collection = create_interactive_frame_collection(frames_data, time_indices)

    # Create animated dashboard with 3D views
    print("\nCreating animated dashboard with 3D views...")
    fig = create_animated_dashboard(
        frame_collection,
        title="3D Animated Flow Dashboard",
    )

    output_dir = Path("output/html")
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / "3d_animated_dashboard.html"
    fig.write_html(str(out_path))
    print(f"  Saved: {out_path}")


def main():
    """Main function."""
    print("3D Animated Dashboard Example")
    print("=" * 40)

    vtk_files = run_transient_simulations()
    create_animated_3d_dashboard(vtk_files)

    print("\n" + "=" * 40)
    print("Done!")
    print("\nOutput file (open in browser):")
    print("  - output/html/3d_animated_dashboard.html")
    print("\nThe dashboard includes:")
    print("  - Velocity magnitude heatmap (animated)")
    print("  - Velocity vectors (animated)")
    print("  - Pressure heatmap (animated)")
    print("  - Vorticity heatmap (animated)")
    print("  - 3D flow scatter plot (drag to rotate)")
    print("  - 3D velocity magnitude surface (drag to rotate)")
    print("  - Play/Pause controls and timeline slider")


if __name__ == "__main__":
    main()
