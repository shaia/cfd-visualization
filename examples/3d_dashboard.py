#!/usr/bin/env python3
"""Multi-panel Dashboard with 3D Surface Example.

This example demonstrates how to create a multi-panel interactive
dashboard that includes a 3D surface subplot alongside 2D heatmaps
and vector field visualizations.

Features demonstrated:
1. Multi-panel Plotly dashboard with 2x3 grid layout
2. 3D surface subplot showing velocity magnitude
3. 2D heatmaps for pressure and vorticity
4. Velocity vector field overlay

Usage:
    python examples/3d_dashboard.py

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
from cfd_viz.interactive import create_dashboard_figure, create_interactive_frame


def run_simulation() -> str:
    """Run a steady-state CFD simulation.

    Returns:
        Path to the output VTK file.
    """
    output_file = "output/vtk/3d_dashboard_flow.vtk"
    Path("output/vtk").mkdir(parents=True, exist_ok=True)

    print("Running simulation: 80x80 grid, 500 steps...")
    cfd_python.run_simulation_with_params(
        nx=80,
        ny=80,
        xmin=0.0,
        xmax=1.0,
        ymin=0.0,
        ymax=1.0,
        steps=500,
        output_file=output_file,
    )
    print("Simulation complete!")
    return output_file


def create_dashboard(vtk_file: str):
    """Create a multi-panel dashboard with 3D surface.

    Args:
        vtk_file: Path to VTK file with simulation results.
    """
    print(f"\nLoading results from: {vtk_file}")

    data = read_vtk_file(vtk_file)
    if data is None:
        print(f"Error: Could not read {vtk_file}")
        return

    x, y = data.x, data.y
    u, v = data.u, data.v
    p = data.get("p", np.zeros_like(u))

    print(f"Domain: [{x.min():.2f}, {x.max():.2f}] x [{y.min():.2f}, {y.max():.2f}]")
    print(f"Grid: {data.nx} x {data.ny}")

    output_dir = Path("output/html")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create interactive frame (bundles x, y, u, v, p for the dashboard)
    frame = create_interactive_frame(x, y, u, v, p=p, time_index=500)

    # Create multi-panel dashboard (includes 3D surface subplot)
    print("\nCreating multi-panel dashboard with 3D surface...")
    fig = create_dashboard_figure(
        frame,
        title="CFD Flow Dashboard with 3D Surface",
    )

    out_path = output_dir / "3d_dashboard.html"
    fig.write_html(str(out_path))
    print(f"  Saved: {out_path}")


def main():
    """Main function."""
    print("3D Dashboard Example")
    print("=" * 40)

    vtk_file = run_simulation()
    create_dashboard(vtk_file)

    print("\n" + "=" * 40)
    print("Done!")
    print("\nOutput file (open in browser):")
    print("  - output/html/3d_dashboard.html")
    print("\nThe dashboard includes:")
    print("  - Velocity magnitude heatmap")
    print("  - Velocity vector field")
    print("  - Pressure heatmap")
    print("  - Vorticity heatmap")
    print("  - Streamlines")
    print("  - 3D velocity magnitude surface (drag to rotate)")


if __name__ == "__main__":
    main()
