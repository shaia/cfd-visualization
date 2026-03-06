#!/usr/bin/env python3
"""Interactive 3D Surface Plots Example.

This example demonstrates how to create interactive 3D surface
visualizations of different CFD fields using Plotly.

Features demonstrated:
1. Velocity magnitude as a 3D surface
2. Pressure field as a 3D surface
3. Vorticity field as a 3D surface with diverging colorscale

Usage:
    python examples/3d_surface_plots.py

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
from cfd_viz.interactive import create_surface_figure


def run_simulation() -> str:
    """Run a steady-state CFD simulation.

    Returns:
        Path to the output VTK file.
    """
    output_file = "output/vtk/3d_surface_flow.vtk"
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


def create_3d_surfaces(vtk_file: str):
    """Create interactive 3D surface plots for different fields.

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
    dx, dy = data.dx, data.dy

    vel_mag = magnitude(u, v)
    omega = vorticity(u, v, dx, dy)

    print(f"Domain: [{x.min():.2f}, {x.max():.2f}] x [{y.min():.2f}, {y.max():.2f}]")
    print(f"Grid: {data.nx} x {data.ny}")
    print(f"Max velocity: {np.nanmax(vel_mag):.4f}")

    output_dir = Path("output/html")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Velocity magnitude surface
    print("\nCreating velocity magnitude 3D surface...")
    fig_vel = create_surface_figure(
        x,
        y,
        vel_mag,
        title="Velocity Magnitude",
        colorscale="Viridis",
    )
    out_path = output_dir / "3d_surface_velocity.html"
    fig_vel.write_html(str(out_path))
    print(f"  Saved: {out_path}")

    # 2. Pressure field surface
    print("Creating pressure 3D surface...")
    fig_p = create_surface_figure(
        x,
        y,
        p,
        title="Pressure Field",
        colorscale="Plasma",
    )
    out_path = output_dir / "3d_surface_pressure.html"
    fig_p.write_html(str(out_path))
    print(f"  Saved: {out_path}")

    # 3. Vorticity surface (diverging colorscale for signed values)
    print("Creating vorticity 3D surface...")
    fig_omega = create_surface_figure(
        x,
        y,
        omega,
        title="Vorticity Field",
        colorscale="RdBu",
    )
    out_path = output_dir / "3d_surface_vorticity.html"
    fig_omega.write_html(str(out_path))
    print(f"  Saved: {out_path}")


def main():
    """Main function."""
    print("3D Surface Plots Example")
    print("=" * 40)

    vtk_file = run_simulation()
    create_3d_surfaces(vtk_file)

    print("\n" + "=" * 40)
    print("Done!")
    print("\nOutput files (open in browser):")
    print("  - output/html/3d_surface_velocity.html")
    print("  - output/html/3d_surface_pressure.html")
    print("  - output/html/3d_surface_vorticity.html")
    print("\nInteraction tips:")
    print("  - Drag to rotate the surface")
    print("  - Scroll to zoom in/out")
    print("  - Hover to see field values")


if __name__ == "__main__":
    main()
