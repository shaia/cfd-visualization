#!/usr/bin/env python3
"""Basic CFD Simulation and Visualization Example.

This example demonstrates how to:
1. Run a CFD simulation using cfd_python
2. Visualize the results using cfd_viz

The simulation creates a lid-driven cavity flow which develops
vortical structures.

Usage:
    python examples/basic_simulation.py

Requirements:
    - cfd_python package (pip install -e ../cfd-python)
    - cfd_viz package
"""

import sys

import matplotlib.pyplot as plt
import numpy as np

# Check for cfd_python
try:
    import cfd_python
except ImportError:
    print("Error: cfd_python package not installed.")
    print("Install with: pip install -e ../cfd-python")
    sys.exit(1)

from cfd_viz.common import read_vtk_file
from cfd_viz.fields import magnitude, vorticity
from cfd_viz.plotting import (
    plot_pressure_field,
    plot_streamlines,
    plot_vector_field,
    plot_velocity_field,
    plot_vorticity_field,
)


def run_simulation(
    nx: int = 100,
    ny: int = 100,
    steps: int = 500,
    output_file: str = "flow_simulation.vtk",
) -> str:
    """Run CFD simulation using cfd_python.

    Args:
        nx: Number of grid points in x-direction.
        ny: Number of grid points in y-direction.
        steps: Number of simulation steps.
        output_file: Path for VTK output file.

    Returns:
        Path to the output VTK file.
    """
    print(f"Running simulation: {nx}x{ny} grid, {steps} steps...")
    print(f"Available solvers: {cfd_python.list_solvers()}")

    # Run simulation with VTK output
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


def visualize_results(vtk_file: str):
    """Visualize CFD simulation results.

    Args:
        vtk_file: Path to VTK file with simulation results.
    """
    print(f"\nLoading results from: {vtk_file}")

    # Read VTK file
    data = read_vtk_file(vtk_file)
    if data is None:
        print(f"Error: Could not read {vtk_file}")
        return

    # Extract fields
    X, Y = data.X, data.Y
    u, v = data.u, data.v
    p = data.get("p", np.zeros_like(u))
    dx, dy = data.dx, data.dy

    # Compute derived quantities
    vel_mag = magnitude(u, v)
    omega = vorticity(u, v, dx, dy)

    print(
        f"Domain: [{data.x.min():.2f}, {data.x.max():.2f}] x [{data.y.min():.2f}, {data.y.max():.2f}]"
    )
    print(f"Grid: {data.nx} x {data.ny}")
    print(f"Max velocity: {np.nanmax(vel_mag):.4f}")
    print(f"Max vorticity: {np.nanmax(np.abs(omega)):.4f}")

    # Create visualization
    print("\nCreating visualizations...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Velocity magnitude
    ax1 = axes[0, 0]
    plot_velocity_field(X, Y, u, v, ax=ax1, title="Velocity Magnitude")

    # 2. Pressure field
    ax2 = axes[0, 1]
    plot_pressure_field(X, Y, p, ax=ax2, title="Pressure Field")

    # 3. Vorticity
    ax3 = axes[0, 2]
    plot_vorticity_field(X, Y, omega, ax=ax3, title="Vorticity")

    # 4. Vector field
    ax4 = axes[1, 0]
    plot_vector_field(X, Y, u, v, ax=ax4, title="Velocity Vectors", density=15)

    # 5. Streamlines
    ax5 = axes[1, 1]
    plot_streamlines(X, Y, u, v, ax=ax5, title="Streamlines", density=2)

    # 6. Combined view
    ax6 = axes[1, 2]
    contour = ax6.contourf(X, Y, vel_mag, levels=20, cmap="viridis", alpha=0.8)
    plt.colorbar(contour, ax=ax6, label="Velocity")
    ax6.streamplot(X, Y, u, v, color="white", density=1.5, linewidth=0.8)
    ax6.set_xlabel("x")
    ax6.set_ylabel("y")
    ax6.set_title("Combined: Velocity + Streamlines")
    ax6.set_aspect("equal")

    plt.suptitle("CFD Flow Simulation Results", fontsize=14)
    plt.tight_layout()

    # Save figure
    output_file = "flow_simulation_results.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_file}")

    plt.show()


def main():
    """Main function to run simulation and visualize results."""
    print("CFD Flow Simulation Example")
    print("=" * 40)
    print()

    # Run simulation
    vtk_file = run_simulation(
        nx=80,
        ny=80,
        steps=300,
        output_file="flow_simulation.vtk",
    )

    # Visualize results
    visualize_results(vtk_file)

    print("\nDone!")


if __name__ == "__main__":
    main()
