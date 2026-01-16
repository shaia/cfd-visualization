#!/usr/bin/env python3
"""Lid-Driven Cavity Variations Example.

This example explores different cavity configurations:
1. Square cavity (standard benchmark)
2. Tall cavity (height > width)
3. Wide cavity (width > height)

Each configuration produces different vortex structures and flow patterns.

Physical Setup:
- Top wall moves at constant velocity (the "lid")
- All other walls are stationary (no-slip)
- Flow is driven purely by viscous drag from the lid

Usage:
    python examples/cavity_variations.py

Requirements:
    - cfd_python package (pip install -e ../cfd-python)
    - cfd_viz package
"""

import sys

import matplotlib.pyplot as plt
import numpy as np

try:
    import cfd_python
except ImportError:
    print("Error: cfd_python package not installed.")
    print("Install with: pip install -e ../cfd-python")
    sys.exit(1)

from cfd_viz.common import read_vtk_file
from cfd_viz.fields import magnitude, vorticity


def run_cavity_simulation(
    nx: int,
    ny: int,
    xmax: float,
    ymax: float,
    steps: int,
    output_file: str,
) -> str:
    """Run cavity simulation with given parameters.

    Args:
        nx, ny: Grid dimensions.
        xmax, ymax: Domain size.
        steps: Number of time steps.
        output_file: Output VTK file path.

    Returns:
        Path to output file.
    """
    result = cfd_python.run_simulation_with_params(
        nx=nx,
        ny=ny,
        xmin=0.0,
        xmax=xmax,
        ymin=0.0,
        ymax=ymax,
        steps=steps,
        output_file=output_file,
    )

    max_vel = "N/A"
    if isinstance(result, dict) and "stats" in result:
        max_vel = f"{result['stats'].get('max_velocity', 0):.4f}"

    print(f"  {output_file}: max_vel = {max_vel}")
    return output_file


def visualize_cavity_comparison(vtk_files: list, titles: list):
    """Create comparison visualization of different cavity configurations.

    Args:
        vtk_files: List of VTK file paths.
        titles: List of titles for each configuration.
    """
    n_cases = len(vtk_files)
    fig, axes = plt.subplots(3, n_cases, figsize=(5 * n_cases, 12))

    for col, (vtk_file, title) in enumerate(zip(vtk_files, titles)):
        data = read_vtk_file(vtk_file)
        if data is None:
            print(f"Error reading {vtk_file}")
            continue

        X, Y = data.X, data.Y
        u, v = data.u, data.v
        dx, dy = data.dx, data.dy

        vel_mag = magnitude(u, v)
        omega = vorticity(u, v, dx, dy)

        # Row 1: Velocity magnitude
        ax1 = axes[0, col]
        c1 = ax1.contourf(X, Y, vel_mag, levels=20, cmap="viridis")
        plt.colorbar(c1, ax=ax1, label="Velocity")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_title(f"{title}\nVelocity Magnitude")
        ax1.set_aspect("equal")

        # Row 2: Streamlines
        ax2 = axes[1, col]
        ax2.contourf(X, Y, vel_mag, levels=20, cmap="viridis", alpha=0.5)
        ax2.streamplot(X, Y, u, v, color="black", density=1.5, linewidth=0.8)
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_title("Streamlines")
        ax2.set_aspect("equal")

        # Row 3: Vorticity
        ax3 = axes[2, col]
        vort_max = np.max(np.abs(omega))
        levels = np.linspace(-vort_max, vort_max, 21)
        c3 = ax3.contourf(X, Y, omega, levels=levels, cmap="RdBu_r")
        plt.colorbar(c3, ax=ax3, label="Vorticity")
        ax3.set_xlabel("x")
        ax3.set_ylabel("y")
        ax3.set_title("Vorticity")
        ax3.set_aspect("equal")

    plt.suptitle("Lid-Driven Cavity: Effect of Aspect Ratio", fontsize=14)
    plt.tight_layout()

    output_file = "cavity_variations_results.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {output_file}")

    plt.show()


def main():
    """Main function."""
    print("Lid-Driven Cavity Variations Example")
    print("=" * 40)
    print()

    # Define cavity configurations
    configurations = [
        {
            "name": "Square (1:1)",
            "nx": 80,
            "ny": 80,
            "xmax": 1.0,
            "ymax": 1.0,
            "steps": 500,
            "output": "cavity_square.vtk",
        },
        {
            "name": "Tall (1:2)",
            "nx": 50,
            "ny": 100,
            "xmax": 0.5,
            "ymax": 1.0,
            "steps": 500,
            "output": "cavity_tall.vtk",
        },
        {
            "name": "Wide (2:1)",
            "nx": 100,
            "ny": 50,
            "xmax": 1.0,
            "ymax": 0.5,
            "steps": 500,
            "output": "cavity_wide.vtk",
        },
    ]

    print("Running cavity simulations...")
    vtk_files = []
    titles = []

    for config in configurations:
        print(f"\n{config['name']}:")
        vtk_file = run_cavity_simulation(
            nx=config["nx"],
            ny=config["ny"],
            xmax=config["xmax"],
            ymax=config["ymax"],
            steps=config["steps"],
            output_file=config["output"],
        )
        vtk_files.append(vtk_file)
        titles.append(config["name"])

    print("\nCreating comparison visualization...")
    visualize_cavity_comparison(vtk_files, titles)

    print("\nDone!")


if __name__ == "__main__":
    main()
