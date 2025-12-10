#!/usr/bin/env python3
"""Transient Flow Animation Example.

This example captures the time evolution of a lid-driven cavity flow,
showing how vortices develop from rest to steady state.

Physical Setup:
- Lid-driven cavity starting from rest
- Multiple snapshots captured during development
- Animation shows vortex formation and growth

Usage:
    python examples/transient_animation.py

Requirements:
    - cfd_python package (pip install -e ../cfd-python)
    - cfd_viz package
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    import cfd_python
except ImportError:
    print("Error: cfd_python package not installed.")
    print("Install with: pip install -e ../cfd-python")
    sys.exit(1)

from cfd_viz.animation import (
    create_animation_frames,
    create_field_animation,
    create_multi_panel_animation,
    save_animation,
)
from cfd_viz.common import read_vtk_file
from cfd_viz.fields import magnitude, vorticity


def run_transient_simulation(
    nx: int = 80,
    ny: int = 80,
    total_steps: int = 500,
    output_interval: int = 50,
    output_dir: str = "transient_output",
) -> list:
    """Run simulation with multiple output snapshots.

    Args:
        nx, ny: Grid dimensions.
        total_steps: Total simulation steps.
        output_interval: Steps between outputs.
        output_dir: Directory for output files.

    Returns:
        List of output VTK file paths.
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print("Running transient simulation...")
    print(f"  Grid: {nx} x {ny}")
    print(f"  Total steps: {total_steps}")
    print(f"  Output interval: {output_interval}")
    print(f"  Number of frames: {total_steps // output_interval}")
    print()

    vtk_files = []

    for step in range(output_interval, total_steps + 1, output_interval):
        output_file = str(output_path / f"flow_{step:04d}.vtk")

        result = cfd_python.run_simulation_with_params(
            nx=nx,
            ny=ny,
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
            steps=step,
            output_file=output_file,
        )

        max_vel = "N/A"
        if isinstance(result, dict) and "stats" in result:
            max_vel = f"{result['stats'].get('max_velocity', 0):.4f}"

        print(f"  Step {step:4d}: {output_file} (max_vel = {max_vel})")
        vtk_files.append(output_file)

    return vtk_files


def load_frames_data(vtk_files: list) -> list:
    """Load all VTK files and prepare frame data.

    Args:
        vtk_files: List of VTK file paths.

    Returns:
        List of (X, Y, u, v, p) tuples for animation.
    """
    frames_data = []

    for vtk_file in vtk_files:
        data = read_vtk_file(vtk_file)
        if data is None:
            print(f"Warning: Could not read {vtk_file}")
            continue

        X, Y = data.X, data.Y
        u, v = data.u, data.v
        p = data.get("p", np.zeros_like(u))

        frames_data.append((X, Y, u, v, p))

    return frames_data


def create_time_evolution_plot(vtk_files: list, time_indices: list):
    """Create a static plot showing time evolution snapshots.

    Args:
        vtk_files: List of VTK file paths.
        time_indices: List of time step values.
    """
    # Select subset of frames to show
    n_show = min(6, len(vtk_files))
    indices = np.linspace(0, len(vtk_files) - 1, n_show, dtype=int)

    fig, axes = plt.subplots(2, n_show, figsize=(4 * n_show, 8))

    for col, idx in enumerate(indices):
        data = read_vtk_file(vtk_files[idx])
        if data is None:
            continue

        X, Y = data.X, data.Y
        u, v = data.u, data.v
        dx, dy = data.dx, data.dy

        vel_mag = magnitude(u, v)
        omega = vorticity(u, v, dx, dy)

        # Row 1: Velocity with streamlines
        ax1 = axes[0, col]
        c1 = ax1.contourf(X, Y, vel_mag, levels=20, cmap="viridis", alpha=0.8)
        ax1.streamplot(X, Y, u, v, color="white", density=1.2, linewidth=0.6)
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_title(f"Step {time_indices[idx]}")
        ax1.set_aspect("equal")

        # Row 2: Vorticity
        ax2 = axes[1, col]
        vort_max = max(np.max(np.abs(omega)), 0.1)
        levels = np.linspace(-vort_max, vort_max, 21)
        c2 = ax2.contourf(X, Y, omega, levels=levels, cmap="RdBu_r")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_aspect("equal")

    # Add colorbars
    plt.colorbar(c1, ax=axes[0, :].tolist(), label="Velocity", shrink=0.8)
    plt.colorbar(c2, ax=axes[1, :].tolist(), label="Vorticity", shrink=0.8)

    plt.suptitle("Transient Flow Development: Lid-Driven Cavity", fontsize=14)
    plt.tight_layout()

    output_file = "transient_evolution.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_file}")

    plt.show()


def create_animations(frames_data: list, time_indices: list):
    """Create animated GIFs showing flow evolution.

    Args:
        frames_data: List of frame data tuples.
        time_indices: List of time step values.
    """
    print("\nCreating animations...")

    # Create AnimationFrames object
    animation_frames = create_animation_frames(frames_data, time_indices=time_indices)

    # 1. Velocity magnitude animation
    print("  Creating velocity animation...")
    fig1, anim1 = create_field_animation(
        animation_frames,
        field_name="velocity_mag",
        title_prefix="Velocity Magnitude Evolution",
    )
    save_animation(anim1, "transient_velocity.gif", fps=5)
    plt.close(fig1)
    print("  Saved: transient_velocity.gif")

    # 2. Vorticity animation
    print("  Creating vorticity animation...")
    fig2, anim2 = create_field_animation(
        animation_frames,
        field_name="vorticity",
        title_prefix="Vorticity Evolution",
        cmap="RdBu_r",
    )
    save_animation(anim2, "transient_vorticity.gif", fps=5)
    plt.close(fig2)
    print("  Saved: transient_vorticity.gif")

    # 3. Multi-panel animation
    print("  Creating multi-panel animation...")
    fig3, anim3 = create_multi_panel_animation(
        animation_frames,
        title="Transient Flow Development",
    )
    save_animation(anim3, "transient_multipanel.gif", fps=4)
    plt.close(fig3)
    print("  Saved: transient_multipanel.gif")


def main():
    """Main function."""
    print("Transient Flow Animation Example")
    print("=" * 40)
    print()

    # Run simulation with multiple snapshots
    vtk_files = run_transient_simulation(
        nx=80,
        ny=80,
        total_steps=500,
        output_interval=50,
        output_dir="transient_output",
    )

    # Extract time indices from filenames
    time_indices = [int(f.split("_")[-1].split(".")[0]) for f in vtk_files]

    # Create static time evolution plot
    print("\nCreating time evolution plot...")
    create_time_evolution_plot(vtk_files, time_indices)

    # Load frame data for animations
    print("\nLoading frame data...")
    frames_data = load_frames_data(vtk_files)
    print(f"Loaded {len(frames_data)} frames")

    # Create animations
    create_animations(frames_data, time_indices)

    print("\nDone!")
    print("\nOutput files:")
    print("  - transient_evolution.png (static snapshots)")
    print("  - transient_velocity.gif (velocity animation)")
    print("  - transient_vorticity.gif (vorticity animation)")
    print("  - transient_multipanel.gif (multi-panel animation)")


if __name__ == "__main__":
    main()
