#!/usr/bin/env python3
"""3D Rotating Surface Animation Example.

This example demonstrates how to create a rotating 3D surface animation
of CFD simulation results using matplotlib, exported as a GIF.

Features demonstrated:
1. Transient simulation with multiple timesteps
2. 3D surface plot with matplotlib's mplot3d
3. Rotating camera animation across frames
4. Contour projection on the base plane
5. GIF export

Usage:
    python examples/3d_surface_animation.py

Requirements:
    - cfd_python package (pip install -e ../cfd-python)
    - cfd_viz package
    - pillow package (for GIF export)
"""

import sys
from pathlib import Path

try:
    import cfd_python
except ImportError:
    print("Error: cfd_python package not installed.")
    print("Install with: pip install -e ../cfd-python")
    sys.exit(1)

from cfd_viz.animation import create_3d_surface_animation, create_animation_frames
from cfd_viz.common import read_vtk_file


def run_transient_simulations() -> list:
    """Run simulations at multiple timesteps.

    Returns:
        List of output VTK file paths.
    """
    output_dir = Path("output/vtk/3d_transient")
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


def create_animation(vtk_files: list):
    """Create a rotating 3D surface animation from transient results.

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

        X, Y = data.X, data.Y
        u, v = data.u, data.v
        p = data.get("p", None)

        frames_data.append((X, Y, u, v, p))
        time_idx = int(vtk_file.split("_")[-1].split(".")[0])
        time_indices.append(time_idx)

    print(f"Loaded {len(frames_data)} frames")

    # Build AnimationFrames (computes velocity_mag, vorticity, etc.)
    animation_frames = create_animation_frames(
        frames_data,
        time_indices=time_indices,
        compute_derived=True,
    )

    # Create 3D rotating surface animation
    print("\nCreating 3D rotating surface animation...")
    _fig, anim = create_3d_surface_animation(
        animation_frames,
        field_name="velocity_mag",
        figsize=(12, 8),
        interval=200,
        rotate_camera=True,
        title_prefix="Velocity Magnitude",
    )

    output_dir = Path("output/animations")
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / "3d_surface_rotation.gif"
    print(f"Saving GIF to: {out_path}")
    anim.save(str(out_path), writer="pillow", fps=5)
    print(f"  Saved: {out_path}")


def main():
    """Main function."""
    print("3D Surface Animation Example")
    print("=" * 40)

    vtk_files = run_transient_simulations()
    create_animation(vtk_files)

    print("\n" + "=" * 40)
    print("Done!")
    print("\nOutput file:")
    print("  - output/animations/3d_surface_rotation.gif")
    print("\nThe animation shows:")
    print("  - Velocity magnitude as 3D surface height")
    print("  - Rotating camera view across frames")
    print("  - Contour projection on the base plane")
    print("  - Flow evolution over simulation timesteps")


if __name__ == "__main__":
    main()
