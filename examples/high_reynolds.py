#!/usr/bin/env python3
"""
High Reynolds Number Flow
=========================

Simulation at higher Reynolds number to observe more complex
flow features like stronger vortices and potential instabilities.

Creates an animation showing the flow development over time.

Note: Higher Re requires finer mesh and more iterations for stability.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import DATA_DIR, ensure_dirs

try:
    import cfd_python
    CFD_AVAILABLE = True
except ImportError:
    CFD_AVAILABLE = False


def run_simulation():
    """Run high Reynolds number simulation with multiple outputs"""

    if not CFD_AVAILABLE:
        print("Error: cfd-python not available")
        return False

    ensure_dirs()
    cfd_python.set_output_dir(str(DATA_DIR))

    print("High Reynolds Number Flow")
    print("=" * 40)
    print()
    print("Re=1000 simulation - expect complex flow patterns.")
    print("Fine mesh required for stability.")
    print()

    # Simulation parameters
    nx, ny = 100, 100  # Fine mesh for stability
    total_steps = 1000
    output_interval = 100  # 10 frames

    available_solvers = cfd_python.list_solvers()
    solver = 'projection_optimized' if 'projection_optimized' in available_solvers else 'projection'

    print(f"Grid: {nx} x {ny}")
    print(f"Reynolds: 1000")
    print(f"Total steps: {total_steps}")
    print(f"Output interval: {output_interval} ({total_steps // output_interval} frames)")
    print(f"Solver: {solver}")
    print()

    print("Running simulation...")

    for step in range(output_interval, total_steps + 1, output_interval):
        output_file = str(DATA_DIR / f"high_re_{step:04d}.vtk")

        result = cfd_python.run_simulation_with_params(
            nx=nx,
            ny=ny,
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
            steps=step,
            solver_type=solver,
            output_file=output_file
        )

        print(f"  Step {step}: saved {os.path.basename(output_file)}")

    print()
    print("Simulation complete!")
    print(f"Generated {total_steps // output_interval} VTK files in {DATA_DIR}")

    return True


def create_animation():
    """Create animation from the generated VTK files"""
    print()
    print("Creating animation...")

    from visualize_cfd import create_animations
    import glob

    # Only animate files from this script (high_re_*.vtk)
    vtk_pattern = str(DATA_DIR / "high_re_*.vtk")
    vtk_files = sorted(glob.glob(vtk_pattern))

    if vtk_files:
        print(f"Found {len(vtk_files)} VTK files")
        create_animations(vtk_files, output_prefix='high_reynolds')
    else:
        print("No VTK files found!")


def main():
    ensure_dirs()

    success = run_simulation()

    if not success:
        print("Simulation failed!")
        return

    create_animation()

    print()
    print("Done! Check output/animations/ for the animated GIF.")


if __name__ == "__main__":
    main()
