#!/usr/bin/env python3
"""
Lid-Driven Cavity with Vector Visualization
============================================

Runs a lid-driven cavity simulation and creates an animated
visualization showing velocity vectors (arrows) that evolve over time.

The top wall moves at constant velocity while other walls are stationary.
Arrows show both the direction and magnitude of the flow field.

Output: lid_driven_cavity_vectors.gif
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import ANIMATIONS_DIR, DATA_DIR, ensure_dirs
from vtk_reader import read_vtk_velocity

try:
    import cfd_python
    CFD_AVAILABLE = True
except ImportError:
    CFD_AVAILABLE = False

import glob

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


def run_simulation() -> bool:
    """Run simulation with multiple outputs for animation."""
    if not CFD_AVAILABLE:
        print("Error: cfd-python not available")
        return False

    ensure_dirs()
    cfd_python.set_output_dir(str(DATA_DIR))

    print("Vector Field Animation")
    print("=" * 40)
    print()
    print("Generating velocity vector animation.")
    print()

    # Simulation parameters - smaller grid for clearer arrows
    nx, ny = 40, 40
    total_steps = 400
    output_interval = 40  # 10 frames

    available_solvers = cfd_python.list_solvers()
    solver = 'projection' if 'projection' in available_solvers else available_solvers[0]

    print(f"Grid: {nx} x {ny}")
    print(f"Total steps: {total_steps}")
    print(f"Frames: {total_steps // output_interval}")
    print(f"Solver: {solver}")
    print()

    print("Running simulation...")

    for step in range(output_interval, total_steps + 1, output_interval):
        output_file = str(DATA_DIR / f"lid_cavity_vec_{step:04d}.vtk")

        cfd_python.run_simulation_with_params(
            nx=nx,
            ny=ny,
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
            steps=step,
            solver_type=solver,
            output_file=output_file,
        )

        print(f"  Step {step}: saved {os.path.basename(output_file)}")

    print()
    print("Simulation complete!")

    return True


def create_vector_animation():
    """Create animated quiver plot showing velocity vectors"""
    print()
    print("Creating vector field animation...")

    # Find VTK files from this script
    vtk_pattern = str(DATA_DIR / "lid_cavity_vec_*.vtk")
    vtk_files = sorted(glob.glob(vtk_pattern))

    if not vtk_files:
        print("No VTK files found!")
        return None

    print(f"Found {len(vtk_files)} frames")

    # Read all frames
    frames_data = []
    times = []
    X, Y = None, None

    for filename in vtk_files:
        X, Y, u, v = read_vtk_velocity(filename)
        if u is not None and v is not None:
            frames_data.append((u, v))
            try:
                step = int(os.path.basename(filename).split("_")[-1].split(".")[0])
            except ValueError:
                step = len(times)
            times.append(step)

    if not frames_data:
        print("No valid velocity data found!")
        return None

    # Set up figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Subsample for clearer arrows
    skip = 2
    X_sub = X[::skip, ::skip]
    Y_sub = Y[::skip, ::skip]

    # Initial frame
    u0, v0 = frames_data[0]
    u_sub = u0[::skip, ::skip]
    v_sub = v0[::skip, ::skip]
    speed = np.sqrt(u_sub**2 + v_sub**2)

    # Create quiver plot
    Q = ax.quiver(X_sub, Y_sub, u_sub, v_sub, speed,
                  cmap='plasma', scale=None, angles='xy')

    # Add colorbar
    plt.colorbar(Q, ax=ax, label="Velocity Magnitude")

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Vector Field Animation')
    ax.set_aspect('equal')
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())

    # Time text
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                        fontsize=12, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    def animate(frame):
        u, v = frames_data[frame]
        u_sub = u[::skip, ::skip]
        v_sub = v[::skip, ::skip]
        speed = np.sqrt(u_sub**2 + v_sub**2)

        Q.set_UVC(u_sub, v_sub, speed)
        time_text.set_text(f'Step: {times[frame]}')
        return Q, time_text

    anim = animation.FuncAnimation(fig, animate, frames=len(frames_data),
                                   interval=300, blit=False, repeat=True)

    output_file = str(ANIMATIONS_DIR / 'lid_driven_cavity_vectors.gif')
    print(f"Saving animation to {output_file}...")
    anim.save(output_file, writer='pillow', fps=4)
    print("Animation saved!")

    plt.close()
    return output_file


def main():
    ensure_dirs()

    success = run_simulation()

    if not success:
        print("Simulation failed!")
        return

    output_file = create_vector_animation()

    if output_file:
        print()
        print(f"Done! Vector animation saved to: {output_file}")


if __name__ == "__main__":
    main()
