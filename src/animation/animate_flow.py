#!/usr/bin/env python3
"""
CFD Flow Animation Tool
=======================

Creates animated visualizations from CFD simulation time series data.
Generates both MP4 videos and GIF animations of velocity fields, pressure,
and streamlines.

Requirements:
    pip install matplotlib numpy pillow

Usage:
    python animate_flow.py [animation_directory]
"""

import matplotlib
import numpy as np

matplotlib.use('Agg')  # Use non-interactive backend
import glob
import os
import re
import sys

import matplotlib.animation as animation
import matplotlib.pyplot as plt


def read_vtk_structured_points(filename):
    """Read VTK structured points file and extract data"""
    try:
        with open(filename) as file:
            lines = file.readlines()
    except FileNotFoundError:
        print(f"Error: File {filename} not found")
        return None

    # Parse header
    data = {}
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith('DIMENSIONS'):
            dims = list(map(int, line.split()[1:4]))
            data['dimensions'] = dims
            nx, ny, nz = dims

        elif line.startswith('ORIGIN'):
            origin = list(map(float, line.split()[1:4]))
            data['origin'] = origin

        elif line.startswith('SPACING'):
            spacing = list(map(float, line.split()[1:4]))
            data['spacing'] = spacing

        elif line.startswith('POINT_DATA'):
            num_points = int(line.split()[1])
            data['num_points'] = num_points

        elif line.startswith('VECTORS'):
            # Read vector data
            vector_name = line.split()[1]
            i += 1
            vectors = []
            for j in range(num_points):
                vector_line = lines[i + j].strip().split()
                try:
                    u, v, w = map(float, vector_line)
                    # Handle NaN values
                    if np.isnan(u) or np.isinf(u):
                        u = 0.0
                    if np.isnan(v) or np.isinf(v):
                        v = 0.0
                    if np.isnan(w) or np.isinf(w):
                        w = 0.0
                except ValueError:
                    # Handle invalid float values like "-nan(ind)"
                    u, v, w = 0.0, 0.0, 0.0
                vectors.append([u, v, w])
            data[f'vectors_{vector_name}'] = np.array(vectors)
            i += num_points - 1

        elif line.startswith('SCALARS'):
            # Read scalar data
            scalar_name = line.split()[1]
            i += 1  # Skip LOOKUP_TABLE line
            if lines[i].strip().startswith('LOOKUP_TABLE'):
                i += 1
            scalars = []
            for j in range(num_points):
                try:
                    scalar_value = float(lines[i + j].strip())
                    # Handle NaN values
                    if np.isnan(scalar_value) or np.isinf(scalar_value):
                        scalar_value = 0.0
                except ValueError:
                    # Handle invalid float values like "-nan(ind)"
                    scalar_value = 0.0
                scalars.append(scalar_value)
            data[f'scalars_{scalar_name}'] = np.array(scalars)
            i += num_points - 1

        i += 1

    # Create coordinate grids
    nx, ny = data['dimensions'][:2]
    origin = data['origin']
    spacing = data['spacing']

    x = np.linspace(origin[0], origin[0] + (nx-1)*spacing[0], nx)
    y = np.linspace(origin[1], origin[1] + (ny-1)*spacing[1], ny)
    data['x_coords'] = x
    data['y_coords'] = y
    data['X'], data['Y'] = np.meshgrid(x, y)

    return data

def reshape_field_data(data, field_name, nx, ny):
    """Reshape 1D field data to 2D grid"""
    if field_name in data:
        return data[field_name].reshape(ny, nx)
    return None

def create_flow_animation(animation_dir, output_filename="output/animations/flow_animation.mp4", fps=10):
    """Create animated visualization of flow evolution"""

    # Find all flow field files (try different patterns)
    patterns = [
        os.path.join(animation_dir, "flow_field_*.vtk"),
        os.path.join(animation_dir, "simple_flow_*.vtk")
    ]

    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))

    files = sorted(files)

    if not files:
        print(f"No flow field files found in {animation_dir}")
        return

    print(f"Found {len(files)} animation frames")

    # Extract time step numbers for sorting
    def extract_step(filename):
        # Try different patterns
        patterns = [r'flow_field_(\d+)\.vtk', r'simple_flow_(\d+)\.vtk']
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                return int(match.group(1))
        return 0

    files.sort(key=extract_step)

    # Read first frame to set up the plot
    first_data = read_vtk_structured_points(files[0])
    if first_data is None:
        return

    nx, ny = first_data['dimensions'][:2]

    # Set up the figure and subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('CFD Flow Animation', fontsize=16)

    # Calculate subsampling for vector plots
    subsample = max(1, min(nx, ny) // 15)  # Dynamic subsampling

    # Initialize empty plots
    im1 = ax1.contourf(first_data['X'], first_data['Y'], np.zeros((ny, nx)), levels=20, cmap='viridis')
    ax1.set_title('Velocity Magnitude')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.axis('equal')
    plt.colorbar(im1, ax=ax1)

    ax2.quiver(first_data['X'][::subsample, ::subsample], first_data['Y'][::subsample, ::subsample],
               np.zeros((ny//subsample, nx//subsample)), np.zeros((ny//subsample, nx//subsample)),
               scale_units='xy', angles='xy')
    ax2.set_title('Velocity Vectors')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.axis('equal')

    ax3.streamplot(first_data['X'], first_data['Y'],
                   np.zeros((ny, nx)), np.zeros((ny, nx)),
                   density=2, linewidth=1, color='blue')
    ax3.set_title('Flow Streamlines')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.axis('equal')

    im4 = ax4.contourf(first_data['X'], first_data['Y'], np.zeros((ny, nx)), levels=20, cmap='RdBu_r')
    ax4.set_title('Pressure Field')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.axis('equal')
    plt.colorbar(im4, ax=ax4)

    # Add time text
    time_text = fig.text(0.02, 0.95, '', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="white"))

    def animate_frame(frame_num):
        """Animation function for each frame"""
        if frame_num >= len(files):
            return

        filename = files[frame_num]
        data = read_vtk_structured_points(filename)

        if data is None:
            return

        # Extract step number for time display
        step = extract_step(filename)
        time_text.set_text(f'Time Step: {step}')

        # Clear previous plots
        ax1.clear()
        ax2.clear()
        ax3.clear()
        ax4.clear()

        # 1. Velocity magnitude
        if 'vectors_velocity' in data:
            vectors = data['vectors_velocity']
            u = vectors[:, 0].reshape(ny, nx)
            v = vectors[:, 1].reshape(ny, nx)
            velocity_mag = np.sqrt(u**2 + v**2)

            ax1.contourf(data['X'], data['Y'], velocity_mag, levels=20, cmap='viridis')
            ax1.set_title('Velocity Magnitude')
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.axis('equal')
            ax1.grid(True, alpha=0.3)

            # 2. Velocity vectors (subsampled)
            subsample = max(1, min(nx, ny) // 20)
            X_sub = data['X'][::subsample, ::subsample]
            Y_sub = data['Y'][::subsample, ::subsample]
            u_sub = u[::subsample, ::subsample]
            v_sub = v[::subsample, ::subsample]

            ax2.quiver(X_sub, Y_sub, u_sub, v_sub,
                      np.sqrt(u_sub**2 + v_sub**2),
                      cmap='plasma', scale_units='xy', angles='xy')
            ax2.set_title('Velocity Vectors')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.axis('equal')
            ax2.grid(True, alpha=0.3)

            # 3. Streamlines
            try:
                ax3.streamplot(data['X'], data['Y'], u, v,
                              color=velocity_mag, cmap='viridis',
                              density=2, linewidth=1)
            except ValueError:
                # If streamplot fails (e.g., all velocities zero), show velocity magnitude
                ax3.contourf(data['X'], data['Y'], velocity_mag, levels=20, cmap='viridis', alpha=0.7)

            ax3.set_title('Flow Streamlines')
            ax3.set_xlabel('X')
            ax3.set_ylabel('Y')
            ax3.axis('equal')
            ax3.grid(True, alpha=0.3)

        # 4. Pressure field
        if 'scalars_pressure' in data:
            pressure = reshape_field_data(data, 'scalars_pressure', nx, ny)
            ax4.contourf(data['X'], data['Y'], pressure, levels=20, cmap='RdBu_r')
            ax4.set_title('Pressure Field')
            ax4.set_xlabel('X')
            ax4.set_ylabel('Y')
            ax4.axis('equal')
            ax4.grid(True, alpha=0.3)

    # Create animation
    print(f"Creating animation with {len(files)} frames...")
    ani = animation.FuncAnimation(fig, animate_frame, frames=len(files),
                                 interval=1000//fps, repeat=True)

    # Save as MP4
    print(f"Saving animation as {output_filename}...")
    try:
        ani.save(output_filename, writer='pillow', fps=fps)
        print(f"Animation saved successfully: {output_filename}")
    except Exception as e:
        print(f"Error saving animation: {e}")
        # Try saving as GIF instead
        gif_filename = output_filename.replace('.mp4', '.gif')
        try:
            ani.save(gif_filename, writer='pillow', fps=fps)
            print(f"Animation saved as GIF: {gif_filename}")
        except Exception as e2:
            print(f"Error saving GIF: {e2}")

    plt.close(fig)

def create_individual_frames(animation_dir, output_dir="output/animations/animation_frames"):
    """Create individual PNG frames for each time step"""

    # Find all flow field files (try different patterns)
    patterns = [
        os.path.join(animation_dir, "flow_field_*.vtk"),
        os.path.join(animation_dir, "simple_flow_*.vtk")
    ]

    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))

    files = sorted(files)

    if not files:
        print(f"No flow field files found in {animation_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    print(f"Creating {len(files)} individual frames...")

    for i, filename in enumerate(files):
        data = read_vtk_structured_points(filename)
        if data is None:
            continue

        nx, ny = data['dimensions'][:2]

        # Extract step number
        patterns = [r'flow_field_(\d+)\.vtk', r'simple_flow_(\d+)\.vtk']
        step = 0
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                step = int(match.group(1))
                break

        # Create comprehensive plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'CFD Flow Visualization - Time Step {step}', fontsize=16)

        if 'vectors_velocity' in data:
            vectors = data['vectors_velocity']
            u = vectors[:, 0].reshape(ny, nx)
            v = vectors[:, 1].reshape(ny, nx)
            velocity_mag = np.sqrt(u**2 + v**2)

            # Velocity magnitude
            im1 = ax1.contourf(data['X'], data['Y'], velocity_mag, levels=20, cmap='viridis')
            ax1.set_title('Velocity Magnitude')
            ax1.axis('equal')
            plt.colorbar(im1, ax=ax1)

            # Velocity vectors
            subsample = max(1, min(nx, ny) // 20)
            X_sub = data['X'][::subsample, ::subsample]
            Y_sub = data['Y'][::subsample, ::subsample]
            u_sub = u[::subsample, ::subsample]
            v_sub = v[::subsample, ::subsample]

            ax2.quiver(X_sub, Y_sub, u_sub, v_sub,
                      np.sqrt(u_sub**2 + v_sub**2),
                      cmap='plasma', scale_units='xy', angles='xy')
            ax2.set_title('Velocity Vectors')
            ax2.axis('equal')

            # Streamlines
            try:
                ax3.streamplot(data['X'], data['Y'], u, v,
                              color=velocity_mag, cmap='viridis', density=2)
            except ValueError:
                ax3.contourf(data['X'], data['Y'], velocity_mag, levels=20, cmap='viridis')
            ax3.set_title('Flow Streamlines')
            ax3.axis('equal')

        # Pressure field
        if 'scalars_pressure' in data:
            pressure = reshape_field_data(data, 'scalars_pressure', nx, ny)
            im4 = ax4.contourf(data['X'], data['Y'], pressure, levels=20, cmap='RdBu_r')
            ax4.set_title('Pressure Field')
            ax4.axis('equal')
            plt.colorbar(im4, ax=ax4)

        # Save frame
        frame_filename = os.path.join(output_dir, f"frame_{step:04d}.png")
        plt.savefig(frame_filename, dpi=150, bbox_inches='tight')
        plt.close(fig)

        if (i + 1) % 10 == 0:
            print(f"Created {i + 1}/{len(files)} frames")

    print(f"All frames saved to: {output_dir}")

def main():
    """Main function"""
    if len(sys.argv) > 1:
        animation_dir = sys.argv[1]
    else:
        animation_dir = "output/animation"

    if not os.path.exists(animation_dir):
        print(f"Animation directory not found: {animation_dir}")
        print("Please run the animated_flow_simulation first!")
        return

    print("CFD Flow Animation Tool")
    print("======================")
    print(f"Animation directory: {animation_dir}")

    # Create output directory
    os.makedirs("output/animations", exist_ok=True)

    # Create animated MP4/GIF
    print("\n1. Creating animated visualization...")
    create_flow_animation(animation_dir, "output/animations/flow_animation.mp4", fps=8)

    # Create individual frames
    print("\n2. Creating individual frames...")
    create_individual_frames(animation_dir, "animation_frames")

    print("\nAnimation creation complete!")
    print("Generated files:")
    print("  - flow_animation.mp4 (or .gif)")
    print("  - animation_frames/ directory with PNG frames")

if __name__ == "__main__":
    main()
