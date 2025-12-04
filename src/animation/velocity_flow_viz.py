#!/usr/bin/env python3
"""
Advanced Velocity and Flow Visualization for CFD Framework
=========================================================

This script provides comprehensive visualization of velocity fields including:
- Velocity magnitude contours
- Velocity vector fields
- Streamlines
- Combined flow field plots

Requirements:
    pip install matplotlib numpy vtk

Usage:
    python velocity_flow_viz.py [vtk_file]
"""

import matplotlib
import numpy as np

matplotlib.use('Agg')  # Use non-interactive backend
import glob
import os
import sys

import matplotlib.pyplot as plt


def read_vtk_structured_points(filename):
    """Read VTK structured points file and extract data"""
    print(f"Reading VTK file: {filename}")

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

    print(f"Grid: {nx} x {ny}, Domain: [{origin[0]:.2f}, {origin[0] + (nx-1)*spacing[0]:.2f}] x [{origin[1]:.2f}, {origin[1] + (ny-1)*spacing[1]:.2f}]")

    return data

def reshape_field_data(data, field_name, nx, ny):
    """Reshape 1D field data to 2D grid"""
    if field_name in data:
        return data[field_name].reshape(ny, nx)
    return None

def plot_velocity_magnitude(data, save_path=None):
    """Plot velocity magnitude contours"""
    nx, ny = data['dimensions'][:2]

    # Try to find velocity magnitude data
    velocity_mag = None
    if 'scalars_velocity_magnitude' in data:
        velocity_mag = reshape_field_data(data, 'scalars_velocity_magnitude', nx, ny)
    elif 'vectors_velocity' in data:
        # Calculate magnitude from vectors
        vectors = data['vectors_velocity']
        u = vectors[:, 0].reshape(ny, nx)
        v = vectors[:, 1].reshape(ny, nx)
        velocity_mag = np.sqrt(u**2 + v**2)

    if velocity_mag is None:
        print("No velocity magnitude data found")
        return

    plt.figure(figsize=(12, 6))

    # Velocity magnitude contour plot
    plt.contourf(data['X'], data['Y'], velocity_mag, levels=20, cmap='viridis')
    plt.colorbar(label='Velocity Magnitude')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Velocity Magnitude Contours')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Velocity magnitude plot saved to: {save_path}")
    plt.close()

def plot_velocity_vectors(data, save_path=None, subsample=3):
    """Plot velocity vector field"""
    nx, ny = data['dimensions'][:2]

    if 'vectors_velocity' not in data:
        print("No velocity vector data found")
        return

    vectors = data['vectors_velocity']
    u = vectors[:, 0].reshape(ny, nx)
    v = vectors[:, 1].reshape(ny, nx)

    # Subsample for cleaner vector display
    X_sub = data['X'][::subsample, ::subsample]
    Y_sub = data['Y'][::subsample, ::subsample]
    u_sub = u[::subsample, ::subsample]
    v_sub = v[::subsample, ::subsample]

    plt.figure(figsize=(12, 6))

    # Vector field plot
    plt.quiver(X_sub, Y_sub, u_sub, v_sub,
               np.sqrt(u_sub**2 + v_sub**2),
               cmap='plasma', scale_units='xy', angles='xy')
    plt.colorbar(label='Velocity Magnitude')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Velocity Vector Field')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Velocity vectors plot saved to: {save_path}")
    plt.close()

def plot_streamlines(data, save_path=None):
    """Plot streamlines of the flow"""
    nx, ny = data['dimensions'][:2]

    if 'vectors_velocity' not in data:
        print("No velocity vector data found")
        return

    vectors = data['vectors_velocity']
    u = vectors[:, 0].reshape(ny, nx)
    v = vectors[:, 1].reshape(ny, nx)

    plt.figure(figsize=(12, 6))

    # Streamlines
    plt.streamplot(data['X'], data['Y'], u, v,
                   color=np.sqrt(u**2 + v**2),
                   cmap='viridis', density=2, linewidth=1)
    plt.colorbar(label='Velocity Magnitude')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Flow Streamlines')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Streamlines plot saved to: {save_path}")
    plt.close()

def plot_comprehensive_flow(data, save_path=None):
    """Create comprehensive flow visualization with multiple subplots"""
    nx, ny = data['dimensions'][:2]

    if 'vectors_velocity' not in data:
        print("No velocity vector data found")
        return

    vectors = data['vectors_velocity']
    u = vectors[:, 0].reshape(ny, nx)
    v = vectors[:, 1].reshape(ny, nx)
    velocity_mag = np.sqrt(u**2 + v**2)

    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Velocity magnitude contours
    im1 = ax1.contourf(data['X'], data['Y'], velocity_mag, levels=20, cmap='viridis')
    ax1.set_title('Velocity Magnitude Contours')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(im1, ax=ax1, label='Velocity Magnitude')

    # 2. Velocity vectors (subsampled)
    subsample = max(1, min(nx, ny) // 20)
    X_sub = data['X'][::subsample, ::subsample]
    Y_sub = data['Y'][::subsample, ::subsample]
    u_sub = u[::subsample, ::subsample]
    v_sub = v[::subsample, ::subsample]

    im2 = ax2.quiver(X_sub, Y_sub, u_sub, v_sub,
                     np.sqrt(u_sub**2 + v_sub**2),
                     cmap='plasma', scale_units='xy', angles='xy')
    ax2.set_title('Velocity Vector Field')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.axis('equal')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(im2, ax=ax2, label='Velocity Magnitude')

    # 3. Streamlines
    ax3.streamplot(data['X'], data['Y'], u, v,
                   color=velocity_mag, cmap='viridis', density=2, linewidth=1)
    ax3.set_title('Flow Streamlines')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.axis('equal')
    ax3.grid(True, alpha=0.3)

    # 4. Pressure (if available) or combined view
    if 'scalars_pressure' in data:
        pressure = reshape_field_data(data, 'scalars_pressure', nx, ny)
        im4 = ax4.contourf(data['X'], data['Y'], pressure, levels=20, cmap='RdBu_r')
        ax4.set_title('Pressure Field')
        plt.colorbar(im4, ax=ax4, label='Pressure')
    else:
        # Combined velocity magnitude + streamlines
        im4 = ax4.contourf(data['X'], data['Y'], velocity_mag, levels=20, cmap='viridis', alpha=0.7)
        ax4.streamplot(data['X'], data['Y'], u, v, color='white', density=1, linewidth=0.8)
        ax4.set_title('Combined: Magnitude + Streamlines')
        plt.colorbar(im4, ax=ax4, label='Velocity Magnitude')

    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.axis('equal')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comprehensive flow plot saved to: {save_path}")
    plt.close()

def visualize_flow_evolution(pattern="output/flow_field_*.vtk"):
    """Visualize evolution of flow field over time"""
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No files found matching pattern: {pattern}")
        return

    print(f"Found {len(files)} flow field files")

    # Create animation-like sequence
    for filename in files[::2]:  # Every other file to reduce output
        print(f"Processing {filename}...")
        data = read_vtk_structured_points(filename)

        if data is None:
            continue

        # Extract step number from filename
        step = os.path.basename(filename).split('_')[-1].split('.')[0]
        save_path = f"output/flow_evolution_step_{step}.png"

        plot_comprehensive_flow(data, save_path)

def main():
    """Main function"""
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        # Look for VTK files in output directory
        vtk_files = glob.glob("output/*.vtk")
        if not vtk_files:
            print("No VTK files found in output directory")
            print("Usage: python velocity_flow_viz.py [vtk_file]")
            return

        # Find a flow field file if available, otherwise use the first file
        flow_files = [f for f in vtk_files if 'flow_field' in f]
        if flow_files:
            filename = flow_files[-1]  # Use latest flow field file
        else:
            filename = vtk_files[-1]  # Use latest VTK file

    print("CFD Framework - Advanced Velocity Flow Visualization")
    print("====================================================")

    # Read and visualize the data
    data = read_vtk_structured_points(filename)

    if data is None:
        return

    print("\nAvailable data fields:")
    for key in data.keys():
        if key.startswith('scalars_') or key.startswith('vectors_'):
            print(f"  - {key}")

    # Create output directory
    os.makedirs("output", exist_ok=True)

    # Generate visualizations
    print("\nGenerating velocity visualizations...")

    base_name = os.path.splitext(os.path.basename(filename))[0]

    plot_velocity_magnitude(data, f"output/{base_name}_velocity_magnitude.png")
    plot_velocity_vectors(data, f"output/{base_name}_velocity_vectors.png")
    plot_streamlines(data, f"output/{base_name}_streamlines.png")
    plot_comprehensive_flow(data, f"output/{base_name}_comprehensive_flow.png")

    print("\nVisualization complete!")
    print("Generated files:")
    print(f"  - {base_name}_velocity_magnitude.png")
    print(f"  - {base_name}_velocity_vectors.png")
    print(f"  - {base_name}_streamlines.png")
    print(f"  - {base_name}_comprehensive_flow.png")

if __name__ == "__main__":
    main()
