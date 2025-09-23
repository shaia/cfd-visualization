#!/usr/bin/env python3
"""
Simple animation script without complex color scaling issues
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob
import os
import re

def read_vtk_structured_points(filename):
    """Read VTK structured points file and extract data"""
    try:
        with open(filename, 'r') as file:
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
                    if np.isnan(u) or np.isinf(u): u = 0.0
                    if np.isnan(v) or np.isinf(v): v = 0.0
                    if np.isnan(w) or np.isinf(w): w = 0.0
                except ValueError:
                    u, v, w = 0.0, 0.0, 0.0
                vectors.append([u, v, w])
            data[f'vectors_{vector_name}'] = np.array(vectors)
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

def main():
    # Find all VTK files
    pattern = "output/animations/simple_flow_*.vtk"
    files = sorted(glob.glob(pattern))

    if not files:
        print("No animation files found!")
        return

    print(f"Found {len(files)} files")

    # Read all data first to get global min/max for consistent scaling
    all_velocity_mags = []
    all_data = []

    for filename in files:
        data = read_vtk_structured_points(filename)
        if data is None:
            continue

        if 'vectors_velocity' in data:
            nx, ny = data['dimensions'][:2]
            vectors = data['vectors_velocity']
            u = vectors[:, 0].reshape(ny, nx)
            v = vectors[:, 1].reshape(ny, nx)
            velocity_mag = np.sqrt(u**2 + v**2)
            all_velocity_mags.append(velocity_mag)
            all_data.append((data, u, v, velocity_mag))

    if not all_data:
        print("No valid data found!")
        return

    # Calculate global min/max for consistent color scaling
    global_vmin = min(vmag.min() for vmag in all_velocity_mags)
    global_vmax = max(vmag.max() for vmag in all_velocity_mags)

    print(f"Global velocity magnitude range: {global_vmin:.3f} to {global_vmax:.3f}")

    # Create animation
    fig, ax = plt.subplots(figsize=(10, 6))

    def animate(frame):
        ax.clear()

        if frame >= len(all_data):
            return

        data, u, v, velocity_mag = all_data[frame]
        step = int(re.search(r'simple_flow_(\d+)\.vtk', files[frame]).group(1))

        # Plot with consistent color scaling
        im = ax.contourf(data['X'], data['Y'], velocity_mag,
                        levels=20, cmap='viridis', vmin=global_vmin, vmax=global_vmax)

        # Add velocity vectors (subsampled)
        subsample = max(1, min(data['dimensions'][0], data['dimensions'][1]) // 15)
        X_sub = data['X'][::subsample, ::subsample]
        Y_sub = data['Y'][::subsample, ::subsample]
        u_sub = u[::subsample, ::subsample]
        v_sub = v[::subsample, ::subsample]

        ax.quiver(X_sub, Y_sub, u_sub, v_sub, scale_units='xy', angles='xy', alpha=0.7)

        ax.set_title(f'CFD Flow Animation - Step {step} (Max Vel: {velocity_mag.max():.3f})')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.axis('equal')

        return [im]

    print("Creating animation...")
    ani = animation.FuncAnimation(fig, animate, frames=len(all_data),
                                 interval=200, blit=False, repeat=True)

    # Save as GIF
    output_file = "output/animations/simple_flow_animation.gif"
    print(f"Saving animation to {output_file}...")
    ani.save(output_file, writer='pillow', fps=5)
    print("Animation saved successfully!")

    plt.close()

if __name__ == "__main__":
    main()