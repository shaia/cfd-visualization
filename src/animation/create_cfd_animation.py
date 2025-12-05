#!/usr/bin/env python3
"""
Animation script for the CFD physics-based simulation with proper color scaling
"""
import matplotlib
import numpy as np

matplotlib.use('Agg')
import glob
import re

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
                    if np.isnan(u) or np.isinf(u):
                        u = 0.0
                    if np.isnan(v) or np.isinf(v):
                        v = 0.0
                    if np.isnan(w) or np.isinf(w):
                        w = 0.0
                except ValueError:
                    u, v, w = 0.0, 0.0, 0.0
                vectors.append([u, v, w])
            data[f'vectors_{vector_name}'] = np.array(vectors)
            i += num_points - 1

        elif line.startswith('SCALARS'):
            # Read scalar data
            scalar_name = line.split()[1]
            i += 1  # Skip LOOKUP_TABLE line
            if i < len(lines) and lines[i].strip().startswith('LOOKUP_TABLE'):
                i += 1
            scalars = []
            for j in range(num_points):
                try:
                    scalar_value = float(lines[i + j].strip())
                    if np.isnan(scalar_value) or np.isinf(scalar_value):
                        scalar_value = 0.0
                except ValueError:
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

def main():
    # Find all CFD VTK files
    pattern = "build/Debug/output/animation/flow_field_*.vtk"
    files = sorted(glob.glob(pattern))

    if not files:
        print("No CFD animation files found!")
        return

    print(f"Found {len(files)} CFD files")

    # Read all data first to get global min/max for consistent scaling
    all_velocity_mags = []
    all_pressure = []
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

            pressure = None
            if 'scalars_pressure' in data:
                pressure = data['scalars_pressure'].reshape(ny, nx)
                all_pressure.append(pressure)

            all_velocity_mags.append(velocity_mag)
            all_data.append((data, u, v, velocity_mag, pressure))

    if not all_data:
        print("No valid CFD data found!")
        return

    # Calculate global min/max for consistent color scaling
    global_vmin = min(vmag.min() for vmag in all_velocity_mags)
    global_vmax = max(vmag.max() for vmag in all_velocity_mags)

    global_pmin = min(p.min() for p in all_pressure) if all_pressure else 0
    global_pmax = max(p.max() for p in all_pressure) if all_pressure else 1

    print(f"Global velocity magnitude range: {global_vmin:.3f} to {global_vmax:.3f}")
    print(f"Global pressure range: {global_pmin:.3f} to {global_pmax:.3f}")

    # Create animation with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('CFD Physics-Based Flow Animation', fontsize=16)

    def animate(frame):
        for ax in [ax1, ax2, ax3, ax4]:
            ax.clear()

        if frame >= len(all_data):
            return

        data, u, v, velocity_mag, pressure = all_data[frame]
        step = int(re.search(r'flow_field_(\d+)\.vtk', files[frame]).group(1))

        # 1. Velocity magnitude with consistent scaling
        ax1.contourf(data['X'], data['Y'], velocity_mag,
                     levels=20, cmap='viridis', vmin=global_vmin, vmax=global_vmax)
        ax1.set_title(f'Velocity Magnitude - Step {step}')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.axis('equal')

        # 2. Velocity vectors (subsampled)
        subsample = max(1, min(data['dimensions'][0], data['dimensions'][1]) // 15)
        X_sub = data['X'][::subsample, ::subsample]
        Y_sub = data['Y'][::subsample, ::subsample]
        u_sub = u[::subsample, ::subsample]
        v_sub = v[::subsample, ::subsample]

        # Only plot vectors if there's significant flow
        if velocity_mag.max() > 1e-6:
            ax2.quiver(X_sub, Y_sub, u_sub, v_sub, scale_units='xy', angles='xy', alpha=0.7)
        ax2.set_title(f'Velocity Vectors (Max: {velocity_mag.max():.3f})')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.axis('equal')

        # 3. Streamlines
        try:
            if velocity_mag.max() > 1e-6:
                ax3.streamplot(data['X'], data['Y'], u, v,
                              color=velocity_mag, cmap='viridis',
                              density=1.5, linewidth=1)
            else:
                ax3.contourf(data['X'], data['Y'], velocity_mag, levels=20, cmap='viridis')
        except ValueError:
            ax3.contourf(data['X'], data['Y'], velocity_mag, levels=20, cmap='viridis')
        ax3.set_title('Flow Streamlines')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.axis('equal')

        # 4. Pressure field
        if pressure is not None:
            ax4.contourf(data['X'], data['Y'], pressure,
                         levels=20, cmap='RdBu_r', vmin=global_pmin, vmax=global_pmax)
            ax4.set_title(f'Pressure Field - Step {step}')
        else:
            # Fallback to velocity magnitude if no pressure data
            ax4.contourf(data['X'], data['Y'], velocity_mag,
                         levels=20, cmap='plasma', vmin=global_vmin, vmax=global_vmax)
            ax4.set_title(f'Velocity (Alternative) - Step {step}')
        ax4.set_xlabel('X')
        ax4.set_ylabel('Y')
        ax4.axis('equal')

        return []

    print("Creating CFD animation...")
    ani = animation.FuncAnimation(fig, animate, frames=len(all_data),
                                 interval=300, blit=False, repeat=True)

    # Save as GIF
    output_file = "output/animations/cfd_physics_animation.gif"
    print(f"Saving animation to {output_file}...")
    ani.save(output_file, writer='pillow', fps=4)
    print("CFD animation saved successfully!")

    plt.close()

if __name__ == "__main__":
    main()
