#!/usr/bin/env python3
"""
Quick test to generate individual frames from the animation data
"""
import matplotlib
import numpy as np

matplotlib.use('Agg')  # Use non-interactive backend
import os

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

def create_test_frame(filename, output_name):
    """Create a single test frame"""
    data = read_vtk_structured_points(filename)
    if data is None:
        return

    nx, ny = data['dimensions'][:2]

    # Create plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(f'Test Frame: {os.path.basename(filename)}', fontsize=14)

    if 'vectors_velocity' in data:
        vectors = data['vectors_velocity']
        u = vectors[:, 0].reshape(ny, nx)
        v = vectors[:, 1].reshape(ny, nx)
        velocity_mag = np.sqrt(u**2 + v**2)

        # 1. Velocity magnitude
        im1 = ax1.contourf(data['X'], data['Y'], velocity_mag, levels=20, cmap='viridis')
        ax1.set_title(f'Velocity Magnitude (max={velocity_mag.max():.3f})')
        ax1.axis('equal')
        plt.colorbar(im1, ax=ax1)

        # 2. Velocity vectors (subsampled)
        subsample = max(1, min(nx, ny) // 10)
        X_sub = data['X'][::subsample, ::subsample]
        Y_sub = data['Y'][::subsample, ::subsample]
        u_sub = u[::subsample, ::subsample]
        v_sub = v[::subsample, ::subsample]

        ax2.quiver(X_sub, Y_sub, u_sub, v_sub, scale_units='xy', angles='xy')
        ax2.set_title(f'Velocity Vectors (subsampled {subsample})')
        ax2.axis('equal')

        # 3. U component
        im3 = ax3.contourf(data['X'], data['Y'], u, levels=20, cmap='RdBu_r')
        ax3.set_title(f'U Velocity (range: {u.min():.3f} to {u.max():.3f})')
        ax3.axis('equal')
        plt.colorbar(im3, ax=ax3)

        # 4. V component
        im4 = ax4.contourf(data['X'], data['Y'], v, levels=20, cmap='RdBu_r')
        ax4.set_title(f'V Velocity (range: {v.min():.3f} to {v.max():.3f})')
        ax4.axis('equal')
        plt.colorbar(im4, ax=ax4)

    plt.tight_layout()
    plt.savefig(output_name, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Created test frame: {output_name}")

def main():
    # Test a few frames
    test_files = [
        "output/vtk_files/simple_flow_0000.vtk",
        "output/vtk_files/simple_flow_0020.vtk",
        "output/vtk_files/simple_flow_0050.vtk",
        "output/vtk_files/simple_flow_0100.vtk"
    ]

    os.makedirs("test_frames", exist_ok=True)

    for i, filename in enumerate(test_files):
        if os.path.exists(filename):
            output_name = f"test_frames/test_frame_{i:02d}.png"
            create_test_frame(filename, output_name)
        else:
            print(f"File not found: {filename}")

if __name__ == "__main__":
    main()
