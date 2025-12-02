#!/usr/bin/env python3
"""
Vorticity and Circulation Analysis Tool for CFD Framework
========================================================

This script provides comprehensive vorticity analysis including:
- Vorticity field calculation and visualization
- Circulation analysis around regions
- Vortex core detection
- Q-criterion visualization

Requirements:
    numpy, matplotlib, scipy

Usage:
    python vorticity_visualizer.py [vtk_file] [options]
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize, LinearSegmentedColormap
from scipy import ndimage
import argparse
import glob
import os
import sys

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, PLOTS_DIR, find_vtk_files, ensure_dirs

def read_vtk_file(filename):
    """Read a VTK structured points file and extract velocity data"""
    print(f"Reading VTK file: {filename}")

    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: File {filename} not found")
        return None

    # Parse header
    dimensions = None
    origin = None
    spacing = None
    data_start = None

    for i, line in enumerate(lines):
        if line.startswith('DIMENSIONS'):
            dimensions = [int(x) for x in line.split()[1:4]]
        elif line.startswith('ORIGIN'):
            origin = [float(x) for x in line.split()[1:4]]
        elif line.startswith('SPACING'):
            spacing = [float(x) for x in line.split()[1:4]]
        elif line.startswith('POINT_DATA'):
            data_start = i + 1
            break

    if not all([dimensions, origin, spacing, data_start]):
        raise ValueError(f"Invalid VTK file format: {filename}")

    nx, ny = dimensions[0], dimensions[1]

    # Create coordinate arrays
    x = np.linspace(origin[0], origin[0] + (nx-1)*spacing[0], nx)
    y = np.linspace(origin[1], origin[1] + (ny-1)*spacing[1], ny)

    # Parse velocity data
    u_data = None
    v_data = None

    i = data_start
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith('SCALARS u_velocity'):
            i += 2  # Skip LOOKUP_TABLE line
            u_values = []
            while i < len(lines) and len(u_values) < nx * ny:
                values = [float(x) for x in lines[i].split()]
                u_values.extend(values)
                i += 1
            u_data = np.array(u_values).reshape((ny, nx))

        elif line.startswith('SCALARS v_velocity'):
            i += 2  # Skip LOOKUP_TABLE line
            v_values = []
            while i < len(lines) and len(v_values) < nx * ny:
                values = [float(x) for x in lines[i].split()]
                v_values.extend(values)
                i += 1
            v_data = np.array(v_values).reshape((ny, nx))
        else:
            i += 1

    if u_data is None or v_data is None:
        print("Error: Could not find velocity data in VTK file")
        return None

    return {
        'x': x, 'y': y, 'u': u_data, 'v': v_data,
        'nx': nx, 'ny': ny, 'dx': spacing[0], 'dy': spacing[1]
    }

def calculate_vorticity(u, v, dx, dy):
    """Calculate vorticity field (curl of velocity)"""
    # Calculate derivatives using central differences
    du_dy = np.gradient(u, dy, axis=0)
    dv_dx = np.gradient(v, dx, axis=1)

    # Vorticity = dv/dx - du/dy
    vorticity = dv_dx - du_dy

    return vorticity

def calculate_q_criterion(u, v, dx, dy):
    """Calculate Q-criterion for vortex identification"""
    # Calculate velocity gradients
    du_dx = np.gradient(u, dx, axis=1)
    du_dy = np.gradient(u, dy, axis=0)
    dv_dx = np.gradient(v, dx, axis=1)
    dv_dy = np.gradient(v, dy, axis=0)

    # Strain rate tensor components
    S11 = du_dx
    S12 = 0.5 * (du_dy + dv_dx)
    S22 = dv_dy

    # Rotation rate tensor components
    O12 = 0.5 * (du_dy - dv_dx)

    # Q-criterion = 0.5 * (||Ω||² - ||S||²)
    omega_squared = 2 * O12**2
    strain_squared = 2 * (S11**2 + 2 * S12**2 + S22**2)

    Q = 0.5 * (omega_squared - strain_squared)

    return Q

def detect_vortex_cores(vorticity, Q, threshold_factor=0.1):
    """Detect vortex cores using combined vorticity and Q-criterion"""
    # Normalize Q-criterion
    Q_normalized = Q / np.max(np.abs(Q))

    # Find regions where Q > threshold and |vorticity| is significant
    vorticity_threshold = threshold_factor * np.max(np.abs(vorticity))
    Q_threshold = threshold_factor

    vortex_cores = (Q_normalized > Q_threshold) & (np.abs(vorticity) > vorticity_threshold)

    return vortex_cores

def calculate_circulation(u, v, x, y, center, radius):
    """Calculate circulation around a circular path"""
    # Create circular path
    theta = np.linspace(0, 2*np.pi, 100)
    path_x = center[0] + radius * np.cos(theta)
    path_y = center[1] + radius * np.sin(theta)

    # Interpolate velocity at path points
    from scipy.interpolate import RegularGridInterpolator

    # Create interpolators
    interp_u = RegularGridInterpolator((y, x), u, bounds_error=False, fill_value=0)
    interp_v = RegularGridInterpolator((y, x), v, bounds_error=False, fill_value=0)

    # Get velocity along path
    path_points = np.column_stack([path_y, path_x])
    u_path = interp_u(path_points)
    v_path = interp_v(path_points)

    # Calculate tangent vectors
    dx_dt = np.gradient(path_x)
    dy_dt = np.gradient(path_y)

    # Calculate circulation = ∮ v⃗ · dl⃗
    circulation = np.trapz(u_path * dx_dt + v_path * dy_dt)

    return circulation

def create_vorticity_visualization(data, output_dir="visualization_output"):
    """Create comprehensive vorticity visualization"""
    os.makedirs(output_dir, exist_ok=True)

    x, y = data['x'], data['y']
    u, v = data['u'], data['v']
    dx, dy = data['dx'], data['dy']

    # Calculate vorticity and Q-criterion
    vorticity = calculate_vorticity(u, v, dx, dy)
    Q = calculate_q_criterion(u, v, dx, dy)
    vortex_cores = detect_vortex_cores(vorticity, Q)

    # Create meshgrid for plotting
    X, Y = np.meshgrid(x, y)

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))

    # 1. Vorticity contours
    ax1 = plt.subplot(2, 3, 1)
    vort_levels = np.linspace(-np.max(np.abs(vorticity)), np.max(np.abs(vorticity)), 20)
    cs1 = ax1.contourf(X, Y, vorticity, levels=vort_levels, cmap='RdBu_r', extend='both')
    ax1.contour(X, Y, vorticity, levels=vort_levels[::4], colors='black', linewidths=0.5, alpha=0.3)
    plt.colorbar(cs1, ax=ax1, label='Vorticity (1/s)')
    ax1.set_title('Vorticity Field')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_aspect('equal')

    # 2. Q-criterion
    ax2 = plt.subplot(2, 3, 2)
    Q_levels = np.linspace(0, np.max(Q), 15)
    cs2 = ax2.contourf(X, Y, Q, levels=Q_levels, cmap='viridis')
    plt.colorbar(cs2, ax=ax2, label='Q-criterion')
    ax2.set_title('Q-Criterion (Vortex Identification)')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_aspect('equal')

    # 3. Vortex cores overlay
    ax3 = plt.subplot(2, 3, 3)
    velocity_magnitude = np.sqrt(u**2 + v**2)
    cs3 = ax3.contourf(X, Y, velocity_magnitude, levels=20, cmap='plasma', alpha=0.7)
    ax3.contour(X, Y, vortex_cores.astype(int), levels=[0.5], colors='red', linewidths=2)
    plt.colorbar(cs3, ax=ax3, label='Velocity Magnitude (m/s)')
    ax3.set_title('Detected Vortex Cores (Red Lines)')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_aspect('equal')

    # 4. Vorticity with streamlines
    ax4 = plt.subplot(2, 3, 4)
    cs4 = ax4.contourf(X, Y, vorticity, levels=vort_levels, cmap='RdBu_r', alpha=0.8)
    # Add streamlines
    ax4.streamplot(X, Y, u, v, density=1.5, color='black', linewidth=0.8, arrowsize=1.2)
    plt.colorbar(cs4, ax=ax4, label='Vorticity (1/s)')
    ax4.set_title('Vorticity with Streamlines')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_aspect('equal')

    # 5. Circulation analysis
    ax5 = plt.subplot(2, 3, 5)
    cs5 = ax5.contourf(X, Y, vorticity, levels=vort_levels, cmap='RdBu_r', alpha=0.6)

    # Calculate circulation at several radii around domain center
    center_x, center_y = x[len(x)//2], y[len(y)//2]
    radii = np.linspace(0.1, min(x.max()-x.min(), y.max()-y.min())/3, 5)

    circulations = []
    for radius in radii:
        circ = calculate_circulation(u, v, x, y, (center_x, center_y), radius)
        circulations.append(circ)

        # Draw circulation paths
        circle = plt.Circle((center_x, center_y), radius, fill=False,
                          color='white', linewidth=2, linestyle='--')
        ax5.add_patch(circle)

        # Annotate circulation value
        ax5.text(center_x + radius * 0.7, center_y + radius * 0.7,
                f'Γ={circ:.3f}', fontsize=8, color='white',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))

    plt.colorbar(cs5, ax=ax5, label='Vorticity (1/s)')
    ax5.set_title('Circulation Analysis')
    ax5.set_xlabel('x')
    ax5.set_ylabel('y')
    ax5.set_aspect('equal')

    # 6. Statistics and analysis
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    # Calculate statistics
    max_vorticity = np.max(np.abs(vorticity))
    mean_vorticity = np.mean(vorticity)
    std_vorticity = np.std(vorticity)
    max_Q = np.max(Q)
    vortex_area = np.sum(vortex_cores) * dx * dy

    stats_text = f"""
    Vorticity Statistics:
    Max |ω|: {max_vorticity:.4f} 1/s
    Mean ω: {mean_vorticity:.4f} 1/s
    Std ω: {std_vorticity:.4f} 1/s

    Q-Criterion:
    Max Q: {max_Q:.4f}

    Vortex Detection:
    Core area: {vortex_area:.4f} m²

    Circulation Values:
    """

    for i, (r, circ) in enumerate(zip(radii, circulations)):
        stats_text += f"  r={r:.3f}: Γ={circ:.4f}\n"

    ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))

    plt.tight_layout()

    # Save figure
    output_file = os.path.join(output_dir, 'vorticity_analysis.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Vorticity analysis saved to: {output_file}")

    plt.show()

    return vorticity, Q, vortex_cores

def main():
    # Ensure output directories exist
    ensure_dirs()

    parser = argparse.ArgumentParser(description='Vorticity and circulation analysis for CFD data')
    parser.add_argument('input_file', nargs='?', help='VTK file to analyze')
    parser.add_argument('--output', '-o', default=None,
                       help='Output directory for visualizations (default: centralized PLOTS_DIR)')
    parser.add_argument('--latest', '-l', action='store_true',
                       help='Use latest VTK file in data directory')

    args = parser.parse_args()

    # Use centralized output dir if not specified
    output_dir = args.output if args.output else str(PLOTS_DIR)

    # Determine input file
    if args.latest:
        vtk_files = find_vtk_files()
        if not vtk_files:
            print(f"No VTK files found in {DATA_DIR}")
            print("Set CFD_VIZ_DATA_DIR environment variable to specify a different location.")
            return
        input_file = str(max(vtk_files, key=lambda f: f.stat().st_ctime))
        print(f"Using latest file: {input_file}")
    elif args.input_file:
        input_file = args.input_file
    else:
        # Try to find a VTK file
        vtk_files = find_vtk_files()
        if vtk_files:
            input_file = str(vtk_files[0])
            print(f"Using file: {input_file}")
        else:
            print(f"No VTK file specified. Use --help for usage information.")
            print(f"Set CFD_VIZ_DATA_DIR environment variable to specify data location.")
            return

    # Read and analyze data
    data = read_vtk_file(input_file)
    if data is None:
        return

    # Create visualization
    vorticity, Q, vortex_cores = create_vorticity_visualization(data, output_dir)

    print("Vorticity analysis complete!")

if __name__ == "__main__":
    main()