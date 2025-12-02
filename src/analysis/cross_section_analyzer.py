#!/usr/bin/env python3
"""
Cross-Sectional Analysis Tool for CFD Framework
===============================================

This script provides detailed cross-sectional analysis including:
- Line plots along user-defined paths
- Boundary layer profile analysis
- Vertical and horizontal slice analysis
- Wake analysis behind objects

Requirements:
    numpy, matplotlib, scipy

Usage:
    python cross_section_analyzer.py [vtk_file] [options]
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy.interpolate import RegularGridInterpolator
import argparse
import glob
import os
import sys

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, PLOTS_DIR, find_vtk_files, ensure_dirs

def read_vtk_file(filename):
    """Read a VTK structured points file and extract all data"""
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

    # Parse all data fields
    data_fields = {}
    field_names = ['u_velocity', 'v_velocity', 'pressure', 'density', 'temperature']

    i = data_start
    while i < len(lines):
        line = lines[i].strip()

        for field_name in field_names:
            if line.startswith(f'SCALARS {field_name}'):
                i += 2  # Skip LOOKUP_TABLE line
                values = []
                while i < len(lines) and len(values) < nx * ny:
                    row_values = [float(x) for x in lines[i].split()]
                    values.extend(row_values)
                    i += 1
                data_fields[field_name] = np.array(values).reshape((ny, nx))
                break
        else:
            i += 1

    # Ensure we have at least velocity data
    if 'u_velocity' not in data_fields or 'v_velocity' not in data_fields:
        print("Error: Could not find velocity data in VTK file")
        return None

    return {
        'x': x, 'y': y, 'data_fields': data_fields,
        'nx': nx, 'ny': ny, 'dx': spacing[0], 'dy': spacing[1]
    }

def extract_line_data(x, y, field, start_point, end_point, num_points=100):
    """Extract data along a line between two points"""
    # Create line coordinates
    line_x = np.linspace(start_point[0], end_point[0], num_points)
    line_y = np.linspace(start_point[1], end_point[1], num_points)

    # Create interpolator
    interp = RegularGridInterpolator((y, x), field, bounds_error=False, fill_value=np.nan)

    # Extract values along line
    line_points = np.column_stack([line_y, line_x])
    line_values = interp(line_points)

    # Calculate distance along line
    distance = np.sqrt((line_x - start_point[0])**2 + (line_y - start_point[1])**2)

    return distance, line_values, line_x, line_y

def analyze_boundary_layer(x, y, u_field, v_field, wall_y, start_x, end_x, num_profiles=5):
    """Analyze boundary layer profiles at multiple locations"""
    # Select x-locations for boundary layer analysis
    x_locations = np.linspace(start_x, end_x, num_profiles)

    profiles = []
    for x_loc in x_locations:
        # Find closest x index
        x_idx = np.argmin(np.abs(x - x_loc))

        # Extract vertical profile from wall
        wall_idx = np.argmin(np.abs(y - wall_y))
        y_profile = y[wall_idx:]
        u_profile = u_field[wall_idx:, x_idx]
        v_profile = v_field[wall_idx:, x_idx]

        # Calculate distance from wall
        wall_distance = y_profile - wall_y

        # Find boundary layer thickness (99% of free stream velocity)
        u_freestream = u_profile[-1]  # Assume top of domain is free stream
        bl_thickness_idx = np.where(u_profile >= 0.99 * u_freestream)[0]
        bl_thickness = wall_distance[bl_thickness_idx[0]] if len(bl_thickness_idx) > 0 else np.nan

        profiles.append({
            'x_location': x_loc,
            'wall_distance': wall_distance,
            'u_velocity': u_profile,
            'v_velocity': v_profile,
            'bl_thickness': bl_thickness
        })

    return profiles

def create_cross_section_analysis(data, output_dir="visualization_output"):
    """Create comprehensive cross-sectional analysis"""
    os.makedirs(output_dir, exist_ok=True)

    x, y = data['x'], data['y']
    fields = data['data_fields']
    u = fields['u_velocity']
    v = fields['v_velocity']

    # Get pressure if available
    pressure = fields.get('pressure', np.zeros_like(u))

    # Create meshgrid for plotting
    X, Y = np.meshgrid(x, y)

    # Create main figure
    fig = plt.figure(figsize=(20, 16))

    # 1. Velocity magnitude with analysis lines
    ax1 = plt.subplot(3, 4, 1)
    velocity_mag = np.sqrt(u**2 + v**2)
    cs1 = ax1.contourf(X, Y, velocity_mag, levels=20, cmap='viridis')
    plt.colorbar(cs1, ax=ax1, label='Velocity Magnitude (m/s)')

    # Add predefined analysis lines
    lines = [
        {'start': (x.min(), y.mean()), 'end': (x.max(), y.mean()), 'name': 'Horizontal Centerline', 'color': 'red'},
        {'start': (x.mean(), y.min()), 'end': (x.mean(), y.max()), 'name': 'Vertical Centerline', 'color': 'white'},
        {'start': (x[len(x)//4], y.min()), 'end': (x[len(x)//4], y.max()), 'name': 'Quarter Section', 'color': 'yellow'},
        {'start': (x[3*len(x)//4], y.min()), 'end': (x[3*len(x)//4], y.max()), 'name': 'Three-Quarter Section', 'color': 'cyan'}
    ]

    for line in lines:
        ax1.plot([line['start'][0], line['end'][0]], [line['start'][1], line['end'][1]],
                color=line['color'], linewidth=2, linestyle='--', label=line['name'])

    ax1.set_title('Velocity Field with Analysis Lines')
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.legend(fontsize=8)
    ax1.set_aspect('equal')

    # 2-5. Line plots for each analysis line
    for i, line in enumerate(lines):
        ax = plt.subplot(3, 4, i + 2)

        # Extract data along line
        distance, u_line, _, _ = extract_line_data(x, y, u, line['start'], line['end'])
        _, v_line, _, _ = extract_line_data(x, y, v, line['start'], line['end'])
        _, p_line, _, _ = extract_line_data(x, y, pressure, line['start'], line['end'])

        # Plot velocity components
        ax.plot(distance, u_line, 'b-', label='u-velocity', linewidth=1.5)
        ax.plot(distance, v_line, 'r-', label='v-velocity', linewidth=1.5)
        ax.plot(distance, np.sqrt(u_line**2 + v_line**2), 'k-', label='|v|', linewidth=2)

        # Add pressure on secondary axis
        ax2 = ax.twinx()
        ax2.plot(distance, p_line, 'g--', label='pressure', alpha=0.7)
        ax2.set_ylabel('Pressure', color='g')
        ax2.tick_params(axis='y', labelcolor='g')

        ax.set_title(f'{line["name"]}')
        ax.set_xlabel('Distance along line (m)')
        ax.set_ylabel('Velocity (m/s)')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)

    # 6. Boundary layer analysis
    ax6 = plt.subplot(3, 4, 6)

    # Analyze boundary layer near bottom wall
    wall_y = y.min()
    bl_profiles = analyze_boundary_layer(x, y, u, v, wall_y, x[len(x)//4], x[3*len(x)//4])

    colors = plt.cm.plasma(np.linspace(0, 1, len(bl_profiles)))
    for profile, color in zip(bl_profiles, colors):
        # Normalize wall distance by boundary layer thickness for comparison
        if not np.isnan(profile['bl_thickness']) and profile['bl_thickness'] > 0:
            y_norm = profile['wall_distance'] / profile['bl_thickness']
            u_norm = profile['u_velocity'] / np.max(profile['u_velocity'])
            ax6.plot(u_norm, y_norm, color=color, linewidth=2,
                    label=f'x={profile["x_location"]:.2f}')

    ax6.set_xlabel('u/u_max')
    ax6.set_ylabel('y/δ (normalized wall distance)')
    ax6.set_title('Boundary Layer Profiles')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim(0, 2)

    # 7. Velocity profiles at different x-locations
    ax7 = plt.subplot(3, 4, 7)

    x_stations = [x[len(x)//6], x[len(x)//3], x[len(x)//2], x[2*len(x)//3], x[5*len(x)//6]]
    colors = plt.cm.viridis(np.linspace(0, 1, len(x_stations)))

    for x_station, color in zip(x_stations, colors):
        # Extract vertical profile
        _, u_profile, _, y_profile = extract_line_data(x, y, u, (x_station, y.min()), (x_station, y.max()))

        ax7.plot(u_profile, y_profile, color=color, linewidth=2, label=f'x={x_station:.2f}')

    ax7.set_xlabel('u-velocity (m/s)')
    ax7.set_ylabel('y (m)')
    ax7.set_title('Velocity Profiles at Different Stations')
    ax7.legend(fontsize=8)
    ax7.grid(True, alpha=0.3)

    # 8. Pressure distribution along centerline
    ax8 = plt.subplot(3, 4, 8)

    # Horizontal centerline pressure
    _, p_horizontal, x_line, _ = extract_line_data(x, y, pressure, (x.min(), y.mean()), (x.max(), y.mean()))

    ax8.plot(x_line, p_horizontal, 'b-', linewidth=2, label='Centerline Pressure')

    # Add pressure gradient
    dp_dx = np.gradient(p_horizontal, x_line)
    ax8_twin = ax8.twinx()
    ax8_twin.plot(x_line, dp_dx, 'r--', alpha=0.7, label='Pressure Gradient')
    ax8_twin.set_ylabel('dp/dx', color='r')
    ax8_twin.tick_params(axis='y', labelcolor='r')

    ax8.set_xlabel('x (m)')
    ax8.set_ylabel('Pressure')
    ax8.set_title('Pressure Distribution')
    ax8.legend(loc='upper left')
    ax8.grid(True, alpha=0.3)

    # 9-12. Detailed analysis plots
    # 9. Wake analysis (if there's flow separation)
    ax9 = plt.subplot(3, 4, 9)

    # Look for minimum velocity regions (potential wake)
    velocity_mag = np.sqrt(u**2 + v**2)
    min_vel_threshold = 0.1 * np.max(velocity_mag)
    wake_regions = velocity_mag < min_vel_threshold

    ax9.contourf(X, Y, velocity_mag, levels=20, cmap='viridis', alpha=0.7)
    ax9.contour(X, Y, wake_regions.astype(int), levels=[0.5], colors='red', linewidths=2)
    ax9.set_title('Wake Regions (Red Contours)')
    ax9.set_xlabel('x (m)')
    ax9.set_ylabel('y (m)')
    ax9.set_aspect('equal')

    # 10. Velocity fluctuations
    ax10 = plt.subplot(3, 4, 10)

    # Calculate velocity fluctuations from mean
    u_mean = np.mean(u, axis=1, keepdims=True)
    v_mean = np.mean(v, axis=1, keepdims=True)
    u_fluct = u - u_mean
    v_fluct = v - v_mean
    fluct_magnitude = np.sqrt(u_fluct**2 + v_fluct**2)

    cs10 = ax10.contourf(X, Y, fluct_magnitude, levels=15, cmap='hot')
    plt.colorbar(cs10, ax=ax10, label='Velocity Fluctuation Magnitude')
    ax10.set_title('Velocity Fluctuations')
    ax10.set_xlabel('x (m)')
    ax10.set_ylabel('y (m)')
    ax10.set_aspect('equal')

    # 11. Statistics summary
    ax11 = plt.subplot(3, 4, 11)
    ax11.axis('off')

    # Calculate comprehensive statistics
    stats_text = f"""
    Flow Statistics:

    Domain:
    x: [{x.min():.3f}, {x.max():.3f}] m
    y: [{y.min():.3f}, {y.max():.3f}] m
    Grid: {len(x)} × {len(y)}

    Velocity:
    Max |v|: {np.max(velocity_mag):.3f} m/s
    Mean |v|: {np.mean(velocity_mag):.3f} m/s
    Min |v|: {np.min(velocity_mag):.3f} m/s

    Pressure:
    Max p: {np.max(pressure):.3f}
    Mean p: {np.mean(pressure):.3f}
    Min p: {np.min(pressure):.3f}

    Boundary Layer:
    """

    for i, profile in enumerate(bl_profiles):
        if not np.isnan(profile['bl_thickness']):
            stats_text += f"  δ at x={profile['x_location']:.2f}: {profile['bl_thickness']:.4f} m\n"

    ax11.text(0.05, 0.95, stats_text, transform=ax11.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))

    # 12. Cross-sectional averages
    ax12 = plt.subplot(3, 4, 12)

    # Calculate streamwise averages
    u_avg_y = np.mean(u, axis=1)  # Average across x for each y
    v_avg_y = np.mean(v, axis=1)
    u_avg_x = np.mean(u, axis=0)  # Average across y for each x
    v_avg_x = np.mean(v, axis=0)

    ax12.plot(u_avg_y, y, 'b-', linewidth=2, label='<u> vs y')
    ax12.plot(v_avg_y, y, 'r-', linewidth=2, label='<v> vs y')
    ax12.set_xlabel('Average Velocity (m/s)')
    ax12.set_ylabel('y (m)')
    ax12.set_title('Cross-sectional Velocity Averages')
    ax12.legend()
    ax12.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_file = os.path.join(output_dir, 'cross_section_analysis.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Cross-section analysis saved to: {output_file}")

    plt.show()

    return lines, bl_profiles

def interactive_line_analysis(data):
    """Create interactive tool for custom line analysis"""
    print("\nStarting interactive line analysis...")
    print("Click two points to define a line for analysis")

    x, y = data['x'], data['y']
    fields = data['data_fields']
    u = fields['u_velocity']
    v = fields['v_velocity']

    # Create interactive plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot velocity field
    X, Y = np.meshgrid(x, y)
    velocity_mag = np.sqrt(u**2 + v**2)
    cs = ax1.contourf(X, Y, velocity_mag, levels=20, cmap='viridis')
    plt.colorbar(cs, ax=ax1, label='Velocity Magnitude (m/s)')
    ax1.set_title('Click two points to define analysis line')
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_aspect('equal')

    # Store clicked points
    clicked_points = []
    line_plot = None

    def on_click(event):
        nonlocal line_plot
        if event.inaxes == ax1:
            clicked_points.append((event.xdata, event.ydata))
            ax1.plot(event.xdata, event.ydata, 'ro', markersize=8)

            if len(clicked_points) == 2:
                # Draw line
                if line_plot:
                    line_plot.remove()
                line_plot, = ax1.plot([clicked_points[0][0], clicked_points[1][0]],
                                     [clicked_points[0][1], clicked_points[1][1]],
                                     'r-', linewidth=3)

                # Extract and plot data along line
                distance, u_line, _, _ = extract_line_data(x, y, u, clicked_points[0], clicked_points[1])
                _, v_line, _, _ = extract_line_data(x, y, v, clicked_points[0], clicked_points[1])

                ax2.clear()
                ax2.plot(distance, u_line, 'b-', label='u-velocity', linewidth=2)
                ax2.plot(distance, v_line, 'r-', label='v-velocity', linewidth=2)
                ax2.plot(distance, np.sqrt(u_line**2 + v_line**2), 'k-', label='|v|', linewidth=2)
                ax2.set_xlabel('Distance along line (m)')
                ax2.set_ylabel('Velocity (m/s)')
                ax2.set_title('Velocity Profile Along Line')
                ax2.legend()
                ax2.grid(True, alpha=0.3)

                plt.draw()

                # Reset for next line
                clicked_points.clear()

    # Connect click event
    fig.canvas.mpl_connect('button_press_event', on_click)

    plt.show()

def main():
    # Ensure output directories exist
    ensure_dirs()

    parser = argparse.ArgumentParser(description='Cross-sectional analysis for CFD data')
    parser.add_argument('input_file', nargs='?', help='VTK file to analyze')
    parser.add_argument('--output', '-o', default=None,
                       help='Output directory for visualizations (default: centralized PLOTS_DIR)')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Enable interactive line analysis')
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

    # Create comprehensive analysis
    lines, bl_profiles = create_cross_section_analysis(data, output_dir)

    # Interactive analysis if requested
    if args.interactive:
        interactive_line_analysis(data)

    print("Cross-section analysis complete!")

if __name__ == "__main__":
    main()