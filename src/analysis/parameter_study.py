#!/usr/bin/env python3
"""
Parameter Study and Comparison Tool for CFD Framework
=====================================================

This script provides comprehensive parameter study analysis including:
- Side-by-side comparisons of different simulation runs
- Parameter sweep visualizations
- Solver comparison (basic vs optimized)
- Before/after analysis for design changes

Requirements:
    numpy, matplotlib, scipy, pandas

Usage:
    python parameter_study.py [options]
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import argparse
import glob
import os
import sys
from pathlib import Path
import json

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, PLOTS_DIR, ensure_dirs

def read_vtk_file(filename):
    """Read a VTK structured points file and extract all data"""
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
        return None

    nx, ny = dimensions[0], dimensions[1]
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

    if 'u_velocity' not in data_fields or 'v_velocity' not in data_fields:
        return None

    return {
        'x': x, 'y': y, 'data_fields': data_fields,
        'nx': nx, 'ny': ny, 'dx': spacing[0], 'dy': spacing[1],
        'filename': filename
    }

def extract_parameters_from_filename(filename):
    """Extract parameters from filename patterns"""
    basename = os.path.basename(filename)
    params = {}

    # Common parameter patterns in CFD filenames
    patterns = {
        'Re': r'Re(\d+\.?\d*)',
        'dt': r'dt(\d+\.?\d*e?-?\d*)',
        'mu': r'mu(\d+\.?\d*e?-?\d*)',
        'iter': r'(\d+)\.vtk$',
        'solver': r'(optimized|basic)',
        'grid': r'(\d+)x(\d+)'
    }

    import re
    for param, pattern in patterns.items():
        match = re.search(pattern, basename)
        if match:
            if param == 'grid':
                params['nx'] = int(match.group(1))
                params['ny'] = int(match.group(2))
            elif param == 'solver':
                params[param] = match.group(1)
            else:
                try:
                    params[param] = float(match.group(1))
                except:
                    params[param] = match.group(1)

    return params

def calculate_flow_metrics(data):
    """Calculate key flow metrics for comparison"""
    fields = data['data_fields']
    u = fields['u_velocity']
    v = fields['v_velocity']
    pressure = fields.get('pressure', np.zeros_like(u))

    # Basic metrics
    velocity_mag = np.sqrt(u**2 + v**2)

    metrics = {
        'max_velocity': np.max(velocity_mag),
        'mean_velocity': np.mean(velocity_mag),
        'min_velocity': np.min(velocity_mag),
        'velocity_std': np.std(velocity_mag),
        'max_pressure': np.max(pressure),
        'mean_pressure': np.mean(pressure),
        'min_pressure': np.min(pressure),
        'pressure_std': np.std(pressure),
        'max_u': np.max(u),
        'min_u': np.min(u),
        'max_v': np.max(v),
        'min_v': np.min(v),
    }

    # Advanced metrics
    # Kinetic energy
    kinetic_energy = 0.5 * (u**2 + v**2)
    metrics['total_kinetic_energy'] = np.sum(kinetic_energy) * data['dx'] * data['dy']
    metrics['mean_kinetic_energy'] = np.mean(kinetic_energy)

    # Vorticity
    du_dy = np.gradient(u, data['dy'], axis=0)
    dv_dx = np.gradient(v, data['dx'], axis=1)
    vorticity = dv_dx - du_dy
    metrics['max_vorticity'] = np.max(np.abs(vorticity))
    metrics['mean_vorticity'] = np.mean(np.abs(vorticity))

    # Flow uniformity (coefficient of variation)
    metrics['velocity_uniformity'] = metrics['velocity_std'] / (metrics['mean_velocity'] + 1e-10)

    return metrics

def compare_two_cases(data1, data2, case1_name, case2_name, output_dir):
    """Create detailed comparison between two cases"""
    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(4, 5, figure=fig)

    # Get field data
    u1, v1 = data1['data_fields']['u_velocity'], data1['data_fields']['v_velocity']
    u2, v2 = data2['data_fields']['u_velocity'], data2['data_fields']['v_velocity']
    p1 = data1['data_fields'].get('pressure', np.zeros_like(u1))
    p2 = data2['data_fields'].get('pressure', np.zeros_like(u2))

    # Create meshgrids
    X1, Y1 = np.meshgrid(data1['x'], data1['y'])
    X2, Y2 = np.meshgrid(data2['x'], data2['y'])

    velocity_mag1 = np.sqrt(u1**2 + v1**2)
    velocity_mag2 = np.sqrt(u2**2 + v2**2)

    # 1. Case 1 velocity field
    ax1 = fig.add_subplot(gs[0, 0])
    cs1 = ax1.contourf(X1, Y1, velocity_mag1, levels=20, cmap='viridis')
    plt.colorbar(cs1, ax=ax1, label='|v| (m/s)')
    ax1.set_title(f'{case1_name}\nVelocity Magnitude')
    ax1.set_aspect('equal')

    # 2. Case 2 velocity field
    ax2 = fig.add_subplot(gs[0, 1])
    cs2 = ax2.contourf(X2, Y2, velocity_mag2, levels=20, cmap='viridis')
    plt.colorbar(cs2, ax=ax2, label='|v| (m/s)')
    ax2.set_title(f'{case2_name}\nVelocity Magnitude')
    ax2.set_aspect('equal')

    # 3. Difference plot (if grids match)
    ax3 = fig.add_subplot(gs[0, 2])
    if velocity_mag1.shape == velocity_mag2.shape:
        diff = velocity_mag2 - velocity_mag1
        cs3 = ax3.contourf(X1, Y1, diff, levels=20, cmap='RdBu_r')
        plt.colorbar(cs3, ax=ax3, label='Δ|v| (m/s)')
        ax3.set_title('Velocity Difference\n(Case2 - Case1)')
    else:
        ax3.text(0.5, 0.5, 'Different\nGrid Sizes', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Cannot Compare\n(Different Grids)')
    ax3.set_aspect('equal')

    # 4. Case 1 pressure
    ax4 = fig.add_subplot(gs[1, 0])
    cs4 = ax4.contourf(X1, Y1, p1, levels=20, cmap='plasma')
    plt.colorbar(cs4, ax=ax4, label='Pressure')
    ax4.set_title(f'{case1_name}\nPressure Field')
    ax4.set_aspect('equal')

    # 5. Case 2 pressure
    ax5 = fig.add_subplot(gs[1, 1])
    cs5 = ax5.contourf(X2, Y2, p2, levels=20, cmap='plasma')
    plt.colorbar(cs5, ax=ax5, label='Pressure')
    ax5.set_title(f'{case2_name}\nPressure Field')
    ax5.set_aspect('equal')

    # 6. Pressure difference
    ax6 = fig.add_subplot(gs[1, 2])
    if p1.shape == p2.shape:
        p_diff = p2 - p1
        cs6 = ax6.contourf(X1, Y1, p_diff, levels=20, cmap='RdBu_r')
        plt.colorbar(cs6, ax=ax6, label='ΔP')
        ax6.set_title('Pressure Difference\n(Case2 - Case1)')
    else:
        ax6.text(0.5, 0.5, 'Different\nGrid Sizes', ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('Cannot Compare\n(Different Grids)')
    ax6.set_aspect('equal')

    # 7. Velocity profiles comparison
    ax7 = fig.add_subplot(gs[2, 0])
    # Extract centerline profiles
    y_center_idx1 = len(data1['y']) // 2
    y_center_idx2 = len(data2['y']) // 2

    ax7.plot(data1['x'], u1[y_center_idx1, :], 'b-', linewidth=2, label=f'{case1_name} u-vel')
    ax7.plot(data2['x'], u2[y_center_idx2, :], 'r-', linewidth=2, label=f'{case2_name} u-vel')
    ax7.set_xlabel('x (m)')
    ax7.set_ylabel('u-velocity (m/s)')
    ax7.set_title('Centerline u-velocity Comparison')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # 8. Metrics comparison bar chart
    ax8 = fig.add_subplot(gs[2, 1])
    metrics1 = calculate_flow_metrics(data1)
    metrics2 = calculate_flow_metrics(data2)

    # Select key metrics for comparison
    key_metrics = ['max_velocity', 'mean_velocity', 'total_kinetic_energy', 'max_vorticity']
    metric_values1 = [metrics1[m] for m in key_metrics]
    metric_values2 = [metrics2[m] for m in key_metrics]

    x_pos = np.arange(len(key_metrics))
    width = 0.35

    ax8.bar(x_pos - width/2, metric_values1, width, label=case1_name, alpha=0.8)
    ax8.bar(x_pos + width/2, metric_values2, width, label=case2_name, alpha=0.8)

    ax8.set_xlabel('Metrics')
    ax8.set_ylabel('Values')
    ax8.set_title('Key Metrics Comparison')
    ax8.set_xticks(x_pos)
    ax8.set_xticklabels([m.replace('_', ' ').title() for m in key_metrics], rotation=45)
    ax8.legend()

    # 9. Velocity distribution histograms
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.hist(velocity_mag1.flatten(), bins=50, alpha=0.7, label=case1_name, density=True)
    ax9.hist(velocity_mag2.flatten(), bins=50, alpha=0.7, label=case2_name, density=True)
    ax9.set_xlabel('Velocity Magnitude (m/s)')
    ax9.set_ylabel('Probability Density')
    ax9.set_title('Velocity Distribution Comparison')
    ax9.legend()
    ax9.grid(True, alpha=0.3)

    # 10. Streamlines comparison
    ax10 = fig.add_subplot(gs[3, 0])
    ax10.streamplot(X1, Y1, u1, v1, density=1.5, color='blue', linewidth=0.8, alpha=0.7)
    ax10.set_title(f'{case1_name}\nStreamlines')
    ax10.set_aspect('equal')

    ax11 = fig.add_subplot(gs[3, 1])
    ax11.streamplot(X2, Y2, u2, v2, density=1.5, color='red', linewidth=0.8, alpha=0.7)
    ax11.set_title(f'{case2_name}\nStreamlines')
    ax11.set_aspect('equal')

    # 11. Statistics table
    ax12 = fig.add_subplot(gs[0:2, 3:5])
    ax12.axis('off')

    # Create comprehensive statistics table
    stats_data = []
    all_metrics = set(metrics1.keys()) | set(metrics2.keys())

    for metric in sorted(all_metrics):
        val1 = metrics1.get(metric, 'N/A')
        val2 = metrics2.get(metric, 'N/A')

        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            diff = val2 - val1
            pct_change = (diff / val1 * 100) if val1 != 0 else float('inf')
            stats_data.append([
                metric.replace('_', ' ').title(),
                f'{val1:.4f}',
                f'{val2:.4f}',
                f'{diff:+.4f}',
                f'{pct_change:+.2f}%'
            ])
        else:
            stats_data.append([
                metric.replace('_', ' ').title(),
                str(val1), str(val2), 'N/A', 'N/A'
            ])

    # Create table
    table = ax12.table(cellText=stats_data,
                      colLabels=[f'Metric', case1_name, case2_name, 'Difference', '% Change'],
                      cellLoc='center',
                      loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)

    # Color code the table
    for i in range(len(stats_data)):
        if stats_data[i][4] != 'N/A':
            try:
                pct = float(stats_data[i][4].replace('%', ''))
                if abs(pct) > 10:
                    color = 'lightcoral' if pct > 0 else 'lightgreen'
                    table[(i+1, 4)].set_facecolor(color)
            except:
                pass

    ax12.set_title('Detailed Metrics Comparison', fontsize=14, pad=20)

    # 12. Parameter comparison
    ax13 = fig.add_subplot(gs[2:4, 3:5])
    ax13.axis('off')

    # Extract parameters from filenames
    params1 = extract_parameters_from_filename(data1['filename'])
    params2 = extract_parameters_from_filename(data2['filename'])

    param_text = f"""
    Case Comparison Summary:

    {case1_name}:
    File: {os.path.basename(data1['filename'])}
    Grid: {data1['nx']} × {data1['ny']}
    """

    for key, value in params1.items():
        param_text += f"    {key}: {value}\n"

    param_text += f"""
    {case2_name}:
    File: {os.path.basename(data2['filename'])}
    Grid: {data2['nx']} × {data2['ny']}
    """

    for key, value in params2.items():
        param_text += f"    {key}: {value}\n"

    param_text += f"""
    Key Differences:
    - Max velocity change: {(metrics2['max_velocity'] - metrics1['max_velocity'])/metrics1['max_velocity']*100:+.2f}%
    - Mean velocity change: {(metrics2['mean_velocity'] - metrics1['mean_velocity'])/metrics1['mean_velocity']*100:+.2f}%
    - Kinetic energy change: {(metrics2['total_kinetic_energy'] - metrics1['total_kinetic_energy'])/metrics1['total_kinetic_energy']*100:+.2f}%
    """

    ax13.text(0.05, 0.95, param_text, transform=ax13.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()

    # Save comparison
    output_file = os.path.join(output_dir, f'comparison_{case1_name}_vs_{case2_name}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Comparison saved to: {output_file}")

    plt.show()

    return metrics1, metrics2

def parameter_sweep_analysis(data_list, param_name, output_dir):
    """Analyze parameter sweep results"""
    if len(data_list) < 2:
        print("Need at least 2 cases for parameter sweep analysis")
        return

    # Extract parameter values and calculate metrics
    param_values = []
    all_metrics = []

    for data in data_list:
        params = extract_parameters_from_filename(data['filename'])
        if param_name in params:
            param_values.append(params[param_name])
            all_metrics.append(calculate_flow_metrics(data))

    if len(param_values) == 0:
        print(f"Parameter '{param_name}' not found in filenames")
        return

    # Sort by parameter value
    sorted_indices = np.argsort(param_values)
    param_values = [param_values[i] for i in sorted_indices]
    all_metrics = [all_metrics[i] for i in sorted_indices]

    # Create parameter sweep visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # Key metrics to plot
    metrics_to_plot = [
        'max_velocity', 'mean_velocity', 'total_kinetic_energy',
        'max_vorticity', 'velocity_uniformity', 'mean_pressure'
    ]

    for i, metric in enumerate(metrics_to_plot):
        metric_values = [m[metric] for m in all_metrics]

        axes[i].plot(param_values, metric_values, 'bo-', linewidth=2, markersize=8)
        axes[i].set_xlabel(f'{param_name}')
        axes[i].set_ylabel(metric.replace('_', ' ').title())
        axes[i].set_title(f'{metric.replace("_", " ").title()} vs {param_name}')
        axes[i].grid(True, alpha=0.3)

        # Add trend line
        if len(param_values) > 2:
            z = np.polyfit(param_values, metric_values, 1)
            p = np.poly1d(z)
            axes[i].plot(param_values, p(param_values), "r--", alpha=0.8, linewidth=1)

    plt.tight_layout()

    # Save parameter sweep analysis
    output_file = os.path.join(output_dir, f'parameter_sweep_{param_name}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Parameter sweep analysis saved to: {output_file}")

    plt.show()

    return param_values, all_metrics

def main():
    # Ensure output directories exist
    ensure_dirs()

    parser = argparse.ArgumentParser(description='Parameter study and comparison tool for CFD data')
    parser.add_argument('--input_dir', '-d', default=None,
                       help='Directory containing VTK files (default: centralized DATA_DIR)')
    parser.add_argument('--output', '-o', default=None,
                       help='Output directory for visualizations (default: centralized PLOTS_DIR)')
    parser.add_argument('--compare', '-c', nargs=2, metavar=('FILE1', 'FILE2'),
                       help='Compare two specific VTK files')
    parser.add_argument('--sweep', '-s', help='Parameter name for sweep analysis')
    parser.add_argument('--pattern', '-p', help='File pattern to match (e.g., "optimized*.vtk")')

    args = parser.parse_args()

    # Use centralized directories if not specified
    input_dir = args.input_dir if args.input_dir else str(DATA_DIR)
    output_dir = args.output if args.output else str(PLOTS_DIR)

    if args.compare:
        # Compare two specific files
        data1 = read_vtk_file(args.compare[0])
        data2 = read_vtk_file(args.compare[1])

        if data1 and data2:
            case1_name = os.path.splitext(os.path.basename(args.compare[0]))[0]
            case2_name = os.path.splitext(os.path.basename(args.compare[1]))[0]
            compare_two_cases(data1, data2, case1_name, case2_name, output_dir)
        else:
            print("Error reading VTK files")

    else:
        # Find all VTK files
        if args.pattern:
            pattern = os.path.join(input_dir, args.pattern)
        else:
            pattern = os.path.join(input_dir, '*.vtk')

        vtk_files = glob.glob(pattern)

        if len(vtk_files) == 0:
            print(f"No VTK files found matching: {pattern}")
            print(f"Set CFD_VIZ_DATA_DIR environment variable to specify data location.")
            return

        print(f"Found {len(vtk_files)} VTK files")

        # Read all data
        data_list = []
        for vtk_file in vtk_files:
            data = read_vtk_file(vtk_file)
            if data:
                data_list.append(data)

        if len(data_list) == 0:
            print("No valid VTK files found")
            return

        # Parameter sweep analysis
        if args.sweep:
            parameter_sweep_analysis(data_list, args.sweep, output_dir)

        # If we have exactly 2 files, do a comparison
        elif len(data_list) == 2:
            case1_name = os.path.splitext(os.path.basename(data_list[0]['filename']))[0]
            case2_name = os.path.splitext(os.path.basename(data_list[1]['filename']))[0]
            compare_two_cases(data_list[0], data_list[1], case1_name, case2_name, output_dir)

        else:
            print(f"Found {len(data_list)} files. Use --sweep to analyze parameter variations")
            print("Available files:")
            for data in data_list:
                print(f"  - {os.path.basename(data['filename'])}")

    print("Parameter study analysis complete!")

if __name__ == "__main__":
    main()