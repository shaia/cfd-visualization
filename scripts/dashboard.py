#!/usr/bin/env python3
"""Interactive CFD Dashboard using Plotly.

CLI tool for creating web-based interactive visualizations from CFD VTK output files.
Uses the cfd_viz.interactive module for figure generation.

Example:
    python scripts/dashboard.py --output-dir visualization_output
    python scripts/dashboard.py --vtk-pattern "output/*.vtk" --auto-open
"""

import argparse
import glob
import os
from pathlib import Path

import numpy as np
from plotly.offline import plot

from cfd_viz.interactive import (
    create_animated_dashboard,
    create_convergence_figure,
    create_interactive_frame_collection,
)


def read_vtk_file(filename: str) -> tuple:
    """Read a VTK structured points file and extract data.

    Args:
        filename: Path to VTK file.

    Returns:
        Tuple of (x, y, data_fields dict).
    """
    with open(filename) as f:
        lines = f.readlines()

    dimensions = None
    origin = None
    spacing = None
    data_start = None

    for i, line in enumerate(lines):
        if line.startswith("DIMENSIONS"):
            dimensions = [int(x) for x in line.split()[1:4]]
        elif line.startswith("ORIGIN"):
            origin = [float(x) for x in line.split()[1:4]]
        elif line.startswith("SPACING"):
            spacing = [float(x) for x in line.split()[1:4]]
        elif line.startswith("POINT_DATA"):
            data_start = i + 1
            break

    if not all([dimensions, origin, spacing, data_start]):
        raise ValueError(f"Invalid VTK file format: {filename}")

    nx, ny = dimensions[0], dimensions[1]

    x = np.linspace(origin[0], origin[0] + (nx - 1) * spacing[0], nx)
    y = np.linspace(origin[1], origin[1] + (ny - 1) * spacing[1], ny)

    data_fields = {}
    i = data_start

    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("SCALARS"):
            field_name = line.split()[1]
            i += 2  # Skip LOOKUP_TABLE line

            field_data = []
            while i < len(lines) and not lines[i].strip().startswith(
                ("SCALARS", "VECTORS")
            ):
                values = lines[i].strip().split()
                field_data.extend([float(v) for v in values])
                i += 1

            field_data = np.array(field_data).reshape((ny, nx))
            data_fields[field_name] = field_data
        else:
            i += 1

    return x, y, data_fields


def load_vtk_files(vtk_files: list) -> tuple:
    """Load all VTK files and extract frame data.

    Args:
        vtk_files: List of VTK file paths.

    Returns:
        Tuple of (frames_list, time_indices).
    """
    frames_list = []
    time_indices = []

    for filename in sorted(vtk_files):
        try:
            x, y, data_fields = read_vtk_file(filename)
            if "u" in data_fields and "v" in data_fields and "p" in data_fields:
                frames_list.append(
                    (x, y, data_fields["u"], data_fields["v"], data_fields["p"])
                )
                iteration = int(os.path.basename(filename).split("_")[-1].split(".")[0])
                time_indices.append(iteration)
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue

    return frames_list, time_indices


def create_dashboards(vtk_files: list, output_dir: Path, auto_open: bool = False):
    """Create interactive dashboards from VTK files.

    Args:
        vtk_files: List of VTK file paths.
        output_dir: Output directory for HTML files.
        auto_open: Whether to open HTML files in browser.
    """
    frames_list, time_indices = load_vtk_files(vtk_files)

    if not frames_list:
        print("No valid data found!")
        return

    print(f"Loaded {len(frames_list)} frames")

    # Create frame collection using the interactive module
    collection = create_interactive_frame_collection(
        frames_list, time_indices=time_indices
    )

    # Create animated dashboard
    print("Creating animated dashboard...")
    dashboard_fig = create_animated_dashboard(collection)
    dashboard_path = output_dir / "cfd_interactive_dashboard.html"
    plot(dashboard_fig, filename=str(dashboard_path), auto_open=auto_open)
    print(f"Animated dashboard saved to: {dashboard_path}")

    # Create convergence figure
    print("Creating convergence plot...")
    convergence_fig = create_convergence_figure(collection)
    convergence_path = output_dir / "cfd_convergence.html"
    plot(convergence_fig, filename=str(convergence_path), auto_open=auto_open)
    print(f"Convergence plot saved to: {convergence_path}")


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Create interactive CFD dashboards from VTK files."
    )
    parser.add_argument(
        "--vtk-pattern",
        default="output/output_optimized_*.vtk",
        help="Glob pattern for VTK files (default: output/output_optimized_*.vtk)",
    )
    parser.add_argument(
        "--output-dir",
        default="visualization/visualization_output",
        help="Output directory for HTML files",
    )
    parser.add_argument(
        "--auto-open",
        action="store_true",
        help="Auto-open HTML files in browser",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vtk_files = glob.glob(args.vtk_pattern)

    if not vtk_files:
        print(f"No VTK files found matching pattern: {args.vtk_pattern}")
        return

    print(f"Found {len(vtk_files)} VTK files")

    create_dashboards(vtk_files, output_dir, args.auto_open)

    print("\nInteractive visualizations complete!")
    print(
        "Open the HTML files in your web browser to interact with the visualizations."
    )


if __name__ == "__main__":
    main()
