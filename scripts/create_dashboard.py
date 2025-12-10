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

from plotly.offline import plot

from cfd_viz.common import read_vtk_file
from cfd_viz.interactive import (
    create_animated_dashboard,
    create_convergence_figure,
    create_interactive_frame_collection,
)


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
            data = read_vtk_file(filename)
            if data is None:
                print(f"Warning: Could not read {filename}")
                continue

            u = data.fields.get("u")
            v = data.fields.get("v")
            p = data.fields.get("p")

            if u is not None and v is not None:
                frames_list.append((data.x, data.y, u, v, p))
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
        default="visualization_output",
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
