#!/usr/bin/env python3
"""
CFD Visualization CLI
=====================

Command-line interface for creating visualizations from VTK output files.

Usage:
    python visualize_cfd.py [options]

    # Create static PNG plots
    python visualize_cfd.py --static

    # Create animated GIFs
    python visualize_cfd.py --animate

    # Create both (default)
    python visualize_cfd.py --all

    # Visualize specific field
    python visualize_cfd.py --field pressure --animate
"""

import argparse
import os

from cfd_viz.animation.animation import (
    animate_3d_rotating_surface,
    animate_flow_fields_2x3_grid,
    animate_lagrangian_particle_traces,
    animate_vorticity_field_and_streamlines,
    create_animations,
    create_static_plots,
    export_individual_frames,
    plot_flow_statistics_over_time,
)
from cfd_viz.common import DATA_DIR, ensure_dirs, find_vtk_files


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Create visualizations from CFD VTK files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_cfd.py --static          # Create PNG plots
  python visualize_cfd.py --animate         # Create GIF animations
  python visualize_cfd.py --all             # Create both (default)
  python visualize_cfd.py --field p         # Only visualize pressure field
  python visualize_cfd.py --latest --static # Plot only the latest VTK file
  python visualize_cfd.py --advanced        # Create all advanced visualizations
        """,
    )

    parser.add_argument(
        "--static", "-s", action="store_true", help="Create static PNG plots"
    )
    parser.add_argument(
        "--animate", "-a", action="store_true", help="Create animated GIFs"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Create both static and animated outputs (default)",
    )
    parser.add_argument(
        "--advanced",
        action="store_true",
        help="Create advanced visualizations (2x3 grid, 3D, particles, etc.)",
    )
    parser.add_argument(
        "--field",
        "-f",
        type=str,
        default=None,
        help="Specific field to visualize (velocity_magnitude, p, u, v)",
    )
    parser.add_argument(
        "--input", "-i", type=str, default=None, help="Single VTK file to visualize"
    )
    parser.add_argument(
        "--latest", "-l", action="store_true", help="Use only the most recent VTK file"
    )
    parser.add_argument(
        "--streamlines", action="store_true", help="Include streamline animations"
    )
    parser.add_argument(
        "--frames", action="store_true", help="Export individual frames as PNGs"
    )
    parser.add_argument(
        "--statistics", action="store_true", help="Create statistical analysis plots"
    )
    parser.add_argument(
        "--prefix", "-p", type=str, default=None, help="Prefix for output filenames"
    )

    args = parser.parse_args()

    # Ensure output directories exist
    ensure_dirs()

    # Determine which VTK files to use
    if args.input:
        if not os.path.exists(args.input):
            print(f"Error: File not found: {args.input}")
            return
        vtk_files = [args.input]
    else:
        vtk_files = find_vtk_files()
        if not vtk_files:
            print(f"No VTK files found in {DATA_DIR}")
            print(
                "Set CFD_VIZ_DATA_DIR environment variable to specify a different location."
            )
            return
        vtk_files = [str(f) for f in vtk_files]

        if args.latest:
            vtk_files = [max(vtk_files, key=os.path.getmtime)]

    print(f"Found {len(vtk_files)} VTK file(s)")

    # Determine what to create (default: do both if neither specified)
    do_both = (
        not args.static
        and not args.animate
        and not args.advanced
        and not args.frames
        and not args.statistics
    )
    do_static = args.static or args.all or do_both
    do_animate = args.animate or args.all or do_both

    if do_static:
        create_static_plots(vtk_files, args.field)

    if do_animate and len(vtk_files) > 1:
        create_animations(
            vtk_files,
            args.field,
            include_streamlines=args.streamlines,
            output_prefix=args.prefix,
        )
    elif do_animate and len(vtk_files) == 1:
        print("\nSkipping animations (need multiple VTK files for animation)")

    if args.advanced and len(vtk_files) > 1:
        print("\nCreating advanced visualizations...")

        print("\n1. Animating flow fields (2x3 grid)...")
        animate_flow_fields_2x3_grid(vtk_files)

        print("\n2. Animating 3D rotating surface (velocity magnitude)...")
        animate_3d_rotating_surface(vtk_files, "velocity_mag")

        print("\n3. Animating 3D rotating surface (pressure)...")
        animate_3d_rotating_surface(vtk_files, "p")

        print("\n4. Animating Lagrangian particle traces...")
        animate_lagrangian_particle_traces(vtk_files)

        print("\n5. Animating vorticity field and streamlines...")
        animate_vorticity_field_and_streamlines(vtk_files)

    if args.statistics and len(vtk_files) > 1:
        print("\nPlotting flow statistics over time...")
        plot_flow_statistics_over_time(vtk_files)

    if args.frames:
        print("\nExporting individual frames...")
        export_individual_frames(vtk_files)

    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
