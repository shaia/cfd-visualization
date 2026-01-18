#!/usr/bin/env python3
"""
CFD Animation Script
====================

Command-line tool for creating animations from CFD VTK output files.

Features:
- 2D flow field animations (velocity, pressure, vorticity)
- 3D rotating surface plots
- Lagrangian particle trace animations
- Vorticity analysis with streamlines
- Multi-panel dashboard animations
- Individual frame export

Uses the cfd_viz.animation module for animation creation.

Usage:
    python animate.py [options] <vtk_files_pattern>

Examples:
    python animate.py --type velocity output/*.vtk
    python animate.py --type dashboard --fps 5 data/flow_*.vtk
    python animate.py --export-frames --output frames/ data/*.vtk
"""

import argparse
import glob
import os
import sys
from typing import Dict, List, Tuple

import matplotlib
from numpy.typing import NDArray

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt

from cfd_viz.animation import (
    AnimationFrames,
    advect_particles_through_frames,
    create_3d_surface_animation,
    create_animation_frames,
    create_field_animation,
    create_multi_panel_animation,
    create_particle_trace_animation,
    create_streamline_animation,
    create_vector_animation,
    create_vorticity_analysis_animation,
    export_animation_frames,
    save_animation,
)
from cfd_viz.common import (
    ANIMATIONS_DIR,
    PLOTS_DIR,
    ensure_dirs,
    read_vtk_file as _read_vtk,
)


def read_vtk_file(filename: str) -> Tuple[NDArray, NDArray, Dict[str, NDArray]]:
    """Read VTK file and return (X, Y, data_fields) tuple.

    Args:
        filename: Path to VTK file

    Returns:
        Tuple of (X, Y, data_fields) where X, Y are coordinate meshgrids
        and data_fields is a dict of field name -> 2D array.

    Raises:
        ValueError: If VTK file format is invalid.
    """
    data = _read_vtk(filename)
    if data is None:
        raise ValueError(f"Failed to read VTK file: {filename}")
    return data.X, data.Y, data.fields


def extract_iteration_from_filename(filename: str, fallback: int) -> int:
    """Extract iteration number from VTK filename.

    Expects filenames like 'flow_field_0100.vtk' where the iteration
    number is the last numeric segment before the extension.

    Args:
        filename: Path to the VTK file.
        fallback: Value to return if extraction fails.

    Returns:
        Extracted iteration number, or fallback value.
    """
    try:
        return int(os.path.basename(filename).split("_")[-1].split(".")[0])
    except ValueError:
        return fallback


def load_vtk_files_to_frames(vtk_files: List[str]) -> AnimationFrames:
    """Load VTK files and create AnimationFrames.

    Args:
        vtk_files: List of VTK file paths.

    Returns:
        AnimationFrames containing all loaded data.
    """
    frames_data = []
    time_indices = []

    for filename in sorted(vtk_files):
        try:
            X, Y, data_fields = read_vtk_file(filename)

            u = data_fields.get("u")
            v = data_fields.get("v")
            p = data_fields.get("p")

            if u is not None and v is not None:
                frames_data.append((X, Y, u, v, p))
                time_indices.append(
                    extract_iteration_from_filename(filename, len(frames_data))
                )
        except Exception as e:
            print(f"Warning: Error reading {filename}: {e}")
            continue

    if not frames_data:
        raise ValueError("No valid VTK files found with velocity data")

    print(f"Loaded {len(frames_data)} frames")
    return create_animation_frames(frames_data, time_indices=time_indices)


def create_and_save_animation(
    animation_frames: AnimationFrames,
    animation_type: str,
    output_path: str,
    fps: int = 5,
    field: str = "velocity_mag",
) -> None:
    """Create and save an animation.

    Args:
        animation_frames: AnimationFrames data.
        animation_type: Type of animation to create.
        output_path: Path to save animation.
        fps: Frames per second.
        field: Field name for field animations.
    """
    print(f"Creating {animation_type} animation...")

    if animation_type in {"velocity", "field"}:
        fig, anim = create_field_animation(animation_frames, field)
    elif animation_type == "streamlines":
        fig, anim = create_streamline_animation(animation_frames)
    elif animation_type == "vectors":
        fig, anim = create_vector_animation(animation_frames)
    elif animation_type == "dashboard":
        fig, anim = create_multi_panel_animation(animation_frames)
    elif animation_type == "vorticity":
        fig, anim = create_vorticity_analysis_animation(animation_frames)
    elif animation_type == "3d":
        fig, anim = create_3d_surface_animation(animation_frames, field)
    elif animation_type == "particles":
        traces = advect_particles_through_frames(
            animation_frames, n_particles=50, dt=0.01, steps_per_frame=10
        )
        fig, anim = create_particle_trace_animation(animation_frames, traces)
    else:
        raise ValueError(f"Unknown animation type: {animation_type}")

    print(f"Saving animation to {output_path}...")
    save_animation(anim, output_path, fps=fps)
    plt.close(fig)
    print("Animation saved!")


def main():
    """Main entry point for the animation script."""
    ensure_dirs()

    parser = argparse.ArgumentParser(
        description="Create animations from CFD VTK files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Animation types:
  velocity    - Velocity magnitude field animation
  field       - Generic field animation (use --field to specify)
  streamlines - Streamline animation
  vectors     - Vector field animation
  dashboard   - Multi-panel dashboard (6 views)
  vorticity   - Vorticity analysis with streamlines
  3d          - 3D rotating surface plot
  particles   - Lagrangian particle traces

Examples:
  python animate.py output/*.vtk
  python animate.py --type dashboard --fps 3 data/flow_*.vtk
  python animate.py --type vorticity --output vorticity.gif data/*.vtk
  python animate.py --export-frames --output frames/ data/*.vtk
        """,
    )

    parser.add_argument(
        "vtk_files",
        nargs="+",
        help="VTK files to process (glob patterns supported)",
    )
    parser.add_argument(
        "--type",
        "-t",
        choices=[
            "velocity",
            "field",
            "streamlines",
            "vectors",
            "dashboard",
            "vorticity",
            "3d",
            "particles",
        ],
        default="velocity",
        help="Type of animation to create (default: velocity)",
    )
    parser.add_argument(
        "--field",
        "-f",
        default="velocity_mag",
        help="Field name for field animations (default: velocity_mag)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output path (default: auto-generated in ANIMATIONS_DIR)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=5,
        help="Frames per second (default: 5)",
    )
    parser.add_argument(
        "--export-frames",
        action="store_true",
        help="Export individual frames instead of animation",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Create all animation types",
    )

    args = parser.parse_args()

    # Expand glob patterns
    vtk_files = []
    for pattern in args.vtk_files:
        matches = glob.glob(pattern)
        if matches:
            vtk_files.extend(matches)
        else:
            print(f"Warning: No files matching pattern: {pattern}")

    if not vtk_files:
        print("Error: No VTK files found")
        sys.exit(1)

    vtk_files = sorted(set(vtk_files))  # Remove duplicates and sort
    print(f"Found {len(vtk_files)} VTK files")

    # Load data
    try:
        animation_frames = load_vtk_files_to_frames(vtk_files)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Export frames or create animation
    if args.export_frames:
        output_dir = args.output or str(PLOTS_DIR / "frames")
        print(f"Exporting frames to {output_dir}...")
        exported = export_animation_frames(animation_frames, output_dir)
        print(f"Exported {len(exported)} frames")
    elif args.all:
        # Create all animation types
        animation_types = [
            "velocity",
            "streamlines",
            "vectors",
            "dashboard",
            "vorticity",
            "3d",
            "particles",
        ]
        for anim_type in animation_types:
            output_path = str(ANIMATIONS_DIR / f"cfd_{anim_type}.gif")
            try:
                create_and_save_animation(
                    animation_frames, anim_type, output_path, args.fps, args.field
                )
            except Exception as e:
                print(f"Warning: Failed to create {anim_type} animation: {e}")
    else:
        # Create single animation
        if args.output:
            output_path = args.output
        else:
            output_path = str(ANIMATIONS_DIR / f"cfd_{args.type}.gif")

        create_and_save_animation(
            animation_frames, args.type, output_path, args.fps, args.field
        )

    print("Done!")


if __name__ == "__main__":
    main()
