"""
Unified CLI for cfd-visualization.

Provides ``cfd-viz`` command with subcommands that dispatch to library
functions in ``cfd_viz``.

Usage::

    cfd-viz info
    cfd-viz animate output/*.vtk --type velocity
    cfd-viz dashboard --vtk-pattern "output/*.vtk"
    cfd-viz vorticity data/flow.vtk
    cfd-viz profiles data/flow.vtk
    cfd-viz monitor --watch-dir data/vtk_files
    cfd-viz batch --config batch.toml
"""

import argparse
import glob


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cfd-viz",
        description="CFD visualization toolkit",
    )
    sub = parser.add_subparsers(dest="command")

    # ── info ──────────────────────────────────────────────────────────
    sub.add_parser(
        "info",
        help="Print system capabilities, backends, and optional dependencies",
    )

    # ── animate ───────────────────────────────────────────────────────
    p_anim = sub.add_parser(
        "animate",
        help="Create animations from VTK time-series files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Animation types:\n"
            "  velocity, field, streamlines, vectors,\n"
            "  dashboard, vorticity, 3d, particles"
        ),
    )
    p_anim.add_argument(
        "vtk_files", nargs="+", help="VTK files (glob patterns supported)"
    )
    p_anim.add_argument(
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
        help="Animation type (default: velocity)",
    )
    p_anim.add_argument(
        "--field", "-f", default="velocity_mag", help="Field for field animations"
    )
    p_anim.add_argument(
        "--output", "-o", default=None, help="Output file path for animation"
    )
    p_anim.add_argument("--fps", type=int, default=5, help="Frames per second")
    p_anim.add_argument(
        "--export-frames", action="store_true", help="Export individual frames"
    )
    p_anim.add_argument(
        "--output-dir", default=None, help="Output directory for --export-frames"
    )
    p_anim.add_argument("--all", action="store_true", help="Create all animation types")

    # ── dashboard ─────────────────────────────────────────────────────
    p_dash = sub.add_parser(
        "dashboard", help="Create interactive Plotly dashboards from VTK files"
    )
    p_dash.add_argument(
        "--vtk-pattern",
        default="output/output_optimized_*.vtk",
        help="Glob pattern for VTK files",
    )
    p_dash.add_argument(
        "--output-dir", default="visualization_output", help="Output directory"
    )
    p_dash.add_argument(
        "--auto-open", action="store_true", help="Auto-open HTML in browser"
    )

    # ── vorticity ─────────────────────────────────────────────────────
    p_vort = sub.add_parser("vorticity", help="Vorticity and circulation analysis")
    p_vort.add_argument("input_file", nargs="?", help="VTK file to analyze")
    p_vort.add_argument("--output", "-o", default=None, help="Output directory")
    p_vort.add_argument(
        "--latest", "-l", action="store_true", help="Use latest VTK file"
    )

    # ── profiles ──────────────────────────────────────────────────────
    p_prof = sub.add_parser("profiles", help="Cross-sectional line profile analysis")
    p_prof.add_argument("input_file", nargs="?", help="VTK file to analyze")
    p_prof.add_argument("--output", "-o", default=None, help="Output directory")
    p_prof.add_argument(
        "--interactive", "-i", action="store_true", help="Interactive line analysis"
    )
    p_prof.add_argument(
        "--latest", "-l", action="store_true", help="Use latest VTK file"
    )

    # ── monitor ───────────────────────────────────────────────────────
    p_mon = sub.add_parser("monitor", help="Real-time simulation monitoring dashboard")
    p_mon.add_argument(
        "--watch-dir", "-w", default=None, help="Directory to watch for VTK files"
    )
    p_mon.add_argument("--output", "-o", default=None, help="Output directory")
    p_mon.add_argument(
        "--interval", type=float, default=2.0, help="Polling interval (seconds)"
    )
    p_mon.add_argument(
        "--manual", "-m", action="store_true", help="Manual polling mode"
    )

    # ── batch ─────────────────────────────────────────────────────────
    p_batch = sub.add_parser(
        "batch",
        help="Batch-process multiple VTK files from a TOML config",
    )
    p_batch.add_argument(
        "--config",
        "-c",
        required=True,
        help="Path to batch config file (TOML)",
    )

    return parser


# ── subcommand handlers ──────────────────────────────────────────────


def _cmd_info(_args):
    from cfd_viz.info import print_system_info

    print_system_info()


def _cmd_animate(args):
    from cfd_viz._cli_impl import create_and_save_animation, load_vtk_files_to_frames
    from cfd_viz.animation import export_animation_frames
    from cfd_viz.common import ANIMATIONS_DIR, PLOTS_DIR, ensure_dirs

    ensure_dirs()

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
        raise SystemExit(1)

    vtk_files = sorted(set(vtk_files))
    print(f"Found {len(vtk_files)} VTK files")

    animation_frames = load_vtk_files_to_frames(vtk_files)

    if args.export_frames:
        output_dir = args.output_dir or str(PLOTS_DIR / "frames")
        print(f"Exporting frames to {output_dir}...")
        exported = export_animation_frames(animation_frames, output_dir)
        print(f"Exported {len(exported)} frames")
    elif getattr(args, "all", False):
        for anim_type in [
            "velocity",
            "streamlines",
            "vectors",
            "dashboard",
            "vorticity",
            "3d",
            "particles",
            "field",
        ]:
            output_path = str(ANIMATIONS_DIR / f"cfd_{anim_type}.gif")
            try:
                create_and_save_animation(
                    animation_frames, anim_type, output_path, args.fps, args.field
                )
            except Exception as e:
                print(f"Warning: Failed to create {anim_type} animation: {e}")
    else:
        output_path = args.output or str(ANIMATIONS_DIR / f"cfd_{args.type}.gif")
        create_and_save_animation(
            animation_frames, args.type, output_path, args.fps, args.field
        )

    print("Done!")


def _cmd_dashboard(args):
    from cfd_viz._cli_impl import create_dashboards

    vtk_files = glob.glob(args.vtk_pattern)
    if not vtk_files:
        print(f"No VTK files found matching pattern: {args.vtk_pattern}")
        return

    print(f"Found {len(vtk_files)} VTK files")
    create_dashboards(vtk_files, args.output_dir, args.auto_open)
    print("\nInteractive visualizations complete!")


def _cmd_vorticity(args):
    from cfd_viz._cli_impl import (
        create_vorticity_visualization,
        resolve_vtk_input,
    )
    from cfd_viz.common import PLOTS_DIR, ensure_dirs, read_vtk_file

    ensure_dirs()
    input_file = resolve_vtk_input(args.input_file, args.latest)
    if input_file is None:
        return

    data = read_vtk_file(input_file)
    if data is None:
        return
    if data.u is None or data.v is None:
        print("Error: Could not find velocity data in VTK file")
        return

    output_dir = args.output or str(PLOTS_DIR)
    create_vorticity_visualization(data.to_dict(), output_dir)
    print("Vorticity analysis complete!")


def _cmd_profiles(args):
    from cfd_viz._cli_impl import (
        _vtk_to_profiles_dict,
        create_cross_section_analysis,
        resolve_vtk_input,
    )
    from cfd_viz.common import PLOTS_DIR, ensure_dirs, read_vtk_file

    ensure_dirs()
    input_file = resolve_vtk_input(args.input_file, args.latest)
    if input_file is None:
        return

    data = read_vtk_file(input_file)
    if data is None:
        return

    data_dict = _vtk_to_profiles_dict(data)
    if data_dict is None:
        print("Error: Could not find velocity data in VTK file")
        return

    output_dir = args.output or str(PLOTS_DIR)
    create_cross_section_analysis(data_dict, output_dir)
    print("Cross-section analysis complete!")


def _cmd_monitor(args):
    from cfd_viz._cli_impl import run_monitor
    from cfd_viz.common import DATA_DIR, PLOTS_DIR, ensure_dirs

    ensure_dirs()
    watch_dir = args.watch_dir or str(DATA_DIR)
    output_dir = args.output or str(PLOTS_DIR)
    run_monitor(watch_dir, output_dir, args.interval, args.manual)


def _cmd_batch(args):
    from cfd_viz._batch import run_batch

    run_batch(args.config)


_DISPATCH = {
    "info": _cmd_info,
    "animate": _cmd_animate,
    "dashboard": _cmd_dashboard,
    "vorticity": _cmd_vorticity,
    "profiles": _cmd_profiles,
    "monitor": _cmd_monitor,
    "batch": _cmd_batch,
}


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        raise SystemExit(0)

    handler = _DISPATCH.get(args.command)
    if handler is None:
        parser.print_help()
        raise SystemExit(1)

    handler(args)
