"""
Unified CLI for cfd-visualization.

Provides ``cfd-viz`` command with subcommands that dispatch to existing
scripts and library functions.

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
import importlib
import sys


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
    p_anim.add_argument("--output", "-o", default=None, help="Output path")
    p_anim.add_argument("--fps", type=int, default=5, help="Frames per second")
    p_anim.add_argument(
        "--export-frames", action="store_true", help="Export individual frames"
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


# ── helpers ──────────────────────────────────────────────────────────


def _run_script_main(module_name, argv):
    """Import a script module and call its ``main()`` after overriding sys.argv."""
    old_argv = sys.argv
    sys.argv = ["cfd-viz", *argv]
    try:
        mod = importlib.import_module(module_name)
        mod.main()
    finally:
        sys.argv = old_argv


def _build_animate_argv(args):
    argv = list(args.vtk_files)
    argv += ["--type", args.type]
    argv += ["--field", args.field]
    argv += ["--fps", str(args.fps)]
    if args.output:
        argv += ["--output", args.output]
    if args.export_frames:
        argv.append("--export-frames")
    if getattr(args, "all", False):
        argv.append("--all")
    return argv


# ── subcommand handlers ──────────────────────────────────────────────


def _cmd_info(_args):
    from cfd_viz.info import print_system_info

    print_system_info()


def _cmd_animate(args):
    _run_script_main("scripts.create_animation", _build_animate_argv(args))


def _cmd_dashboard(args):
    argv = ["--vtk-pattern", args.vtk_pattern, "--output-dir", args.output_dir]
    if args.auto_open:
        argv.append("--auto-open")
    _run_script_main("scripts.create_dashboard", argv)


def _cmd_vorticity(args):
    argv = []
    if args.input_file:
        argv.append(args.input_file)
    if args.output:
        argv += ["--output", args.output]
    if args.latest:
        argv.append("--latest")
    _run_script_main("scripts.create_vorticity_analysis", argv)


def _cmd_profiles(args):
    argv = []
    if args.input_file:
        argv.append(args.input_file)
    if args.output:
        argv += ["--output", args.output]
    if args.interactive:
        argv.append("--interactive")
    if args.latest:
        argv.append("--latest")
    _run_script_main("scripts.create_line_profiles", argv)


def _cmd_monitor(args):
    argv = []
    if args.watch_dir:
        argv += ["--watch_dir", args.watch_dir]
    if args.output:
        argv += ["--output", args.output]
    argv += ["--interval", str(args.interval)]
    if args.manual:
        argv.append("--manual")
    _run_script_main("scripts.create_monitor", argv)


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
