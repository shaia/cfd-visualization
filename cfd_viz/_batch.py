"""Batch processing for multiple VTK files from a TOML config.

Config format (``batch.toml``)::

    [batch]
    output_dir = "output/batch"

    [[batch.jobs]]
    vtk = "data/vtk_files/flow_0100.vtk"
    analyses = ["vorticity", "profiles"]

    [[batch.jobs]]
    vtk = "data/vtk_files/flow_0200.vtk"
    analyses = ["vorticity"]
    output_dir = "output/batch/step200"   # per-job override

    [[batch.jobs]]
    vtk_pattern = "data/vtk_files/flow_*.vtk"
    analyses = ["animate"]
    animate_type = "velocity"
    fps = 10

Supported analyses: ``vorticity``, ``profiles``, ``animate``.
"""

import os
import sys


def _load_toml(path):
    """Load a TOML file, using tomllib (3.11+) or tomli."""
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib

    with open(path, "rb") as f:
        return tomllib.load(f)


def _progress(current, total, label=""):
    """Write a simple progress line to stderr."""
    width = 30
    filled = int(width * current / total) if total > 0 else width
    bar = "#" * filled + "-" * (width - filled)
    pct = (current / total * 100) if total > 0 else 100
    sys.stderr.write(f"\r  [{bar}] {pct:5.1f}%  {label}")
    if current >= total:
        sys.stderr.write("\n")
    sys.stderr.flush()


def _run_vorticity(vtk_file, output_dir):
    from cfd_viz.common import ensure_dirs, read_vtk_file

    ensure_dirs()
    data = read_vtk_file(vtk_file)
    if data is None:
        print(f"  Warning: could not read {vtk_file}")
        return

    if data.u is None or data.v is None:
        print(f"  Warning: no velocity data in {vtk_file}")
        return

    from scripts.create_vorticity_analysis import (
        create_vorticity_visualization,
    )

    data_dict = data.to_dict()
    create_vorticity_visualization(data_dict, output_dir)


def _run_profiles(vtk_file, output_dir):
    from cfd_viz.common import ensure_dirs, read_vtk_file as _read

    ensure_dirs()
    data = _read(vtk_file)
    if data is None:
        print(f"  Warning: could not read {vtk_file}")
        return

    from scripts.create_line_profiles import (
        create_cross_section_analysis,
        read_vtk_file,
    )

    data_dict = read_vtk_file(vtk_file)
    if data_dict is None:
        return
    create_cross_section_analysis(data_dict, output_dir)


def _run_animate(vtk_files, output_dir, animate_type="velocity", fps=5):
    from scripts.create_animation import (
        create_and_save_animation,
        load_vtk_files_to_frames,
    )

    frames = load_vtk_files_to_frames(vtk_files)
    output_path = os.path.join(output_dir, f"cfd_{animate_type}.gif")
    create_and_save_animation(frames, animate_type, output_path, fps=fps)


def run_batch(config_path):
    """Execute a batch config file."""
    import glob as globmod

    cfg = _load_toml(config_path)
    batch = cfg.get("batch", {})
    global_output = batch.get("output_dir", "output/batch")
    jobs = batch.get("jobs", [])

    if not jobs:
        print("No jobs defined in config.")
        return

    total = len(jobs)
    print(f"Batch: {total} job(s) from {config_path}")

    for idx, job in enumerate(jobs, 1):
        analyses = job.get("analyses", [])
        job_output = job.get("output_dir", global_output)
        os.makedirs(job_output, exist_ok=True)

        # Resolve VTK files
        vtk_files = []
        if "vtk" in job:
            vtk_files = [job["vtk"]]
        elif "vtk_pattern" in job:
            vtk_files = sorted(globmod.glob(job["vtk_pattern"]))

        if not vtk_files:
            print(f"  Job {idx}/{total}: no VTK files found, skipping")
            _progress(idx, total, "skipped")
            continue

        label = (
            os.path.basename(vtk_files[0])
            if len(vtk_files) == 1
            else f"{len(vtk_files)} files"
        )
        print(f"  Job {idx}/{total}: {label}  analyses={analyses}")

        for analysis in analyses:
            if analysis == "vorticity":
                for vtk_file in vtk_files:
                    _run_vorticity(vtk_file, job_output)
            elif analysis == "profiles":
                for vtk_file in vtk_files:
                    _run_profiles(vtk_file, job_output)
            elif analysis == "animate":
                animate_type = job.get("animate_type", "velocity")
                fps = job.get("fps", 5)
                _run_animate(vtk_files, job_output, animate_type, fps)
            else:
                print(f"  Warning: unknown analysis '{analysis}', skipping")

        _progress(idx, total, label)

    print("Batch processing complete.")
