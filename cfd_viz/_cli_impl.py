"""Internal helpers for CLI and batch commands.

Contains functions moved from ``scripts/`` so the installed wheel can run
all ``cfd-viz`` subcommands without depending on unpackaged scripts.
"""

import os
import sys
from pathlib import Path

import matplotlib

# Only force non-interactive backend when no display is available.
# This preserves GUI capability for interactive commands (e.g. monitor).
if os.name != "nt" and not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

# ── VTK input file resolution ────────────────────────────────────────


def resolve_vtk_input(input_file=None, latest=False):
    """Resolve a VTK input file from CLI arguments.

    Returns the path string, or None if nothing was found.
    """
    from cfd_viz.common import DATA_DIR, find_vtk_files

    if latest:
        vtk_files = find_vtk_files()
        if not vtk_files:
            print(f"No VTK files found in {DATA_DIR}")
            return None
        path = str(max(vtk_files, key=lambda f: f.stat().st_ctime))
        print(f"Using latest file: {path}")
        return path

    if input_file:
        return input_file

    vtk_files = find_vtk_files()
    if vtk_files:
        path = str(vtk_files[0])
        print(f"Using file: {path}")
        return path

    print("No VTK file specified. Use --help for usage information.")
    return None


# ── Animation helpers (from scripts/create_animation.py) ─────────────


def extract_iteration_from_filename(filename, fallback):
    """Extract iteration number from VTK filename."""
    try:
        return int(os.path.basename(filename).split("_")[-1].split(".")[0])
    except ValueError:
        return fallback


def load_vtk_files_to_frames(vtk_files):
    """Load VTK files and create AnimationFrames."""
    from cfd_viz.animation import create_animation_frames
    from cfd_viz.common import read_vtk_file

    frames_data = []
    time_indices = []

    for filename in sorted(vtk_files):
        try:
            data = read_vtk_file(filename)
            if data is None:
                raise ValueError(f"Failed to read VTK file: {filename}")
            X, Y, fields = data.X, data.Y, data.fields
            u, v, p = fields.get("u"), fields.get("v"), fields.get("p")
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

    _progress_stderr(len(vtk_files), len(vtk_files), "frames loaded")
    return create_animation_frames(frames_data, time_indices=time_indices)


def create_and_save_animation(
    animation_frames, animation_type, output_path, fps=5, field="velocity_mag"
):
    """Create and save an animation."""
    from cfd_viz.animation import (
        advect_particles_through_frames,
        create_3d_surface_animation,
        create_field_animation,
        create_multi_panel_animation,
        create_particle_trace_animation,
        create_streamline_animation,
        create_vector_animation,
        create_vorticity_analysis_animation,
        save_animation,
    )

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


# ── Vorticity visualization (from scripts/create_vorticity_analysis.py)


def create_vorticity_visualization(data, output_dir):
    """Create comprehensive 6-panel vorticity visualization."""
    from cfd_viz.fields.vorticity import (
        circulation,
        detect_vortex_cores,
        q_criterion,
        vorticity,
    )

    os.makedirs(output_dir, exist_ok=True)

    x, y = data["x"], data["y"]
    u, v = data["u"], data["v"]
    dx, dy = data["dx"], data["dy"]

    omega = vorticity(u, v, dx, dy)
    Q = q_criterion(u, v, dx, dy)
    vortex_cores = detect_vortex_cores(omega, Q)
    X, Y = np.meshgrid(x, y)

    plt.figure(figsize=(16, 12))

    # 1. Vorticity contours
    ax1 = plt.subplot(2, 3, 1)
    vort_levels = np.linspace(-np.max(np.abs(omega)), np.max(np.abs(omega)), 20)
    cs1 = ax1.contourf(X, Y, omega, levels=vort_levels, cmap="RdBu_r", extend="both")
    ax1.contour(
        X, Y, omega, levels=vort_levels[::4], colors="black", linewidths=0.5, alpha=0.3
    )
    plt.colorbar(cs1, ax=ax1, label="Vorticity (1/s)")
    ax1.set_title("Vorticity Field")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_aspect("equal")

    # 2. Q-criterion
    ax2 = plt.subplot(2, 3, 2)
    Q_levels = np.linspace(0, np.max(Q), 15)
    cs2 = ax2.contourf(X, Y, Q, levels=Q_levels, cmap="viridis")
    plt.colorbar(cs2, ax=ax2, label="Q-criterion")
    ax2.set_title("Q-Criterion (Vortex Identification)")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_aspect("equal")

    # 3. Vortex cores overlay
    ax3 = plt.subplot(2, 3, 3)
    velocity_magnitude = np.sqrt(u**2 + v**2)
    cs3 = ax3.contourf(X, Y, velocity_magnitude, levels=20, cmap="plasma", alpha=0.7)
    ax3.contour(
        X, Y, vortex_cores.astype(int), levels=[0.5], colors="red", linewidths=2
    )
    plt.colorbar(cs3, ax=ax3, label="Velocity Magnitude (m/s)")
    ax3.set_title("Detected Vortex Cores (Red Lines)")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_aspect("equal")

    # 4. Vorticity with streamlines
    ax4 = plt.subplot(2, 3, 4)
    cs4 = ax4.contourf(X, Y, omega, levels=vort_levels, cmap="RdBu_r", alpha=0.8)
    ax4.streamplot(X, Y, u, v, density=1.5, color="black", linewidth=0.8, arrowsize=1.2)
    plt.colorbar(cs4, ax=ax4, label="Vorticity (1/s)")
    ax4.set_title("Vorticity with Streamlines")
    ax4.set_xlabel("x")
    ax4.set_ylabel("y")
    ax4.set_aspect("equal")

    # 5. Circulation analysis
    ax5 = plt.subplot(2, 3, 5)
    cs5 = ax5.contourf(X, Y, omega, levels=vort_levels, cmap="RdBu_r", alpha=0.6)
    center_x, center_y = x[len(x) // 2], y[len(y) // 2]
    radii = np.linspace(0.1, min(x.max() - x.min(), y.max() - y.min()) / 3, 5)
    circulations = []
    for radius in radii:
        circ = circulation(u, v, x, y, (center_x, center_y), radius)
        circulations.append(circ)
        circle = plt.Circle(
            (center_x, center_y),
            radius,
            fill=False,
            color="white",
            linewidth=2,
            linestyle="--",
        )
        ax5.add_patch(circle)
        ax5.text(
            center_x + radius * 0.7,
            center_y + radius * 0.7,
            f"\u0393={circ:.3f}",
            fontsize=8,
            color="white",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7),
        )
    plt.colorbar(cs5, ax=ax5, label="Vorticity (1/s)")
    ax5.set_title("Circulation Analysis")
    ax5.set_xlabel("x")
    ax5.set_ylabel("y")
    ax5.set_aspect("equal")

    # 6. Statistics
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis("off")
    stats_text = (
        f"\n    Vorticity Statistics:\n"
        f"    Max |\u03c9|: {np.max(np.abs(omega)):.4f} 1/s\n"
        f"    Mean \u03c9: {np.mean(omega):.4f} 1/s\n"
        f"    Std \u03c9: {np.std(omega):.4f} 1/s\n\n"
        f"    Q-Criterion:\n"
        f"    Max Q: {np.max(Q):.4f}\n\n"
        f"    Vortex Detection:\n"
        f"    Core area: {np.sum(vortex_cores) * dx * dy:.4f} m\u00b2\n\n"
        f"    Circulation Values:\n"
    )
    for r, circ in zip(radii, circulations):
        stats_text += f"      r={r:.3f}: \u0393={circ:.4f}\n"
    ax6.text(
        0.05,
        0.95,
        stats_text,
        transform=ax6.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
    )

    plt.tight_layout()
    output_file = os.path.join(output_dir, "vorticity_analysis.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Vorticity analysis saved to: {output_file}")
    plt.close()

    return omega, Q, vortex_cores


# ── Line profiles (from scripts/create_line_profiles.py) ─────────────


def _vtk_to_profiles_dict(data):
    """Convert VTKData to the dict format create_cross_section_analysis expects."""
    field_mapping = {
        "u": "u_velocity",
        "v": "v_velocity",
        "p": "pressure",
        "rho": "density",
        "T": "temperature",
    }
    data_fields = {}
    for old, new in field_mapping.items():
        if old in data.fields:
            data_fields[new] = data.fields[old]
    for name, field in data.fields.items():
        if name not in field_mapping:
            data_fields[name] = field
    if "u_velocity" not in data_fields or "v_velocity" not in data_fields:
        return None
    return {
        "x": data.x,
        "y": data.y,
        "data_fields": data_fields,
        "nx": data.nx,
        "ny": data.ny,
        "dx": data.dx,
        "dy": data.dy,
    }


def _extract_line_data(x, y, field, start_point, end_point, num_points=100):
    """Extract data along a line between two points."""
    from scipy.interpolate import RegularGridInterpolator

    line_x = np.linspace(start_point[0], end_point[0], num_points)
    line_y = np.linspace(start_point[1], end_point[1], num_points)
    interp = RegularGridInterpolator(
        (y, x), field, bounds_error=False, fill_value=np.nan
    )
    line_values = interp(np.column_stack([line_y, line_x]))
    distance = np.sqrt((line_x - start_point[0]) ** 2 + (line_y - start_point[1]) ** 2)
    return distance, line_values, line_x, line_y


def _analyze_boundary_layer(
    x, y, u_field, v_field, wall_y, start_x, end_x, num_profiles=5
):
    """Analyze boundary layer profiles at multiple locations."""
    x_locations = np.linspace(start_x, end_x, num_profiles)
    profiles = []
    for x_loc in x_locations:
        x_idx = np.argmin(np.abs(x - x_loc))
        wall_idx = np.argmin(np.abs(y - wall_y))
        y_profile = y[wall_idx:]
        u_profile = u_field[wall_idx:, x_idx]
        wall_distance = y_profile - wall_y
        u_freestream = u_profile[-1]
        bl_thickness_idx = np.where(u_profile >= 0.99 * u_freestream)[0]
        bl_thickness = (
            wall_distance[bl_thickness_idx[0]] if len(bl_thickness_idx) > 0 else np.nan
        )
        profiles.append(
            {
                "x_location": x_loc,
                "wall_distance": wall_distance,
                "u_velocity": u_profile,
                "v_velocity": v_field[wall_idx:, x_idx],
                "bl_thickness": bl_thickness,
            }
        )
    return profiles


def create_cross_section_analysis(data, output_dir):
    """Create comprehensive 12-panel cross-sectional analysis."""
    from cfd_viz.analysis.flow_features import (
        compute_cross_sectional_averages,
        compute_spatial_fluctuations,
        detect_wake_regions,
    )

    os.makedirs(output_dir, exist_ok=True)

    x, y = data["x"], data["y"]
    fields = data["data_fields"]
    u, v = fields["u_velocity"], fields["v_velocity"]
    pressure = fields.get("pressure", np.zeros_like(u))
    X, Y = np.meshgrid(x, y)

    plt.figure(figsize=(20, 16))

    # 1. Velocity magnitude with analysis lines
    ax1 = plt.subplot(3, 4, 1)
    velocity_mag = np.sqrt(u**2 + v**2)
    cs1 = ax1.contourf(X, Y, velocity_mag, levels=20, cmap="viridis")
    plt.colorbar(cs1, ax=ax1, label="Velocity Magnitude (m/s)")

    lines = [
        {
            "start": (x.min(), y.mean()),
            "end": (x.max(), y.mean()),
            "name": "Horizontal Centerline",
            "color": "red",
        },
        {
            "start": (x.mean(), y.min()),
            "end": (x.mean(), y.max()),
            "name": "Vertical Centerline",
            "color": "white",
        },
        {
            "start": (x[len(x) // 4], y.min()),
            "end": (x[len(x) // 4], y.max()),
            "name": "Quarter Section",
            "color": "yellow",
        },
        {
            "start": (x[3 * len(x) // 4], y.min()),
            "end": (x[3 * len(x) // 4], y.max()),
            "name": "Three-Quarter Section",
            "color": "cyan",
        },
    ]
    for line in lines:
        ax1.plot(
            [line["start"][0], line["end"][0]],
            [line["start"][1], line["end"][1]],
            color=line["color"],
            linewidth=2,
            linestyle="--",
            label=line["name"],
        )
    ax1.set_title("Velocity Field with Analysis Lines")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    ax1.legend(fontsize=8)
    ax1.set_aspect("equal")

    # 2-5. Line plots for each analysis line
    for i, line in enumerate(lines):
        ax = plt.subplot(3, 4, i + 2)
        distance, u_line, _, _ = _extract_line_data(x, y, u, line["start"], line["end"])
        _, v_line, _, _ = _extract_line_data(x, y, v, line["start"], line["end"])
        _, p_line, _, _ = _extract_line_data(x, y, pressure, line["start"], line["end"])
        ax.plot(distance, u_line, "b-", label="u-velocity", linewidth=1.5)
        ax.plot(distance, v_line, "r-", label="v-velocity", linewidth=1.5)
        ax.plot(
            distance, np.sqrt(u_line**2 + v_line**2), "k-", label="|v|", linewidth=2
        )
        ax2 = ax.twinx()
        ax2.plot(distance, p_line, "g--", label="pressure", alpha=0.7)
        ax2.set_ylabel("Pressure", color="g")
        ax2.tick_params(axis="y", labelcolor="g")
        ax.set_title(f"{line['name']}")
        ax.set_xlabel("Distance along line (m)")
        ax.set_ylabel("Velocity (m/s)")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

    # 6. Boundary layer analysis
    ax6 = plt.subplot(3, 4, 6)
    wall_y = y.min()
    bl_profiles = _analyze_boundary_layer(
        x, y, u, v, wall_y, x[len(x) // 4], x[3 * len(x) // 4]
    )
    colors = plt.cm.plasma(np.linspace(0, 1, len(bl_profiles)))
    for profile, color in zip(bl_profiles, colors):
        if not np.isnan(profile["bl_thickness"]) and profile["bl_thickness"] > 0:
            y_norm = profile["wall_distance"] / profile["bl_thickness"]
            u_norm = profile["u_velocity"] / np.max(profile["u_velocity"])
            ax6.plot(
                u_norm,
                y_norm,
                color=color,
                linewidth=2,
                label=f"x={profile['x_location']:.2f}",
            )
    ax6.set_xlabel("u/u_max")
    ax6.set_ylabel("y/\u03b4 (normalized wall distance)")
    ax6.set_title("Boundary Layer Profiles")
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim(0, 2)

    # 7. Velocity profiles at different x-locations
    ax7 = plt.subplot(3, 4, 7)
    x_stations = [
        x[len(x) // 6],
        x[len(x) // 3],
        x[len(x) // 2],
        x[2 * len(x) // 3],
        x[5 * len(x) // 6],
    ]
    colors7 = plt.cm.viridis(np.linspace(0, 1, len(x_stations)))
    for x_station, color in zip(x_stations, colors7):
        _, u_profile, _, y_profile = _extract_line_data(
            x, y, u, (x_station, y.min()), (x_station, y.max())
        )
        ax7.plot(
            u_profile, y_profile, color=color, linewidth=2, label=f"x={x_station:.2f}"
        )
    ax7.set_xlabel("u-velocity (m/s)")
    ax7.set_ylabel("y (m)")
    ax7.set_title("Velocity Profiles at Different Stations")
    ax7.legend(fontsize=8)
    ax7.grid(True, alpha=0.3)

    # 8. Pressure distribution along centerline
    ax8 = plt.subplot(3, 4, 8)
    _, p_horizontal, x_line, _ = _extract_line_data(
        x, y, pressure, (x.min(), y.mean()), (x.max(), y.mean())
    )
    ax8.plot(x_line, p_horizontal, "b-", linewidth=2, label="Centerline Pressure")
    dp_dx = np.gradient(p_horizontal, x_line)
    ax8_twin = ax8.twinx()
    ax8_twin.plot(x_line, dp_dx, "r--", alpha=0.7, label="Pressure Gradient")
    ax8_twin.set_ylabel("dp/dx", color="r")
    ax8_twin.tick_params(axis="y", labelcolor="r")
    ax8.set_xlabel("x (m)")
    ax8.set_ylabel("Pressure")
    ax8.set_title("Pressure Distribution")
    ax8.legend(loc="upper left")
    ax8.grid(True, alpha=0.3)

    # 9. Wake analysis
    ax9 = plt.subplot(3, 4, 9)
    wake_result = detect_wake_regions(u, v, threshold_fraction=0.1)
    ax9.contourf(X, Y, velocity_mag, levels=20, cmap="viridis", alpha=0.7)
    ax9.contour(
        X, Y, wake_result.mask.astype(int), levels=[0.5], colors="red", linewidths=2
    )
    ax9.set_title(f"Wake Regions ({wake_result.area_fraction:.1%} of domain)")
    ax9.set_xlabel("x (m)")
    ax9.set_ylabel("y (m)")
    ax9.set_aspect("equal")

    # 10. Velocity fluctuations
    ax10 = plt.subplot(3, 4, 10)
    fluct_result = compute_spatial_fluctuations(u, v, averaging_axis=1)
    cs10 = ax10.contourf(X, Y, fluct_result.fluct_magnitude, levels=15, cmap="hot")
    plt.colorbar(cs10, ax=ax10, label="Velocity Fluctuation Magnitude")
    ax10.set_title(
        f"Velocity Fluctuations (TI={fluct_result.turbulence_intensity:.1%})"
    )
    ax10.set_xlabel("x (m)")
    ax10.set_ylabel("y (m)")
    ax10.set_aspect("equal")

    # 11. Statistics summary
    ax11 = plt.subplot(3, 4, 11)
    ax11.axis("off")
    stats_text = (
        f"\n    Flow Statistics:\n\n"
        f"    Domain:\n"
        f"    x: [{x.min():.3f}, {x.max():.3f}] m\n"
        f"    y: [{y.min():.3f}, {y.max():.3f}] m\n"
        f"    Grid: {len(x)} x {len(y)}\n\n"
        f"    Velocity:\n"
        f"    Max |v|: {np.max(velocity_mag):.3f} m/s\n"
        f"    Mean |v|: {np.mean(velocity_mag):.3f} m/s\n"
        f"    Min |v|: {np.min(velocity_mag):.3f} m/s\n\n"
        f"    Pressure:\n"
        f"    Max p: {np.max(pressure):.3f}\n"
        f"    Mean p: {np.mean(pressure):.3f}\n"
        f"    Min p: {np.min(pressure):.3f}\n\n"
        f"    Boundary Layer:\n"
    )
    for profile in bl_profiles:
        if not np.isnan(profile["bl_thickness"]):
            stats_text += f"      \u03b4 at x={profile['x_location']:.2f}: {profile['bl_thickness']:.4f} m\n"
    ax11.text(
        0.05,
        0.95,
        stats_text,
        transform=ax11.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
    )

    # 12. Cross-sectional averages
    ax12 = plt.subplot(3, 4, 12)
    avg_result = compute_cross_sectional_averages(u, v, x, y, averaging_axis="x")
    ax12.plot(
        avg_result.u_avg, avg_result.coordinate, "b-", linewidth=2, label="<u> vs y"
    )
    ax12.plot(
        avg_result.v_avg, avg_result.coordinate, "r-", linewidth=2, label="<v> vs y"
    )
    ax12.set_xlabel("Average Velocity (m/s)")
    ax12.set_ylabel("y (m)")
    ax12.set_title(f"Cross-sectional Averages (bulk={avg_result.bulk_velocity:.3f})")
    ax12.legend()
    ax12.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = os.path.join(output_dir, "cross_section_analysis.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Cross-section analysis saved to: {output_file}")
    plt.close()


# ── Dashboard helpers (from scripts/create_dashboard.py) ──────────────


def load_dashboard_vtk_files(vtk_files):
    """Load VTK files for dashboard creation. Returns (frames_list, time_indices)."""
    from cfd_viz.common import read_vtk_file

    frames_list = []
    time_indices = []
    for filename in sorted(vtk_files):
        try:
            data = read_vtk_file(filename)
            if data is None:
                continue
            u, v, p = data.fields.get("u"), data.fields.get("v"), data.fields.get("p")
            if u is not None and v is not None:
                frames_list.append((data.x, data.y, u, v, p))
                iteration = int(os.path.basename(filename).split("_")[-1].split(".")[0])
                time_indices.append(iteration)
        except Exception as e:
            print(f"Error reading {filename}: {e}")
    return frames_list, time_indices


def create_dashboards(vtk_files, output_dir, auto_open=False):
    """Create interactive dashboards from VTK files."""
    from plotly.offline import plot

    from cfd_viz.interactive import (
        create_animated_dashboard,
        create_convergence_figure,
        create_interactive_frame_collection,
    )

    frames_list, time_indices = load_dashboard_vtk_files(vtk_files)
    if not frames_list:
        print("No valid data found!")
        return

    print(f"Loaded {len(frames_list)} frames")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    collection = create_interactive_frame_collection(
        frames_list, time_indices=time_indices
    )

    print("Creating animated dashboard...")
    dashboard_fig = create_animated_dashboard(collection)
    dashboard_path = output_dir / "cfd_interactive_dashboard.html"
    plot(dashboard_fig, filename=str(dashboard_path), auto_open=auto_open)
    print(f"Animated dashboard saved to: {dashboard_path}")

    print("Creating convergence plot...")
    convergence_fig = create_convergence_figure(collection)
    convergence_path = output_dir / "cfd_convergence.html"
    plot(convergence_fig, filename=str(convergence_path), auto_open=auto_open)
    print(f"Convergence plot saved to: {convergence_path}")


# ── Monitor helpers (from scripts/create_monitor.py) ──────────────────


def run_monitor(watch_dir, output_dir, interval=2.0, manual=False):
    """Start the real-time CFD monitoring dashboard."""
    import glob as _glob
    import time

    import pandas as pd
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer

    from cfd_viz.analysis.time_series import (
        compute_flow_metrics,
        create_flow_metrics_time_series,
    )
    from cfd_viz.common import read_vtk_file

    def _read_vtk_for_monitor(filename):
        try:
            data = read_vtk_file(filename)
        except (FileNotFoundError, PermissionError, ValueError):
            return None
        if data is None:
            return None
        field_mapping = {
            "u": "u_velocity",
            "v": "v_velocity",
            "p": "pressure",
            "rho": "density",
            "T": "temperature",
        }
        data_fields = {}
        for old, new in field_mapping.items():
            if old in data.fields:
                data_fields[new] = data.fields[old]
        for name, field in data.fields.items():
            if name not in field_mapping:
                data_fields[name] = field
        if "u_velocity" not in data_fields or "v_velocity" not in data_fields:
            return None
        return {
            "x": data.x,
            "y": data.y,
            "data_fields": data_fields,
            "nx": data.nx,
            "ny": data.ny,
            "dx": data.dx,
            "dy": data.dy,
            "filename": filename,
            "timestamp": time.time(),
        }

    class _CFDMonitor:
        def __init__(self, wd, od):
            self.watch_dir, self.output_dir = wd, od
            self.monitoring = False
            self.monitoring_history = create_flow_metrics_time_series(max_length=100)
            self.fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            self.fig.suptitle("CFD Real-time Monitoring Dashboard", fontsize=16)
            self.axes = axes.flatten()
            self.last_processed_file = None
            os.makedirs(od, exist_ok=True)
            for ax in self.axes:
                ax.set_visible(True)
            plt.tight_layout()

        def process_new_file(self, filepath):
            print(f"Processing new file: {os.path.basename(filepath)}")
            data = _read_vtk_for_monitor(filepath)
            if data is None:
                return
            fields = data["data_fields"]
            u, v = fields["u_velocity"], fields["v_velocity"]
            p = fields.get("pressure", np.zeros_like(u))
            metrics = compute_flow_metrics(
                u=u, v=v, p=p, dx=data["dx"], dy=data["dy"], timestamp=data["timestamp"]
            )
            self.monitoring_history.add(metrics)
            self.last_processed_file = filepath
            self._save_metrics()

        def _save_metrics(self):
            if not self.monitoring_history.snapshots:
                return
            keys = [
                "timestamp",
                "max_velocity",
                "mean_velocity",
                "max_pressure",
                "mean_pressure",
                "total_kinetic_energy",
                "max_vorticity",
            ]
            df = pd.DataFrame(
                {k: self.monitoring_history.get_metric_array(k) for k in keys}
            )
            df.to_csv(
                os.path.join(self.output_dir, "monitoring_metrics.csv"), index=False
            )

    class _VTKHandler(FileSystemEventHandler):
        def __init__(self, mon):
            self.monitor = mon

        def on_created(self, event):
            if not event.is_directory and event.src_path.endswith(".vtk"):
                time.sleep(0.5)
                self.monitor.process_new_file(event.src_path)

    if not os.path.exists(watch_dir):
        print(f"Watch directory does not exist: {watch_dir}")
        return

    monitor = _CFDMonitor(watch_dir, output_dir)
    plt.ion()
    plt.show()

    if manual:
        print(
            f"Manual monitoring mode. Checking {watch_dir} every {interval} seconds..."
        )
        print("Press Ctrl+C to stop.")
        last_file = None
        try:
            while True:
                vtk_files = _glob.glob(os.path.join(watch_dir, "*.vtk"))
                if vtk_files:
                    latest = max(vtk_files, key=os.path.getctime)
                    if latest != last_file:
                        monitor.process_new_file(latest)
                        last_file = latest
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
    else:
        print(f"Monitoring {watch_dir}. Close plot window to stop.")
        existing = sorted(_glob.glob(os.path.join(watch_dir, "*.vtk")))
        if existing:
            monitor.process_new_file(existing[-1])
        observer = Observer()
        observer.schedule(_VTKHandler(monitor), watch_dir, recursive=False)
        observer.start()
        monitor.monitoring = True
        try:
            while monitor.monitoring:
                plt.pause(1.0)
        except KeyboardInterrupt:
            pass
        finally:
            observer.stop()
            observer.join()

    print("Monitoring complete!")


# ── Progress helper ───────────────────────────────────────────────────


def _progress_stderr(current, total, label=""):
    """Write a simple progress line to stderr."""
    width = 30
    filled = int(width * current / total) if total > 0 else width
    bar = "#" * filled + "-" * (width - filled)
    pct = (current / total * 100) if total > 0 else 100
    sys.stderr.write(f"\r  [{bar}] {pct:5.1f}%  {label}")
    if current >= total:
        sys.stderr.write("\n")
    sys.stderr.flush()
