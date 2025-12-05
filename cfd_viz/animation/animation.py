#!/usr/bin/env python3
"""
CFD Animation and Visualization Module
=======================================

Comprehensive visualization toolkit for CFD simulation results from VTK files.

Features:
- 2D flow field animations (velocity, pressure, vorticity)
- 3D rotating surface plots
- Lagrangian particle trace animations
- Vorticity analysis with streamlines
- Statistical time-series plots
- Static flow visualizations
- Individual frame export

Uses the shared vtk_reader module for file parsing.
"""

import os
from typing import Dict, List, Optional, Tuple

import matplotlib
import numpy as np
from numpy.typing import NDArray

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from cfd_viz.common import ANIMATIONS_DIR, PLOTS_DIR
from cfd_viz.common import read_vtk_file as _read_vtk


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


def create_velocity_magnitude(u: NDArray, v: NDArray) -> NDArray:
    """Calculate velocity magnitude from u and v components."""
    return np.sqrt(u**2 + v**2)


# =============================================================================
# Colormaps
# =============================================================================


def create_custom_colormap() -> LinearSegmentedColormap:
    """Create a custom colormap for better CFD visualization."""
    colors = [
        "#000080",
        "#0000FF",
        "#00FFFF",
        "#FFFF00",
        "#FF8000",
        "#FF0000",
        "#800000",
    ]
    n_bins = 256
    return LinearSegmentedColormap.from_list("cfd_custom", colors, N=n_bins)


def create_velocity_colormap() -> LinearSegmentedColormap:
    """Create a velocity-specific colormap."""
    colors = ["#000080", "#0080FF", "#00FFFF", "#FFFF00", "#FF8000", "#FF0000"]
    return LinearSegmentedColormap.from_list("velocity", colors)


# =============================================================================
# Basic Static Plot Functions
# =============================================================================


def create_static_plot(
    X: NDArray,
    Y: NDArray,
    data: NDArray,
    title: str,
    output_file: str,
    cmap: str = "viridis",
    label: str = "Value",
) -> None:
    """Create a single static contour plot."""
    plt.figure(figsize=(10, 6))

    contour = plt.contourf(X, Y, data, levels=20, cmap=cmap)
    plt.colorbar(contour, label=label)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)
    plt.grid(True, alpha=0.3)

    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved: {output_file}")


def create_static_plots(vtk_files: List[str], field: Optional[str] = None) -> None:
    """Create static PNG plots from VTK files."""
    print("\nCreating static plots...")

    for filename in vtk_files:
        print(f"Processing {os.path.basename(filename)}...")

        try:
            X, Y, data_fields = read_vtk_file(filename)
            base_name = os.path.splitext(os.path.basename(filename))[0]

            if field:
                # Plot specific field
                if (
                    field == "velocity_magnitude"
                    and "u" in data_fields
                    and "v" in data_fields
                ):
                    data = create_velocity_magnitude(data_fields["u"], data_fields["v"])
                    output_file = str(PLOTS_DIR / f"{base_name}_velocity_magnitude.png")
                    create_static_plot(
                        X,
                        Y,
                        data,
                        f"Velocity Magnitude - {base_name}",
                        output_file,
                        "viridis",
                        "Velocity Magnitude",
                    )
                elif field in data_fields:
                    output_file = str(PLOTS_DIR / f"{base_name}_{field}.png")
                    create_static_plot(
                        X,
                        Y,
                        data_fields[field],
                        f"{field} - {base_name}",
                        output_file,
                        "viridis",
                        field,
                    )
                else:
                    print(f"  Warning: Field '{field}' not found")
            else:
                # Plot all available fields
                if "u" in data_fields and "v" in data_fields:
                    vel_mag = create_velocity_magnitude(
                        data_fields["u"], data_fields["v"]
                    )
                    output_file = str(PLOTS_DIR / f"{base_name}_velocity_magnitude.png")
                    create_static_plot(
                        X,
                        Y,
                        vel_mag,
                        f"Velocity Magnitude - {base_name}",
                        output_file,
                        "viridis",
                        "Velocity Magnitude",
                    )

                if "p" in data_fields:
                    output_file = str(PLOTS_DIR / f"{base_name}_pressure.png")
                    create_static_plot(
                        X,
                        Y,
                        data_fields["p"],
                        f"Pressure - {base_name}",
                        output_file,
                        "coolwarm",
                        "Pressure",
                    )

        except Exception as e:
            print(f"  Error processing {filename}: {e}")

    print(f"\nStatic plots saved to {PLOTS_DIR}")


# =============================================================================
# Basic Animation Functions
# =============================================================================


def animate_field(
    vtk_files: List[str],
    field_name: str = "velocity_magnitude",
    output_prefix: Optional[str] = None,
) -> Optional[animation.FuncAnimation]:
    """Create animation of a specific field.

    Args:
        vtk_files: List of VTK files to animate
        field_name: Field to animate (default: velocity_magnitude)
        output_prefix: Prefix for output filename (default: 'cfd_animation')
    """
    print(f"\nCreating {field_name} animation...")

    # Read all files and extract data
    frames_data = []
    times = []
    X, Y = None, None

    for filename in sorted(vtk_files):
        try:
            X, Y, data_fields = read_vtk_file(filename)

            if field_name == "velocity_magnitude":
                if "u" in data_fields and "v" in data_fields:
                    field_data = create_velocity_magnitude(
                        data_fields["u"], data_fields["v"]
                    )
                else:
                    continue
            elif field_name in data_fields:
                field_data = data_fields[field_name]
            else:
                continue

            frames_data.append(field_data)
            times.append(extract_iteration_from_filename(filename, len(times)))

        except Exception as e:
            print(f"  Error reading {filename}: {e}")
            continue

    if not frames_data:
        print(f"  No valid data found for {field_name}!")
        return None

    print(f"  Found {len(frames_data)} frames")

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))

    # Calculate global min/max for consistent colormap
    vmin = min(np.min(frame) for frame in frames_data)
    vmax = max(np.max(frame) for frame in frames_data)

    # Create initial plot
    im = ax.imshow(
        frames_data[0],
        extent=[X.min(), X.max(), Y.min(), Y.max()],
        origin="lower",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(field_name.replace("_", " ").title())

    # Set labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"CFD Simulation - {field_name.replace('_', ' ').title()}")

    # Add iteration counter
    time_text = ax.text(
        0.02,
        0.95,
        "",
        transform=ax.transAxes,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    def animate(frame):
        im.set_array(frames_data[frame])
        time_text.set_text(f"Iteration: {times[frame]}")
        return [im, time_text]

    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, frames=len(frames_data), interval=200, blit=True, repeat=True
    )

    # Save animation
    prefix = output_prefix or "cfd_animation"
    output_file = str(ANIMATIONS_DIR / f"{prefix}_{field_name}.gif")
    print(f"  Saving animation to {output_file}...")
    anim.save(output_file, writer="pillow", fps=5)
    print("  Animation saved!")

    plt.close()
    return anim


def create_streamline_animation(
    vtk_files: List[str],
) -> Optional[animation.FuncAnimation]:
    """Create streamline animation."""
    print("\nCreating streamline animation...")

    frames_data = []
    times = []
    X, Y = None, None

    for filename in sorted(vtk_files):
        try:
            X, Y, data_fields = read_vtk_file(filename)

            if "u" in data_fields and "v" in data_fields:
                frames_data.append((data_fields["u"], data_fields["v"]))
                times.append(extract_iteration_from_filename(filename, len(times)))
        except Exception as e:
            print(f"  Error reading {filename}: {e}")
            continue

    if not frames_data:
        print("  No valid velocity data found!")
        return None

    print(f"  Found {len(frames_data)} frames")

    fig, ax = plt.subplots(figsize=(12, 6))

    # For streamplot, we need 1D arrays for x and y coordinates
    x = X[0, :] if len(X.shape) == 2 else X
    y = Y[:, 0] if len(Y.shape) == 2 else Y

    def animate(frame):
        ax.clear()
        u_frame, v_frame = frames_data[frame]

        # Create streamlines - streamplot expects 1D x, y and 2D u, v
        if u_frame.shape == (len(x), len(y)):
            u_frame = u_frame.T
            v_frame = v_frame.T
        elif u_frame.shape != (len(y), len(x)):
            print(
                f"Warning: Frame {frame} shape {u_frame.shape} doesn't match grid ({len(y)}, {len(x)})"
            )

        # Calculate speed for coloring
        speed = np.sqrt(u_frame**2 + v_frame**2)

        ax.streamplot(
            x,
            y,
            u_frame,
            v_frame,
            color=speed,
            cmap="viridis",
            density=1.5,
            linewidth=0.5,
        )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"CFD Streamlines - Iteration: {times[frame]}")
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())

    anim = animation.FuncAnimation(
        fig, animate, frames=len(frames_data), interval=200, repeat=True
    )

    output_file = str(ANIMATIONS_DIR / "cfd_streamlines.gif")
    print(f"  Saving streamline animation to {output_file}...")
    anim.save(output_file, writer="pillow", fps=5)
    print("  Animation saved!")

    plt.close()
    return anim


def create_animations(
    vtk_files: List[str],
    field: Optional[str] = None,
    include_streamlines: bool = False,
    output_prefix: Optional[str] = None,
) -> None:
    """Create animated GIFs from VTK files.

    Args:
        vtk_files: List of VTK files to animate
        field: Specific field to animate (default: all standard fields)
        include_streamlines: Whether to create streamline animation
        output_prefix: Prefix for output filenames (e.g., 'high_reynolds')
    """
    if field:
        animate_field(vtk_files, field, output_prefix=output_prefix)
    else:
        # Create all standard animations
        animate_field(vtk_files, "velocity_magnitude", output_prefix=output_prefix)
        animate_field(vtk_files, "p", output_prefix=output_prefix)
        if include_streamlines:
            try:
                create_streamline_animation(vtk_files)
            except Exception as e:
                print(f"  Streamline animation failed: {e}")

    print(f"\nAnimations saved to {ANIMATIONS_DIR}")


# =============================================================================
# Advanced Animation Functions
# =============================================================================


def animate_flow_fields_2x3_grid(
    vtk_files: List[str], save_animation: bool = True
) -> Optional[animation.FuncAnimation]:
    """Animate velocity, pressure, vorticity, vectors, streamlines in a 2x3 grid layout."""

    frames_data = []
    times = []
    X, Y = None, None

    for filename in sorted(vtk_files):
        try:
            X, Y, data_fields = read_vtk_file(filename)

            if "u" in data_fields and "v" in data_fields and "p" in data_fields:
                velocity_mag = np.sqrt(data_fields["u"] ** 2 + data_fields["v"] ** 2)
                vorticity = np.gradient(data_fields["v"], axis=1) - np.gradient(
                    data_fields["u"], axis=0
                )

                frames_data.append(
                    {
                        "u": data_fields["u"],
                        "v": data_fields["v"],
                        "p": data_fields["p"],
                        "velocity_mag": velocity_mag,
                        "vorticity": vorticity,
                    }
                )

                times.append(extract_iteration_from_filename(filename, len(times)))

        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue

    if not frames_data:
        print("No valid data found!")
        return None

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("CFD Flow Analysis Dashboard", fontsize=16, fontweight="bold")

    vel_min = min(np.min(frame["velocity_mag"]) for frame in frames_data)
    vel_max = max(np.max(frame["velocity_mag"]) for frame in frames_data)
    p_min = min(np.min(frame["p"]) for frame in frames_data)
    p_max = max(np.max(frame["p"]) for frame in frames_data)
    vort_min = min(np.min(frame["vorticity"]) for frame in frames_data)
    vort_max = max(np.max(frame["vorticity"]) for frame in frames_data)

    velocity_cmap = create_velocity_colormap()

    def animate(frame):
        for ax in axes.flat:
            ax.clear()

        data = frames_data[frame]

        axes[0, 0].imshow(
            data["velocity_mag"],
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            origin="lower",
            aspect="auto",
            vmin=vel_min,
            vmax=vel_max,
            cmap=velocity_cmap,
        )
        axes[0, 0].set_title("Velocity Magnitude", fontweight="bold")
        axes[0, 0].set_xlabel("X")
        axes[0, 0].set_ylabel("Y")

        axes[0, 1].imshow(
            data["p"],
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            origin="lower",
            aspect="auto",
            vmin=p_min,
            vmax=p_max,
            cmap="RdBu_r",
        )
        contours = axes[0, 1].contour(
            X, Y, data["p"], levels=8, colors="black", linewidths=0.5
        )
        axes[0, 1].clabel(contours, inline=True, fontsize=8)
        axes[0, 1].set_title("Pressure Field", fontweight="bold")
        axes[0, 1].set_xlabel("X")
        axes[0, 1].set_ylabel("Y")

        axes[0, 2].imshow(
            data["vorticity"],
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            origin="lower",
            aspect="auto",
            vmin=vort_min,
            vmax=vort_max,
            cmap="seismic",
        )
        axes[0, 2].set_title("Vorticity", fontweight="bold")
        axes[0, 2].set_xlabel("X")
        axes[0, 2].set_ylabel("Y")

        skip = 5
        X_sub = X[::skip, ::skip]
        Y_sub = Y[::skip, ::skip]
        U_sub = data["u"][::skip, ::skip]
        V_sub = data["v"][::skip, ::skip]
        vel_sub = data["velocity_mag"][::skip, ::skip]

        axes[1, 0].quiver(
            X_sub,
            Y_sub,
            U_sub,
            V_sub,
            vel_sub,
            cmap=velocity_cmap,
            scale=15,
            width=0.003,
        )
        axes[1, 0].set_xlim(X.min(), X.max())
        axes[1, 0].set_ylim(Y.min(), Y.max())
        axes[1, 0].set_title("Vector Field", fontweight="bold")
        axes[1, 0].set_xlabel("X")
        axes[1, 0].set_ylabel("Y")

        axes[1, 1].streamplot(
            X,
            Y,
            data["u"],
            data["v"],
            color=data["velocity_mag"],
            cmap=velocity_cmap,
            density=2,
            linewidth=1.5,
        )
        axes[1, 1].set_xlim(X.min(), X.max())
        axes[1, 1].set_ylim(Y.min(), Y.max())
        axes[1, 1].set_title("Streamlines", fontweight="bold")
        axes[1, 1].set_xlabel("X")
        axes[1, 1].set_ylabel("Y")

        axes[1, 2].imshow(
            data["velocity_mag"],
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            origin="lower",
            aspect="auto",
            vmin=vel_min,
            vmax=vel_max,
            cmap=velocity_cmap,
            alpha=0.8,
        )
        axes[1, 2].contour(X, Y, data["p"], levels=6, colors="white", linewidths=1.5)
        vort_threshold = np.percentile(np.abs(data["vorticity"]), 85)
        high_vort_mask = np.abs(data["vorticity"]) > vort_threshold
        axes[1, 2].contour(
            X, Y, high_vort_mask, levels=[0.5], colors="red", linewidths=2
        )
        axes[1, 2].set_xlim(X.min(), X.max())
        axes[1, 2].set_ylim(Y.min(), Y.max())
        axes[1, 2].set_title("Combined Analysis", fontweight="bold")
        axes[1, 2].set_xlabel("X")
        axes[1, 2].set_ylabel("Y")

        fig.suptitle(
            f"CFD Flow Analysis - Iteration: {times[frame]}",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()

    anim = animation.FuncAnimation(
        fig, animate, frames=len(frames_data), interval=600, repeat=True
    )

    if save_animation:
        output_file = str(ANIMATIONS_DIR / "cfd_flow_analysis_2x3.gif")
        print(f"Saving animation to {output_file}...")
        try:
            anim.save(output_file, writer="pillow", fps=2)
            print(f"Animation saved as {output_file}")
        except Exception as e:
            print(f"Error saving animation: {e}")

    plt.close()
    return anim


def animate_flow_fields_with_temperature(
    vtk_files: List[str], save_animation: bool = True
) -> Optional[animation.FuncAnimation]:
    """Animate flow fields with temperature in a 3x3 grid (8 panels + combined view)."""

    frames_data = []
    times = []
    X, Y = None, None

    for filename in sorted(vtk_files):
        try:
            X, Y, data_fields = read_vtk_file(filename)

            if "u" in data_fields and "v" in data_fields and "p" in data_fields:
                velocity_mag = np.sqrt(data_fields["u"] ** 2 + data_fields["v"] ** 2)
                vorticity = np.gradient(data_fields["v"], axis=1) - np.gradient(
                    data_fields["u"], axis=0
                )

                frames_data.append(
                    {
                        "u": data_fields["u"],
                        "v": data_fields["v"],
                        "p": data_fields["p"],
                        "velocity_mag": velocity_mag,
                        "vorticity": vorticity,
                        "T": data_fields.get("T", np.ones_like(velocity_mag) * 300),
                    }
                )

                times.append(extract_iteration_from_filename(filename, len(times)))

        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue

    if not frames_data:
        print("No valid data found!")
        return None

    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])
    ax7 = fig.add_subplot(gs[2, :])

    custom_cmap = create_custom_colormap()

    vel_min = min(np.min(frame["velocity_mag"]) for frame in frames_data)
    vel_max = max(np.max(frame["velocity_mag"]) for frame in frames_data)
    p_min = min(np.min(frame["p"]) for frame in frames_data)
    p_max = max(np.max(frame["p"]) for frame in frames_data)
    vort_min = min(np.min(frame["vorticity"]) for frame in frames_data)
    vort_max = max(np.max(frame["vorticity"]) for frame in frames_data)

    im1 = ax1.imshow(
        frames_data[0]["velocity_mag"],
        extent=[X.min(), X.max(), Y.min(), Y.max()],
        origin="lower",
        aspect="auto",
        vmin=vel_min,
        vmax=vel_max,
        cmap=custom_cmap,
    )
    im2 = ax2.imshow(
        frames_data[0]["p"],
        extent=[X.min(), X.max(), Y.min(), Y.max()],
        origin="lower",
        aspect="auto",
        vmin=p_min,
        vmax=p_max,
        cmap="RdBu_r",
    )
    im3 = ax3.imshow(
        frames_data[0]["vorticity"],
        extent=[X.min(), X.max(), Y.min(), Y.max()],
        origin="lower",
        aspect="auto",
        vmin=vort_min,
        vmax=vort_max,
        cmap="seismic",
    )

    skip = 4
    X_sub = X[::skip, ::skip]
    Y_sub = Y[::skip, ::skip]

    im6 = ax6.imshow(
        frames_data[0]["T"],
        extent=[X.min(), X.max(), Y.min(), Y.max()],
        origin="lower",
        aspect="auto",
        cmap="plasma",
        alpha=0.8,
    )

    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label("Velocity Magnitude")
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label("Pressure")
    cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.8)
    cbar3.set_label("Vorticity")
    cbar6 = plt.colorbar(im6, ax=ax6, shrink=0.8)
    cbar6.set_label("Temperature")

    ax1.set_title("Velocity Magnitude", fontsize=12, fontweight="bold")
    ax2.set_title("Pressure Field", fontsize=12, fontweight="bold")
    ax3.set_title("Vorticity", fontsize=12, fontweight="bold")
    ax4.set_title("Vector Field", fontsize=12, fontweight="bold")
    ax5.set_title("Streamlines", fontsize=12, fontweight="bold")
    ax6.set_title("Temperature", fontsize=12, fontweight="bold")
    ax7.set_title("Combined View with Contours", fontsize=14, fontweight="bold")

    time_text = fig.suptitle("", fontsize=16, fontweight="bold")

    def animate(frame):
        ax4.clear()
        ax5.clear()
        ax7.clear()

        data = frames_data[frame]

        im1.set_array(data["velocity_mag"])
        im2.set_array(data["p"])
        im3.set_array(data["vorticity"])
        im6.set_array(data["T"])

        u_sub = data["u"][::skip, ::skip]
        v_sub = data["v"][::skip, ::skip]
        vel_sub = data["velocity_mag"][::skip, ::skip]

        ax4.quiver(
            X_sub,
            Y_sub,
            u_sub,
            v_sub,
            vel_sub,
            cmap=custom_cmap,
            scale=10,
            width=0.002,
            alpha=0.8,
        )
        ax4.set_xlim(X.min(), X.max())
        ax4.set_ylim(Y.min(), Y.max())
        ax4.set_aspect("equal")
        ax4.set_title("Vector Field", fontsize=12, fontweight="bold")

        ax5.streamplot(
            X,
            Y,
            data["u"],
            data["v"],
            color=data["velocity_mag"],
            cmap=custom_cmap,
            density=2,
            linewidth=1.5,
            arrowsize=1.5,
        )
        ax5.set_xlim(X.min(), X.max())
        ax5.set_ylim(Y.min(), Y.max())
        ax5.set_aspect("equal")
        ax5.set_title("Streamlines", fontsize=12, fontweight="bold")

        ax7.imshow(
            data["velocity_mag"],
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            origin="lower",
            aspect="auto",
            vmin=vel_min,
            vmax=vel_max,
            cmap=custom_cmap,
            alpha=0.7,
        )
        contours = ax7.contour(
            X, Y, data["p"], levels=10, colors="white", linewidths=1, alpha=0.8
        )
        ax7.clabel(contours, inline=True, fontsize=8, fmt="%.2f")
        ax7.streamplot(
            X,
            Y,
            data["u"],
            data["v"],
            color="black",
            density=1,
            linewidth=0.8,
            arrowsize=1,
        )
        vort_thresh = np.percentile(np.abs(data["vorticity"]), 90)
        high_vort = np.abs(data["vorticity"]) > vort_thresh
        ax7.contour(X, Y, high_vort, levels=[0.5], colors="red", linewidths=2)
        ax7.set_xlim(X.min(), X.max())
        ax7.set_ylim(Y.min(), Y.max())
        ax7.set_xlabel("X", fontsize=12)
        ax7.set_ylabel("Y", fontsize=12)
        ax7.set_title(
            "Combined View: Velocity + Pressure Contours + Vorticity",
            fontsize=14,
            fontweight="bold",
        )

        time_text.set_text(
            f"CFD Simulation - Iteration: {times[frame]} | Time Step: {frame + 1}/{len(frames_data)}"
        )

        return [im1, im2, im3, im6]

    anim = animation.FuncAnimation(
        fig, animate, frames=len(frames_data), interval=500, repeat=True
    )

    if save_animation:
        output_file = str(ANIMATIONS_DIR / "cfd_flow_with_temperature.gif")
        print(f"Saving animation to {output_file}...")
        anim.save(output_file, writer="pillow", fps=2, dpi=100)
        print(f"Animation saved as {output_file}")

    plt.close()
    return anim


def animate_3d_rotating_surface(
    vtk_files: List[str], field: str = "velocity_mag", save_animation: bool = True
) -> Optional[animation.FuncAnimation]:
    """Animate a 3D surface plot with rotating camera view."""

    frames_data = []
    times = []
    X, Y = None, None

    for filename in sorted(vtk_files):
        try:
            X, Y, data_fields = read_vtk_file(filename)

            if field == "velocity_mag" and "u" in data_fields and "v" in data_fields:
                field_data = np.sqrt(data_fields["u"] ** 2 + data_fields["v"] ** 2)
            elif field in data_fields:
                field_data = data_fields[field]
            else:
                continue

            frames_data.append({"X": X, "Y": Y, "data": field_data})
            times.append(extract_iteration_from_filename(filename, len(times)))

        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue

    if not frames_data:
        print("No valid data found!")
        return None

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    vmin = min(np.min(frame["data"]) for frame in frames_data)
    vmax = max(np.max(frame["data"]) for frame in frames_data)

    X = frames_data[0]["X"]
    Y = frames_data[0]["Y"]

    def animate(frame):
        ax.clear()
        ax.plot_surface(
            X,
            Y,
            frames_data[frame]["data"],
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            alpha=0.8,
            linewidth=0,
            antialiased=True,
        )
        ax.contour(
            X,
            Y,
            frames_data[frame]["data"],
            zdir="z",
            offset=vmin - 0.1 * (vmax - vmin),
            cmap="viridis",
            alpha=0.5,
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel(field.replace("_", " ").title())
        ax.set_title(
            f"3D Surface - {field.replace('_', ' ').title()} - Iteration: {times[frame]}"
        )
        ax.set_xlim(X.min(), X.max())
        ax.set_ylim(Y.min(), Y.max())
        ax.set_zlim(vmin, vmax)
        ax.view_init(elev=30, azim=frame * 2)

    anim = animation.FuncAnimation(
        fig, animate, frames=len(frames_data), interval=200, repeat=True
    )

    if save_animation:
        output_file = str(ANIMATIONS_DIR / f"cfd_3d_{field}.gif")
        print(f"Saving 3D animation to {output_file}...")
        anim.save(output_file, writer="pillow", fps=5)
        print(f"3D animation saved as {output_file}")

    plt.close()
    return anim


def animate_lagrangian_particle_traces(
    vtk_files: List[str], save_animation: bool = True
) -> Optional[animation.FuncAnimation]:
    """Animate Lagrangian particle traces showing flow pathlines."""

    frames_data = []
    times = []
    X, Y = None, None

    for filename in sorted(vtk_files):
        try:
            X, Y, data_fields = read_vtk_file(filename)
            if "u" in data_fields and "v" in data_fields:
                frames_data.append(
                    {"u": data_fields["u"], "v": data_fields["v"], "X": X, "Y": Y}
                )
                times.append(extract_iteration_from_filename(filename, len(times)))
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue

    if not frames_data:
        print("No valid data found!")
        return None

    X = frames_data[0]["X"]
    Y = frames_data[0]["Y"]

    n_particles = 50
    particles_x = np.random.uniform(X.min(), X.min() + 0.1, n_particles)
    particles_y = np.random.uniform(Y.min(), Y.max(), n_particles)

    particle_histories_x = [[] for _ in range(n_particles)]
    particle_histories_y = [[] for _ in range(n_particles)]

    fig, ax = plt.subplots(figsize=(15, 8))

    def animate(frame):
        ax.clear()

        data = frames_data[frame % len(frames_data)]

        velocity_mag = np.sqrt(data["u"] ** 2 + data["v"] ** 2)
        ax.imshow(
            velocity_mag,
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            origin="lower",
            aspect="auto",
            cmap="Blues",
            alpha=0.6,
        )

        dt = 0.01
        for i in range(n_particles):
            if (
                X.min() <= particles_x[i] < X.max()
                and Y.min() <= particles_y[i] < Y.max()
            ):
                xi = int(
                    (particles_x[i] - X.min()) / (X.max() - X.min()) * (X.shape[1] - 1)
                )
                yi = int(
                    (particles_y[i] - Y.min()) / (Y.max() - Y.min()) * (Y.shape[0] - 1)
                )
                xi = max(0, min(xi, data["u"].shape[1] - 1))
                yi = max(0, min(yi, data["u"].shape[0] - 1))
                u_interp = data["u"][yi, xi]
                v_interp = data["v"][yi, xi]
                particles_x[i] += u_interp * dt
                particles_y[i] += v_interp * dt
                particle_histories_x[i].append(particles_x[i])
                particle_histories_y[i].append(particles_y[i])
                if len(particle_histories_x[i]) > 20:
                    particle_histories_x[i] = particle_histories_x[i][-20:]
                    particle_histories_y[i] = particle_histories_y[i][-20:]

            if (
                particles_x[i] >= X.max()
                or particles_x[i] < X.min()
                or particles_y[i] >= Y.max()
                or particles_y[i] < Y.min()
            ):
                particles_x[i] = X.min() + 0.01
                particles_y[i] = np.random.uniform(Y.min(), Y.max())
                particle_histories_x[i] = []
                particle_histories_y[i] = []

        for i in range(n_particles):
            if len(particle_histories_x[i]) > 1:
                alphas = np.linspace(0.2, 1.0, len(particle_histories_x[i]))
                for j in range(len(particle_histories_x[i]) - 1):
                    ax.plot(
                        [particle_histories_x[i][j], particle_histories_x[i][j + 1]],
                        [particle_histories_y[i][j], particle_histories_y[i][j + 1]],
                        "r-",
                        alpha=alphas[j],
                        linewidth=2,
                    )

        ax.scatter(
            particles_x,
            particles_y,
            c="red",
            s=30,
            alpha=0.8,
            edgecolors="white",
            linewidth=1,
        )
        ax.streamplot(
            data["X"],
            data["Y"],
            data["u"],
            data["v"],
            color="gray",
            density=1,
            linewidth=0.5,
        )
        ax.set_xlim(X.min(), X.max())
        ax.set_ylim(Y.min(), Y.max())
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(
            f"Particle Traces in Flow Field - Iteration: {times[frame % len(times)]}"
        )

    anim = animation.FuncAnimation(
        fig, animate, frames=len(frames_data) * 3, interval=100, repeat=True
    )

    if save_animation:
        output_file = str(ANIMATIONS_DIR / "cfd_particle_traces.gif")
        print(f"Saving particle trace animation to {output_file}...")
        anim.save(output_file, writer="pillow", fps=10)
        print(f"Particle trace animation saved as {output_file}")

    plt.close()
    return anim


def animate_vorticity_field_and_streamlines(
    vtk_files: List[str], save_animation: bool = True
) -> Optional[animation.FuncAnimation]:
    """Animate vorticity field alongside streamlines colored by vorticity."""

    frames_data = []
    times = []
    X, Y = None, None

    for filename in sorted(vtk_files):
        try:
            X, Y, data_fields = read_vtk_file(filename)
            if "u" in data_fields and "v" in data_fields:
                vorticity = np.gradient(data_fields["v"], axis=1) - np.gradient(
                    data_fields["u"], axis=0
                )
                velocity_mag = np.sqrt(data_fields["u"] ** 2 + data_fields["v"] ** 2)
                frames_data.append(
                    {
                        "u": data_fields["u"],
                        "v": data_fields["v"],
                        "vorticity": vorticity,
                        "velocity_mag": velocity_mag,
                        "X": X,
                        "Y": Y,
                    }
                )
                times.append(extract_iteration_from_filename(filename, len(times)))
        except Exception:
            continue

    if not frames_data:
        return None

    X = frames_data[0]["X"]
    Y = frames_data[0]["Y"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    vort_min = min(np.min(frame["vorticity"]) for frame in frames_data)
    vort_max = max(np.max(frame["vorticity"]) for frame in frames_data)

    def animate(frame):
        ax1.clear()
        ax2.clear()

        data = frames_data[frame]

        ax1.imshow(
            data["vorticity"],
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            origin="lower",
            aspect="auto",
            vmin=vort_min,
            vmax=vort_max,
            cmap="RdBu",
        )
        ax1.set_title("Vorticity Field", fontweight="bold")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")

        speed = data["velocity_mag"]
        lw = 5 * speed / speed.max()
        ax2.streamplot(
            X,
            Y,
            data["u"],
            data["v"],
            color=data["vorticity"],
            linewidth=lw,
            cmap="RdBu",
            density=2,
        )
        ax2.set_title("Streamlines Colored by Vorticity", fontweight="bold")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_xlim(X.min(), X.max())
        ax2.set_ylim(Y.min(), Y.max())

        fig.suptitle(
            f"Vorticity Analysis - Iteration: {times[frame]}",
            fontsize=14,
            fontweight="bold",
        )

    anim = animation.FuncAnimation(
        fig, animate, frames=len(frames_data), interval=400, repeat=True
    )

    if save_animation:
        output_file = str(ANIMATIONS_DIR / "cfd_vorticity_analysis.gif")
        print(f"Saving vorticity analysis to {output_file}...")
        try:
            anim.save(output_file, writer="pillow", fps=3)
            print(f"Vorticity analysis saved as {output_file}")
        except Exception as e:
            print(f"Error saving animation: {e}")

    plt.close()
    return anim


# =============================================================================
# Advanced Static Plot Functions
# =============================================================================


def plot_flow_statistics_over_time(vtk_files: List[str]) -> Optional[plt.Figure]:
    """Plot velocity, vorticity, pressure, and energy statistics over simulation time."""

    times = []
    max_velocities = []
    avg_velocities = []
    max_vorticity = []
    pressure_range = []

    for filename in sorted(vtk_files):
        try:
            X, Y, data_fields = read_vtk_file(filename)
            if "u" in data_fields and "v" in data_fields and "p" in data_fields:
                velocity_mag = np.sqrt(data_fields["u"] ** 2 + data_fields["v"] ** 2)
                vorticity = np.gradient(data_fields["v"], axis=1) - np.gradient(
                    data_fields["u"], axis=0
                )
                times.append(extract_iteration_from_filename(filename, len(times)))
                max_velocities.append(np.max(velocity_mag))
                avg_velocities.append(np.mean(velocity_mag))
                max_vorticity.append(np.max(np.abs(vorticity)))
                pressure_range.append(
                    np.max(data_fields["p"]) - np.min(data_fields["p"])
                )
        except Exception:
            continue

    if not times:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(times, max_velocities, "r-o", label="Max Velocity", linewidth=2)
    axes[0, 0].plot(times, avg_velocities, "b-s", label="Avg Velocity", linewidth=2)
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Velocity")
    axes[0, 0].set_title("Velocity Statistics", fontweight="bold")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(times, max_vorticity, "g-^", label="Max |Vorticity|", linewidth=2)
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("Vorticity")
    axes[0, 1].set_title("Vorticity Evolution", fontweight="bold")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(times, pressure_range, "m-d", label="Pressure Range", linewidth=2)
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel("Pressure Range")
    axes[1, 0].set_title("Pressure Variation", fontweight="bold")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    flow_energy = [0.5 * avg_vel**2 for avg_vel in avg_velocities]
    axes[1, 1].plot(
        times, flow_energy, "c-p", label="Kinetic Energy Density", linewidth=2
    )
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].set_ylabel("Energy")
    axes[1, 1].set_title("Flow Energy Evolution", fontweight="bold")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.suptitle("CFD Statistical Analysis", fontsize=16, fontweight="bold", y=1.02)

    output_file = str(PLOTS_DIR / "cfd_statistical_analysis.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Statistical analysis saved as '{output_file}'")
    return fig


def plot_velocity_magnitude(vtk_file: str, save_path: Optional[str] = None) -> None:
    """Plot velocity magnitude contours from a single VTK file."""
    X, Y, data_fields = read_vtk_file(vtk_file)

    if "u" not in data_fields or "v" not in data_fields:
        print("No velocity data found")
        return

    velocity_mag = np.sqrt(data_fields["u"] ** 2 + data_fields["v"] ** 2)

    plt.figure(figsize=(12, 6))
    plt.contourf(X, Y, velocity_mag, levels=20, cmap="viridis")
    plt.colorbar(label="Velocity Magnitude")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Velocity Magnitude Contours")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Velocity magnitude plot saved to: {save_path}")
    plt.close()


def plot_velocity_vectors(
    vtk_file: str, save_path: Optional[str] = None, subsample: int = 3
) -> None:
    """Plot velocity vector field from a single VTK file."""
    X, Y, data_fields = read_vtk_file(vtk_file)

    if "u" not in data_fields or "v" not in data_fields:
        print("No velocity data found")
        return

    u = data_fields["u"]
    v = data_fields["v"]

    X_sub = X[::subsample, ::subsample]
    Y_sub = Y[::subsample, ::subsample]
    u_sub = u[::subsample, ::subsample]
    v_sub = v[::subsample, ::subsample]

    plt.figure(figsize=(12, 6))
    plt.quiver(
        X_sub,
        Y_sub,
        u_sub,
        v_sub,
        np.sqrt(u_sub**2 + v_sub**2),
        cmap="plasma",
        scale_units="xy",
        angles="xy",
    )
    plt.colorbar(label="Velocity Magnitude")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Velocity Vector Field")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Velocity vectors plot saved to: {save_path}")
    plt.close()


def plot_streamlines(vtk_file: str, save_path: Optional[str] = None) -> None:
    """Plot streamlines from a single VTK file."""
    X, Y, data_fields = read_vtk_file(vtk_file)

    if "u" not in data_fields or "v" not in data_fields:
        print("No velocity data found")
        return

    u = data_fields["u"]
    v = data_fields["v"]

    plt.figure(figsize=(12, 6))
    plt.streamplot(
        X, Y, u, v, color=np.sqrt(u**2 + v**2), cmap="viridis", density=2, linewidth=1
    )
    plt.colorbar(label="Velocity Magnitude")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Flow Streamlines")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Streamlines plot saved to: {save_path}")
    plt.close()


def plot_comprehensive_flow(vtk_file: str, save_path: Optional[str] = None) -> None:
    """Create comprehensive 2x2 flow visualization from a single VTK file."""
    X, Y, data_fields = read_vtk_file(vtk_file)

    if "u" not in data_fields or "v" not in data_fields:
        print("No velocity data found")
        return

    u = data_fields["u"]
    v = data_fields["v"]
    velocity_mag = np.sqrt(u**2 + v**2)

    _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    im1 = ax1.contourf(X, Y, velocity_mag, levels=20, cmap="viridis")
    ax1.set_title("Velocity Magnitude Contours")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.axis("equal")
    plt.colorbar(im1, ax=ax1, label="Velocity Magnitude")

    subsample = max(1, min(X.shape[0], X.shape[1]) // 20)
    X_sub = X[::subsample, ::subsample]
    Y_sub = Y[::subsample, ::subsample]
    u_sub = u[::subsample, ::subsample]
    v_sub = v[::subsample, ::subsample]

    im2 = ax2.quiver(
        X_sub,
        Y_sub,
        u_sub,
        v_sub,
        np.sqrt(u_sub**2 + v_sub**2),
        cmap="plasma",
        scale_units="xy",
        angles="xy",
    )
    ax2.set_title("Velocity Vector Field")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.axis("equal")
    plt.colorbar(im2, ax=ax2, label="Velocity Magnitude")

    ax3.streamplot(
        X, Y, u, v, color=velocity_mag, cmap="viridis", density=2, linewidth=1
    )
    ax3.set_title("Flow Streamlines")
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.axis("equal")

    if "p" in data_fields:
        im4 = ax4.contourf(X, Y, data_fields["p"], levels=20, cmap="RdBu_r")
        ax4.set_title("Pressure Field")
        plt.colorbar(im4, ax=ax4, label="Pressure")
    else:
        im4 = ax4.contourf(X, Y, velocity_mag, levels=20, cmap="viridis", alpha=0.7)
        ax4.streamplot(X, Y, u, v, color="white", density=1, linewidth=0.8)
        ax4.set_title("Combined: Magnitude + Streamlines")
        plt.colorbar(im4, ax=ax4, label="Velocity Magnitude")

    ax4.set_xlabel("X")
    ax4.set_ylabel("Y")
    ax4.axis("equal")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Comprehensive flow plot saved to: {save_path}")
    plt.close()


# =============================================================================
# Frame Export Functions
# =============================================================================


def export_individual_frames(
    vtk_files: List[str], output_dir: Optional[str] = None
) -> None:
    """Export individual PNG frames for each VTK file.

    Args:
        vtk_files: List of VTK file paths
        output_dir: Directory to save frames (default: PLOTS_DIR/frames)
    """
    if output_dir is None:
        output_dir = str(PLOTS_DIR / "frames")

    os.makedirs(output_dir, exist_ok=True)

    print(f"Exporting {len(vtk_files)} frames to {output_dir}...")

    for i, filename in enumerate(sorted(vtk_files)):
        try:
            X, Y, data_fields = read_vtk_file(filename)
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue

        # Extract step number from filename
        step = extract_iteration_from_filename(filename, i)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f"CFD Flow Visualization - Step {step}", fontsize=16)

        if "u" in data_fields and "v" in data_fields:
            u = data_fields["u"]
            v = data_fields["v"]
            velocity_mag = np.sqrt(u**2 + v**2)

            im1 = ax1.contourf(X, Y, velocity_mag, levels=20, cmap="viridis")
            ax1.set_title("Velocity Magnitude")
            ax1.axis("equal")
            plt.colorbar(im1, ax=ax1)

            subsample = max(1, min(X.shape[0], X.shape[1]) // 20)
            X_sub = X[::subsample, ::subsample]
            Y_sub = Y[::subsample, ::subsample]
            u_sub = u[::subsample, ::subsample]
            v_sub = v[::subsample, ::subsample]

            ax2.quiver(
                X_sub,
                Y_sub,
                u_sub,
                v_sub,
                np.sqrt(u_sub**2 + v_sub**2),
                cmap="plasma",
                scale_units="xy",
                angles="xy",
            )
            ax2.set_title("Velocity Vectors")
            ax2.axis("equal")

            try:
                ax3.streamplot(
                    X, Y, u, v, color=velocity_mag, cmap="viridis", density=2
                )
            except ValueError:
                ax3.contourf(X, Y, velocity_mag, levels=20, cmap="viridis")
            ax3.set_title("Flow Streamlines")
            ax3.axis("equal")

        if "p" in data_fields:
            im4 = ax4.contourf(X, Y, data_fields["p"], levels=20, cmap="RdBu_r")
            ax4.set_title("Pressure Field")
            ax4.axis("equal")
            plt.colorbar(im4, ax=ax4)

        frame_filename = os.path.join(output_dir, f"frame_{step:04d}.png")
        plt.savefig(frame_filename, dpi=150, bbox_inches="tight")
        plt.close(fig)

        if (i + 1) % 10 == 0:
            print(f"Exported {i + 1}/{len(vtk_files)} frames")

    print(f"All frames saved to: {output_dir}")
