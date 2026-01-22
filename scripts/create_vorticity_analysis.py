#!/usr/bin/env python3
"""
Vorticity and Circulation Analysis Tool for CFD Framework
========================================================

This script provides comprehensive vorticity analysis including:
- Vorticity field calculation and visualization
- Circulation analysis around regions
- Vortex core detection
- Q-criterion visualization

Requirements:
    numpy, matplotlib, scipy

Usage:
    python vorticity_visualizer.py [vtk_file] [options]
"""

import argparse
import os
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

from cfd_viz.common import (
    DATA_DIR,
    PLOTS_DIR,
    ensure_dirs,
    find_vtk_files,
    read_vtk_file as _read_vtk_file,
)
from cfd_viz.fields.vorticity import (
    circulation,
    detect_vortex_cores,
    q_criterion,
    vorticity,
)


def read_vtk_file(filename: str) -> Optional[Dict[str, Any]]:
    """Read a VTK structured points file and extract velocity data.

    Args:
        filename: Path to the VTK file.

    Returns:
        Dictionary with keys: x, y, u, v, nx, ny, dx, dy
        or None if the file cannot be read.
    """
    print(f"Reading VTK file: {filename}")
    data = _read_vtk_file(filename)
    if data is None:
        return None
    if data.u is None or data.v is None:
        print("Error: Could not find velocity data in VTK file")
        return None
    return data.to_dict()


def create_vorticity_visualization(data, output_dir="visualization_output"):
    """Create comprehensive vorticity visualization"""
    os.makedirs(output_dir, exist_ok=True)

    x, y = data["x"], data["y"]
    u, v = data["u"], data["v"]
    dx, dy = data["dx"], data["dy"]

    # Calculate vorticity and Q-criterion using fields module
    omega = vorticity(u, v, dx, dy)
    Q = q_criterion(u, v, dx, dy)
    vortex_cores = detect_vortex_cores(omega, Q)

    # Create meshgrid for plotting
    X, Y = np.meshgrid(x, y)

    # Create figure with subplots
    plt.figure(figsize=(16, 12))

    # 1. Vorticity contours
    ax1 = plt.subplot(2, 3, 1)
    vort_levels = np.linspace(-np.max(np.abs(omega)), np.max(np.abs(omega)), 20)
    cs1 = ax1.contourf(X, Y, omega, levels=vort_levels, cmap="RdBu_r", extend="both")
    ax1.contour(
        X,
        Y,
        omega,
        levels=vort_levels[::4],
        colors="black",
        linewidths=0.5,
        alpha=0.3,
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
    # Add streamlines
    ax4.streamplot(X, Y, u, v, density=1.5, color="black", linewidth=0.8, arrowsize=1.2)
    plt.colorbar(cs4, ax=ax4, label="Vorticity (1/s)")
    ax4.set_title("Vorticity with Streamlines")
    ax4.set_xlabel("x")
    ax4.set_ylabel("y")
    ax4.set_aspect("equal")

    # 5. Circulation analysis
    ax5 = plt.subplot(2, 3, 5)
    cs5 = ax5.contourf(X, Y, omega, levels=vort_levels, cmap="RdBu_r", alpha=0.6)

    # Calculate circulation at several radii around domain center
    center_x, center_y = x[len(x) // 2], y[len(y) // 2]
    radii = np.linspace(0.1, min(x.max() - x.min(), y.max() - y.min()) / 3, 5)

    circulations = []
    for radius in radii:
        circ = circulation(u, v, x, y, (center_x, center_y), radius)
        circulations.append(circ)

        # Draw circulation paths
        circle = plt.Circle(
            (center_x, center_y),
            radius,
            fill=False,
            color="white",
            linewidth=2,
            linestyle="--",
        )
        ax5.add_patch(circle)

        # Annotate circulation value
        ax5.text(
            center_x + radius * 0.7,
            center_y + radius * 0.7,
            f"Γ={circ:.3f}",
            fontsize=8,
            color="white",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7),
        )

    plt.colorbar(cs5, ax=ax5, label="Vorticity (1/s)")
    ax5.set_title("Circulation Analysis")
    ax5.set_xlabel("x")
    ax5.set_ylabel("y")
    ax5.set_aspect("equal")

    # 6. Statistics and analysis
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis("off")

    # Calculate statistics
    max_vorticity = np.max(np.abs(omega))
    mean_vorticity = np.mean(omega)
    std_vorticity = np.std(omega)
    max_Q = np.max(Q)
    vortex_area = np.sum(vortex_cores) * dx * dy

    stats_text = f"""
    Vorticity Statistics:
    Max |ω|: {max_vorticity:.4f} 1/s
    Mean ω: {mean_vorticity:.4f} 1/s
    Std ω: {std_vorticity:.4f} 1/s

    Q-Criterion:
    Max Q: {max_Q:.4f}

    Vortex Detection:
    Core area: {vortex_area:.4f} m²

    Circulation Values:
    """

    for r, circ in zip(radii, circulations):
        stats_text += f"  r={r:.3f}: Γ={circ:.4f}\n"

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

    # Save figure
    output_file = os.path.join(output_dir, "vorticity_analysis.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Vorticity analysis saved to: {output_file}")

    plt.show()

    return omega, Q, vortex_cores


def main():
    # Ensure output directories exist
    ensure_dirs()

    parser = argparse.ArgumentParser(
        description="Vorticity and circulation analysis for CFD data"
    )
    parser.add_argument("input_file", nargs="?", help="VTK file to analyze")
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output directory for visualizations (default: centralized PLOTS_DIR)",
    )
    parser.add_argument(
        "--latest",
        "-l",
        action="store_true",
        help="Use latest VTK file in data directory",
    )

    args = parser.parse_args()

    # Use centralized output dir if not specified
    output_dir = args.output if args.output else str(PLOTS_DIR)

    # Determine input file
    if args.latest:
        vtk_files = find_vtk_files()
        if not vtk_files:
            print(f"No VTK files found in {DATA_DIR}")
            print(
                "Set CFD_VIZ_DATA_DIR environment variable to specify a different location."
            )
            return
        input_file = str(max(vtk_files, key=lambda f: f.stat().st_ctime))
        print(f"Using latest file: {input_file}")
    elif args.input_file:
        input_file = args.input_file
    else:
        # Try to find a VTK file
        vtk_files = find_vtk_files()
        if vtk_files:
            input_file = str(vtk_files[0])
            print(f"Using file: {input_file}")
        else:
            print("No VTK file specified. Use --help for usage information.")
            print("Set CFD_VIZ_DATA_DIR environment variable to specify data location.")
            return

    # Read and analyze data
    data = read_vtk_file(input_file)
    if data is None:
        return

    # Create visualization
    create_vorticity_visualization(data, output_dir)

    print("Vorticity analysis complete!")


if __name__ == "__main__":
    main()
