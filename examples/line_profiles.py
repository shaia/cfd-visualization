#!/usr/bin/env python3
"""Line Profile Extraction Example.

This example demonstrates line profile extraction and analysis:

1. Extracting velocity profiles along lines
2. Vertical and horizontal profile extraction
3. Centerline profiles
4. Computing mass flow rate through sections
5. Profile statistics

Usage:
    python examples/line_profiles.py

Requirements:
    - cfd_python package (pip install -e ../cfd-python)
    - cfd_viz package
"""

import sys

import matplotlib.pyplot as plt
import numpy as np

# Check for cfd_python
try:
    import cfd_python
except ImportError:
    print("Error: cfd_python package not installed.")
    print("Install with: pip install -e ../cfd-python")
    sys.exit(1)

from cfd_viz.analysis import (
    compute_centerline_profiles,
    compute_mass_flow_rate,
    compute_profile_statistics,
    extract_horizontal_profile,
    extract_line_profile,
    extract_multiple_profiles,
    extract_vertical_profile,
)
from cfd_viz.common import read_vtk_file
from cfd_viz.fields import magnitude
from cfd_viz.plotting import plot_line_profile, plot_multiple_profiles


def run_simulation():
    """Run a simulation to generate flow data."""
    print("Running simulation...")

    result = cfd_python.run_simulation_with_params(
        nx=80,
        ny=80,
        xmin=0.0,
        xmax=1.0,
        ymin=0.0,
        ymax=1.0,
        steps=400,
        output_file="line_profile_flow.vtk",
    )

    print("Simulation complete!")
    return "line_profile_flow.vtk"


def extract_single_profile(data):
    """Extract a single line profile."""
    print("\n1. Single Line Profile Extraction")
    print("-" * 40)

    # Extract diagonal profile from corner to corner
    profile = extract_line_profile(
        u=data.u,
        v=data.v,
        x=data.x,
        y=data.y,
        start=(0.0, 0.0),
        end=(1.0, 1.0),
        num_points=100,
        p=data.get("p"),
    )

    print(f"Profile from {profile.start} to {profile.end}")
    print(f"  Number of points: {len(profile.distance)}")
    print(f"  Profile length: {profile.distance[-1]:.4f}")
    print(f"  Max velocity: {profile.velocity_mag.max():.4f}")
    print(f"  Mean velocity: {profile.velocity_mag.mean():.4f}")

    return profile


def extract_vertical_profiles(data):
    """Extract vertical profiles at multiple x-locations."""
    print("\n2. Vertical Profile Extraction")
    print("-" * 40)

    x_locations = [0.25, 0.5, 0.75]
    profiles = []

    for x_loc in x_locations:
        profile = extract_vertical_profile(
            u=data.u,
            v=data.v,
            x=data.x,
            y=data.y,
            x_location=x_loc,
            p=data.get("p"),
        )
        profiles.append(profile)
        print(f"  x = {x_loc:.2f}: max |V| = {profile.velocity_mag.max():.4f}")

    return profiles, x_locations


def extract_horizontal_profiles(data):
    """Extract horizontal profiles at multiple y-locations."""
    print("\n3. Horizontal Profile Extraction")
    print("-" * 40)

    y_locations = [0.25, 0.5, 0.75]
    profiles = []

    for y_loc in y_locations:
        profile = extract_horizontal_profile(
            u=data.u,
            v=data.v,
            x=data.x,
            y=data.y,
            y_location=y_loc,
            p=data.get("p"),
        )
        profiles.append(profile)
        print(f"  y = {y_loc:.2f}: max |V| = {profile.velocity_mag.max():.4f}")

    return profiles, y_locations


def extract_centerline(data):
    """Extract centerline profiles."""
    print("\n4. Centerline Profile Extraction")
    print("-" * 40)

    # Horizontal centerline (y = 0.5)
    h_centerline = extract_horizontal_profile(
        u=data.u,
        v=data.v,
        x=data.x,
        y=data.y,
        y_location=0.5,
        p=data.get("p"),
    )

    # Vertical centerline (x = 0.5)
    v_centerline = extract_vertical_profile(
        u=data.u,
        v=data.v,
        x=data.x,
        y=data.y,
        x_location=0.5,
        p=data.get("p"),
    )

    print("Horizontal centerline (y=0.5):")
    print(f"  Max u-velocity: {h_centerline.u.max():.4f}")
    print(f"  Min u-velocity: {h_centerline.u.min():.4f}")

    print("Vertical centerline (x=0.5):")
    print(f"  Max v-velocity: {v_centerline.v.max():.4f}")
    print(f"  Min v-velocity: {v_centerline.v.min():.4f}")

    return h_centerline, v_centerline


def compute_mass_flow_demo(data):
    """Compute mass flow rate through cross-sections."""
    print("\n5. Mass Flow Rate Computation")
    print("-" * 40)

    # Compute mass flow through vertical sections at different x
    x_positions = [0.2, 0.4, 0.6, 0.8]
    rho = 1.0  # Density

    print("Mass flow through vertical sections:")
    for x_pos in x_positions:
        mass_flow = compute_mass_flow_rate(
            u=data.u,
            v=data.v,
            x=data.x,
            y=data.y,
            position=x_pos,
            direction="vertical",
            rho=rho,
        )
        print(f"  x = {x_pos:.1f}: mdot = {mass_flow:.6f} kg/s")


def compute_statistics_demo(data):
    """Compute profile statistics."""
    print("\n6. Profile Statistics")
    print("-" * 40)

    profile = extract_vertical_profile(
        u=data.u,
        v=data.v,
        x=data.x,
        y=data.y,
        x_location=0.5,
        p=data.get("p"),
    )

    stats = compute_profile_statistics(profile)

    print("Statistics for vertical profile at x=0.5:")
    print(f"  U-velocity:")
    print(f"    Mean: {stats['u_mean']:.4f}")
    print(f"    Std: {stats['u_std']:.4f}")
    print(f"    Max: {stats['u_max']:.4f}")
    print(f"    Min: {stats['u_min']:.4f}")
    print(f"  V-velocity:")
    print(f"    Mean: {stats['v_mean']:.4f}")
    print(f"    Max: {stats['v_max']:.4f}")
    print(f"  Velocity magnitude:")
    print(f"    Mean: {stats['velocity_mag_mean']:.4f}")
    print(f"    Max: {stats['velocity_mag_max']:.4f}")

    return stats


def create_visualizations(data, diag_profile, v_profiles, x_locs, h_centerline, v_centerline):
    """Create visualization plots."""
    print("\n7. Creating Visualizations")
    print("-" * 40)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot 1: Flow field with profile lines
    ax1 = axes[0, 0]
    vel_mag = magnitude(data.u, data.v)
    c1 = ax1.contourf(data.X, data.Y, vel_mag, levels=30, cmap="viridis", alpha=0.8)
    plt.colorbar(c1, ax=ax1, label="|V|")
    # Draw profile lines
    ax1.plot([0, 1], [0, 1], "r-", linewidth=2, label="Diagonal")
    for x_loc in x_locs:
        ax1.axvline(x=x_loc, color="white", linestyle="--", linewidth=1)
    ax1.axhline(y=0.5, color="cyan", linestyle="-", linewidth=2, label="H-centerline")
    ax1.axvline(x=0.5, color="magenta", linestyle="-", linewidth=2, label="V-centerline")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("Flow Field with Profile Lines")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.set_aspect("equal")

    # Plot 2: Diagonal profile
    ax2 = axes[0, 1]
    ax2.plot(diag_profile.distance, diag_profile.u, "b-", label="u")
    ax2.plot(diag_profile.distance, diag_profile.v, "r-", label="v")
    ax2.plot(diag_profile.distance, diag_profile.velocity_mag, "k-", linewidth=2, label="|V|")
    ax2.set_xlabel("Distance along diagonal")
    ax2.set_ylabel("Velocity")
    ax2.set_title("Diagonal Profile (0,0) to (1,1)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Vertical profiles (u vs y)
    ax3 = axes[0, 2]
    colors = ["b", "g", "r"]
    for i, (profile, x_loc) in enumerate(zip(v_profiles, x_locs)):
        ax3.plot(profile.u, profile.y_coords, colors[i] + "-", linewidth=2,
                label=f"x = {x_loc:.2f}")
    ax3.set_xlabel("u-velocity")
    ax3.set_ylabel("y")
    ax3.set_title("Vertical Profiles (u vs y)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Horizontal centerline
    ax4 = axes[1, 0]
    ax4.plot(h_centerline.x_coords, h_centerline.u, "b-", linewidth=2, label="u")
    ax4.plot(h_centerline.x_coords, h_centerline.v, "r-", linewidth=2, label="v")
    ax4.set_xlabel("x")
    ax4.set_ylabel("Velocity")
    ax4.set_title("Horizontal Centerline (y = 0.5)")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Vertical centerline
    ax5 = axes[1, 1]
    ax5.plot(v_centerline.u, v_centerline.y_coords, "b-", linewidth=2, label="u")
    ax5.plot(v_centerline.v, v_centerline.y_coords, "r-", linewidth=2, label="v")
    ax5.set_xlabel("Velocity")
    ax5.set_ylabel("y")
    ax5.set_title("Vertical Centerline (x = 0.5)")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Plot 6: Velocity magnitude profiles comparison
    ax6 = axes[1, 2]
    for i, (profile, x_loc) in enumerate(zip(v_profiles, x_locs)):
        ax6.plot(profile.velocity_mag, profile.y_coords, colors[i] + "-",
                linewidth=2, label=f"x = {x_loc:.2f}")
    ax6.set_xlabel("Velocity magnitude |V|")
    ax6.set_ylabel("y")
    ax6.set_title("Velocity Magnitude Profiles")
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.suptitle("Line Profile Analysis", fontsize=14)
    plt.tight_layout()

    output_file = "line_profiles.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_file}")

    plt.show()


def main():
    """Run line profile extraction example."""
    print("Line Profile Extraction Example")
    print("=" * 40)
    print()

    # Run simulation
    vtk_file = run_simulation()

    # Load data
    data = read_vtk_file(vtk_file)
    if data is None:
        print(f"Error: Could not read {vtk_file}")
        return

    print(f"\nLoaded data: {data.nx}x{data.ny} grid")

    # Run extractions and analyses
    diag_profile = extract_single_profile(data)
    v_profiles, x_locs = extract_vertical_profiles(data)
    h_profiles, y_locs = extract_horizontal_profiles(data)
    h_centerline, v_centerline = extract_centerline(data)
    compute_mass_flow_demo(data)
    compute_statistics_demo(data)

    # Create visualizations
    create_visualizations(
        data, diag_profile, v_profiles, x_locs,
        h_centerline, v_centerline
    )

    print("\n" + "=" * 40)
    print("Line profile extraction complete!")
    print()
    print("Key functions demonstrated:")
    print("  - extract_line_profile(): Extract along arbitrary line")
    print("  - extract_vertical_profile(): Extract at constant x")
    print("  - extract_horizontal_profile(): Extract at constant y")
    print("  - compute_mass_flow_rate(): Mass flow through section")
    print("  - compute_profile_statistics(): Profile statistics")


if __name__ == "__main__":
    main()
