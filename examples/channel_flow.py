#!/usr/bin/env python3
"""Channel Flow (Poiseuille Flow) Example.

This example simulates pressure-driven flow through a rectangular channel.
At steady state, the velocity profile becomes parabolic (Poiseuille flow).

Physical Setup:
- Rectangular channel (longer in x-direction)
- No-slip walls at top and bottom
- Flow driven by pressure gradient
- Develops parabolic velocity profile: u(y) = u_max * (1 - (2y/H)^2)

Usage:
    python examples/channel_flow.py

Requirements:
    - cfd_python package (pip install -e ../cfd-python)
    - cfd_viz package
"""

import sys

import matplotlib.pyplot as plt
import numpy as np

try:
    import cfd_python
except ImportError:
    print("Error: cfd_python package not installed.")
    print("Install with: pip install -e ../cfd-python")
    sys.exit(1)

from cfd_viz.analysis import extract_line_profile
from cfd_viz.common import read_vtk_file
from cfd_viz.fields import magnitude
from cfd_viz.plotting import (
    plot_streamlines,
    plot_velocity_field,
)


def run_channel_simulation(
    nx: int = 200,
    ny: int = 50,
    channel_length: float = 4.0,
    channel_height: float = 1.0,
    steps: int = 1000,
    output_file: str = "channel_flow.vtk",
) -> str:
    """Run channel flow simulation.

    Args:
        nx: Grid points in x (flow direction).
        ny: Grid points in y (cross-channel).
        channel_length: Length of channel.
        channel_height: Height of channel.
        steps: Number of simulation steps.
        output_file: Output VTK file path.

    Returns:
        Path to output VTK file.
    """
    print("Running channel flow simulation...")
    print(f"  Channel: {channel_length} x {channel_height}")
    print(f"  Grid: {nx} x {ny}")
    print(f"  Steps: {steps}")

    result = cfd_python.run_simulation_with_params(
        nx=nx,
        ny=ny,
        xmin=0.0,
        xmax=channel_length,
        ymin=0.0,
        ymax=channel_height,
        steps=steps,
        output_file=output_file,
    )

    print("Simulation complete!")
    if isinstance(result, dict) and "stats" in result:
        stats = result["stats"]
        print(f"  Max velocity: {stats.get('max_velocity', 'N/A')}")

    return output_file


def visualize_channel_flow(vtk_file: str):
    """Visualize channel flow results with velocity profiles.

    Args:
        vtk_file: Path to VTK file.
    """
    print(f"\nLoading results from: {vtk_file}")

    data = read_vtk_file(vtk_file)
    if data is None:
        print(f"Error: Could not read {vtk_file}")
        return

    X, Y = data.X, data.Y
    x, y = data.x, data.y
    u, v = data.u, data.v

    vel_mag = magnitude(u, v)

    print(f"Domain: [{x.min():.2f}, {x.max():.2f}] x [{y.min():.2f}, {y.max():.2f}]")
    print(f"Grid: {data.nx} x {data.ny}")
    print(f"Max velocity: {np.nanmax(vel_mag):.4f}")

    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Velocity magnitude field
    ax1 = axes[0, 0]
    plot_velocity_field(X, Y, u, v, ax=ax1, title="Velocity Magnitude")

    # 2. Streamlines
    ax2 = axes[0, 1]
    plot_streamlines(X, Y, u, v, ax=ax2, title="Streamlines", density=2)

    # 3. U-velocity contours
    ax3 = axes[0, 2]
    contour = ax3.contourf(X, Y, u, levels=20, cmap="viridis")
    plt.colorbar(contour, ax=ax3, label="u-velocity")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_title("Streamwise Velocity (u)")
    ax3.set_aspect("equal")

    # 4. Velocity profiles at different x-locations
    ax4 = axes[1, 0]
    x_locations = [0.25, 0.5, 0.75, 0.9]  # Fraction of channel length
    colors = plt.cm.viridis(np.linspace(0, 1, len(x_locations)))

    for i, x_frac in enumerate(x_locations):
        x_pos = x_frac * x.max()
        # Extract vertical profile
        profile = extract_line_profile(
            u,
            v,
            x,
            y,
            start_point=(x_pos, y.min()),
            end_point=(x_pos, y.max()),
            num_points=50,
        )
        ax4.plot(
            profile.u,
            profile.y_coords,
            color=colors[i],
            linewidth=2,
            label=f"x = {x_frac:.0%}L",
        )

    ax4.set_xlabel("u-velocity")
    ax4.set_ylabel("y")
    ax4.set_title("Velocity Profiles Along Channel")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Centerline velocity
    ax5 = axes[1, 1]
    y_center = y[len(y) // 2]
    centerline_profile = extract_line_profile(
        u,
        v,
        x,
        y,
        start_point=(x.min(), y_center),
        end_point=(x.max(), y_center),
        num_points=100,
    )
    ax5.plot(centerline_profile.x_coords, centerline_profile.u, "b-", linewidth=2)
    ax5.set_xlabel("x")
    ax5.set_ylabel("u-velocity")
    ax5.set_title("Centerline Velocity Development")
    ax5.grid(True, alpha=0.3)

    # 6. Compare with analytical Poiseuille profile
    ax6 = axes[1, 2]

    # Extract profile at outlet (x = 0.9 * L)
    x_outlet = 0.9 * x.max()
    outlet_profile = extract_line_profile(
        u,
        v,
        x,
        y,
        start_point=(x_outlet, y.min()),
        end_point=(x_outlet, y.max()),
        num_points=50,
    )

    # Analytical Poiseuille profile: u(y) = u_max * 4 * y/H * (1 - y/H)
    H = y.max() - y.min()
    y_norm = (outlet_profile.y_coords - y.min()) / H
    u_max = np.max(outlet_profile.u)
    u_analytical = u_max * 4 * y_norm * (1 - y_norm)

    ax6.plot(
        outlet_profile.u, outlet_profile.y_coords, "b-", linewidth=2, label="Simulation"
    )
    ax6.plot(
        u_analytical,
        outlet_profile.y_coords,
        "r--",
        linewidth=2,
        label="Analytical (Poiseuille)",
    )
    ax6.set_xlabel("u-velocity")
    ax6.set_ylabel("y")
    ax6.set_title("Outlet Profile vs Analytical")
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.suptitle("Channel Flow (Poiseuille Flow) Simulation", fontsize=14)
    plt.tight_layout()

    output_file = "channel_flow_results.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_file}")

    plt.show()


def main():
    """Main function."""
    print("Channel Flow (Poiseuille Flow) Example")
    print("=" * 40)
    print()

    vtk_file = run_channel_simulation(
        nx=200,
        ny=50,
        channel_length=4.0,
        channel_height=1.0,
        steps=1000,
        output_file="channel_flow.vtk",
    )

    visualize_channel_flow(vtk_file)

    print("\nDone!")


if __name__ == "__main__":
    main()
