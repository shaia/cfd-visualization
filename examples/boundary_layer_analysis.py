#!/usr/bin/env python3
"""Boundary Layer Analysis Example.

This example demonstrates boundary layer analysis capabilities:

1. Extracting velocity profiles at different streamwise locations
2. Computing boundary layer parameters (delta_99, delta*, theta, H)
3. Analyzing boundary layer development along a surface
4. Computing wall shear stress distribution
5. Comparing to Blasius flat plate solution

Usage:
    python examples/boundary_layer_analysis.py

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
    analyze_boundary_layer,
    analyze_boundary_layer_development,
    blasius_solution,
    compute_wall_shear_distribution,
)
from cfd_viz.common import read_vtk_file
from cfd_viz.plotting import plot_boundary_layer_profiles


def run_simulation():
    """Run a simulation to generate flow data."""
    print("Running simulation for boundary layer analysis...")

    # Run a lid-driven cavity which develops boundary layers on walls
    result = cfd_python.run_simulation_with_params(
        nx=80,
        ny=80,
        xmin=0.0,
        xmax=1.0,
        ymin=0.0,
        ymax=1.0,
        steps=500,
        output_file="boundary_layer_flow.vtk",
    )

    print("Simulation complete!")
    return "boundary_layer_flow.vtk"


def analyze_single_profile(data):
    """Analyze boundary layer at a single location."""
    print("\n1. Single Profile Analysis")
    print("-" * 40)

    # Analyze boundary layer on the bottom wall (y=0) at x=0.5
    bl = analyze_boundary_layer(
        u=data.u,
        v=data.v,
        x=data.x,
        y=data.y,
        wall_y=data.y.min(),
        x_location=0.5,
        mu=0.01,  # Dynamic viscosity
        rho=1.0,
    )

    print(f"Analysis at x = {bl.x_location:.3f}")
    print(f"  Edge velocity (u_edge): {bl.u_edge:.4f}")
    print(f"  BL thickness (delta_99): {bl.delta_99:.4f}")
    print(f"  Displacement thickness (delta*): {bl.delta_star:.6f}")
    print(f"  Momentum thickness (theta): {bl.theta:.6f}")
    print(f"  Shape factor (H): {bl.H:.3f}")

    if bl.cf is not None:
        print(f"  Skin friction coeff (Cf): {bl.cf:.6f}")
    if bl.Re_theta is not None:
        print(f"  Re_theta: {bl.Re_theta:.1f}")

    return bl


def analyze_development(data):
    """Analyze boundary layer development along the surface."""
    print("\n2. Boundary Layer Development")
    print("-" * 40)

    # Analyze at multiple streamwise locations
    x_locations = [0.2, 0.4, 0.6, 0.8]

    bl_dev = analyze_boundary_layer_development(
        u=data.u,
        v=data.v,
        x=data.x,
        y=data.y,
        wall_y=data.y.min(),
        x_locations=x_locations,
        mu=0.01,
        rho=1.0,
    )

    print(f"{'x':>8s} {'delta_99':>10s} {'delta*':>10s} {'theta':>10s} {'H':>8s}")
    print("-" * 50)
    for profile in bl_dev.profiles:
        print(
            f"{profile.x_location:8.3f} "
            f"{profile.delta_99:10.4f} "
            f"{profile.delta_star:10.6f} "
            f"{profile.theta:10.6f} "
            f"{profile.H:8.3f}"
        )

    return bl_dev


def analyze_wall_shear(data):
    """Analyze wall shear stress distribution."""
    print("\n3. Wall Shear Stress Distribution")
    print("-" * 40)

    shear = compute_wall_shear_distribution(
        u=data.u,
        x=data.x,
        y=data.y,
        wall_y=data.y.min(),
        mu=0.01,
        rho=1.0,
    )

    print(f"Max wall shear stress: {np.max(shear.tau_w):.6f}")
    print(f"Min wall shear stress: {np.min(shear.tau_w):.6f}")
    print(f"Mean skin friction Cf: {np.mean(shear.cf):.6f}")

    return shear


def compare_to_blasius():
    """Compare velocity profile to Blasius solution."""
    print("\n4. Comparison with Blasius Solution")
    print("-" * 40)

    # Generate Blasius solution
    eta = np.linspace(0, 8, 100)
    f_prime, _ = blasius_solution(eta, u_inf=1.0)

    print("Blasius solution computed for eta = 0 to 8")
    print(f"  At eta=0: u/U_inf = {f_prime[0]:.4f}")
    print(f"  At eta=5: u/U_inf = {f_prime[eta >= 5][0]:.4f} (should be ~0.99)")

    return eta, f_prime


def create_visualizations(data, bl, bl_dev, shear, eta, f_prime):
    """Create visualization plots."""
    print("\n5. Creating Visualizations")
    print("-" * 40)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Single boundary layer profile
    ax1 = axes[0, 0]
    y_norm = bl.y_normalized
    u_norm = bl.u_normalized

    # Only plot within the boundary layer region
    mask = y_norm <= 2.0
    ax1.plot(u_norm[mask], y_norm[mask], "b-", linewidth=2, label="Simulation")
    ax1.plot(f_prime, eta / 5, "r--", linewidth=2, label="Blasius (scaled)")
    ax1.set_xlabel("u / u_edge")
    ax1.set_ylabel("y / delta_99")
    ax1.set_title(f"Velocity Profile at x = {bl.x_location:.2f}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1.2)
    ax1.set_ylim(0, 2.0)

    # Plot 2: BL thickness development
    ax2 = axes[0, 1]
    ax2.plot(bl_dev.x_locations, bl_dev.delta_99, "b-o", label="delta_99")
    ax2.plot(bl_dev.x_locations, bl_dev.delta_star * 10, "r-s", label="delta* (x10)")
    ax2.plot(bl_dev.x_locations, bl_dev.theta * 10, "g-^", label="theta (x10)")
    ax2.set_xlabel("x")
    ax2.set_ylabel("Thickness")
    ax2.set_title("Boundary Layer Development")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Shape factor
    ax3 = axes[1, 0]
    ax3.plot(bl_dev.x_locations, bl_dev.H, "b-o", linewidth=2)
    ax3.axhline(y=2.59, color="r", linestyle="--", label="Blasius H=2.59")
    ax3.set_xlabel("x")
    ax3.set_ylabel("Shape Factor H")
    ax3.set_title("Shape Factor Development")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Wall shear stress
    ax4 = axes[1, 1]
    ax4.plot(shear.x, shear.cf, "b-", linewidth=2)
    ax4.set_xlabel("x")
    ax4.set_ylabel("Skin Friction Coefficient Cf")
    ax4.set_title("Wall Shear Stress Distribution")
    ax4.grid(True, alpha=0.3)

    plt.suptitle("Boundary Layer Analysis Results", fontsize=14)
    plt.tight_layout()

    output_file = "boundary_layer_analysis.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_file}")

    plt.show()


def main():
    """Run boundary layer analysis example."""
    print("Boundary Layer Analysis Example")
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
    print(f"Domain: [{data.x.min():.2f}, {data.x.max():.2f}] x "
          f"[{data.y.min():.2f}, {data.y.max():.2f}]")

    # Run analyses
    bl = analyze_single_profile(data)
    bl_dev = analyze_development(data)
    shear = analyze_wall_shear(data)
    eta, f_prime = compare_to_blasius()

    # Create visualizations
    create_visualizations(data, bl, bl_dev, shear, eta, f_prime)

    print("\n" + "=" * 40)
    print("Boundary layer analysis complete!")
    print()
    print("Key functions demonstrated:")
    print("  - analyze_boundary_layer(): Full BL analysis at one location")
    print("  - analyze_boundary_layer_development(): Track BL along surface")
    print("  - compute_wall_shear_distribution(): Cf distribution")
    print("  - blasius_solution(): Analytical flat plate solution")


if __name__ == "__main__":
    main()
