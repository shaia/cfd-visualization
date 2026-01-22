#!/usr/bin/env python3
"""Vortex Identification Example.

This example demonstrates vortex identification techniques:

1. Computing vorticity field
2. Using Q-criterion for vortex detection
3. Using Lambda-2 criterion
4. Computing enstrophy
5. Detecting vortex core locations
6. Computing circulation around vortex

Usage:
    python examples/vortex_identification.py

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

from cfd_viz.common import read_vtk_file
from cfd_viz.fields import (
    circulation,
    detect_vortex_cores,
    enstrophy,
    lambda2_criterion,
    magnitude,
    q_criterion,
    vorticity,
)


def run_simulation():
    """Run a simulation to generate vortical flow."""
    print("Running simulation to generate vortical flow...")

    # Lid-driven cavity develops corner vortices
    result = cfd_python.run_simulation_with_params(
        nx=80,
        ny=80,
        xmin=0.0,
        xmax=1.0,
        ymin=0.0,
        ymax=1.0,
        steps=500,
        output_file="vortex_flow.vtk",
    )

    print("Simulation complete!")
    return "vortex_flow.vtk"


def compute_vorticity_analysis(data):
    """Compute vorticity and related quantities."""
    print("\n1. Vorticity Analysis")
    print("-" * 40)

    omega = vorticity(data.u, data.v, data.dx, data.dy)

    print("Vorticity field computed")
    print(f"  Min vorticity: {np.min(omega):.4f}")
    print(f"  Max vorticity: {np.max(omega):.4f}")
    print(f"  Mean |omega|: {np.mean(np.abs(omega)):.4f}")

    return omega


def compute_q_criterion_analysis(data):
    """Compute Q-criterion for vortex identification."""
    print("\n2. Q-Criterion Analysis")
    print("-" * 40)

    Q = q_criterion(data.u, data.v, data.dx, data.dy)

    print("Q-criterion computed")
    print("  Q > 0: Rotation dominates (vortex cores)")
    print("  Q < 0: Strain dominates")
    print(f"  Max Q: {np.max(Q):.4f}")
    print(f"  Min Q: {np.min(Q):.4f}")
    print(f"  Fraction Q > 0: {np.mean(Q > 0) * 100:.1f}%")

    return Q


def compute_lambda2_analysis(data):
    """Compute Lambda-2 criterion for vortex identification."""
    print("\n3. Lambda-2 Criterion Analysis")
    print("-" * 40)

    lam2 = lambda2_criterion(data.u, data.v, data.dx, data.dy)

    print("Lambda-2 criterion computed")
    print("  Lambda2 < 0: Vortex cores")
    print(f"  Max Lambda2: {np.max(lam2):.4f}")
    print(f"  Min Lambda2: {np.min(lam2):.4f}")
    print(f"  Fraction Lambda2 < 0: {np.mean(lam2 < 0) * 100:.1f}%")

    return lam2


def compute_enstrophy_analysis(data):
    """Compute enstrophy (squared vorticity)."""
    print("\n4. Enstrophy Analysis")
    print("-" * 40)

    enst = enstrophy(data.u, data.v, data.dx, data.dy)

    # Total enstrophy (integrated)
    total_enst = np.sum(enst) * data.dx * data.dy

    print("Enstrophy computed: E = 0.5 * omega^2")
    print(f"  Max enstrophy: {np.max(enst):.4f}")
    print(f"  Total enstrophy: {total_enst:.4f}")

    return enst


def detect_vortex_cores_analysis(omega, Q):
    """Detect vortex core locations."""
    print("\n5. Vortex Core Detection")
    print("-" * 40)

    cores = detect_vortex_cores(
        omega,
        Q,
        omega_threshold_factor=0.2,
        Q_threshold_factor=0.1,
    )

    num_vortex_points = np.sum(cores)
    total_points = cores.size
    fraction = num_vortex_points / total_points * 100

    print("Vortex cores detected using combined omega and Q criteria")
    print(f"  Number of vortex points: {num_vortex_points}")
    print(f"  Fraction of domain: {fraction:.2f}%")

    return cores


def compute_circulation_analysis(data):
    """Compute circulation around potential vortex."""
    print("\n6. Circulation Analysis")
    print("-" * 40)

    # Compute circulation around center of domain (main vortex region)
    center = (0.5, 0.75)  # Near the top center where main vortex forms
    radii = [0.1, 0.15, 0.2]

    print(f"Computing circulation around ({center[0]}, {center[1]})")
    for radius in radii:
        gamma = circulation(
            data.u,
            data.v,
            data.x,
            data.y,
            center=center,
            radius=radius,
            num_points=100,
        )
        print(f"  Radius = {radius:.2f}: Gamma = {gamma:.4f}")

    return center


def create_visualizations(data, omega, Q, lam2, enst, cores, vortex_center):
    """Create visualization plots."""
    print("\n7. Creating Visualizations")
    print("-" * 40)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot 1: Vorticity field
    ax1 = axes[0, 0]
    levels1 = np.linspace(-np.max(np.abs(omega)), np.max(np.abs(omega)), 30)
    c1 = ax1.contourf(data.X, data.Y, omega, levels=levels1, cmap="RdBu_r")
    plt.colorbar(c1, ax=ax1, label="omega")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("Vorticity Field")
    ax1.set_aspect("equal")

    # Plot 2: Q-criterion
    ax2 = axes[0, 1]
    Q_clipped = np.clip(Q, np.percentile(Q, 5), np.percentile(Q, 95))
    c2 = ax2.contourf(data.X, data.Y, Q_clipped, levels=30, cmap="RdBu_r")
    plt.colorbar(c2, ax=ax2, label="Q")
    ax2.contour(data.X, data.Y, Q, levels=[0], colors="black", linewidths=1)
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_title("Q-Criterion (black: Q=0)")
    ax2.set_aspect("equal")

    # Plot 3: Lambda-2 criterion
    ax3 = axes[0, 2]
    lam2_clipped = np.clip(lam2, np.percentile(lam2, 5), np.percentile(lam2, 95))
    c3 = ax3.contourf(data.X, data.Y, lam2_clipped, levels=30, cmap="RdBu_r")
    plt.colorbar(c3, ax=ax3, label="Lambda2")
    ax3.contour(data.X, data.Y, lam2, levels=[0], colors="black", linewidths=1)
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_title("Lambda-2 Criterion (black: L2=0)")
    ax3.set_aspect("equal")

    # Plot 4: Enstrophy
    ax4 = axes[1, 0]
    c4 = ax4.contourf(data.X, data.Y, enst, levels=30, cmap="hot")
    plt.colorbar(c4, ax=ax4, label="Enstrophy")
    ax4.set_xlabel("x")
    ax4.set_ylabel("y")
    ax4.set_title("Enstrophy (0.5 * omega^2)")
    ax4.set_aspect("equal")

    # Plot 5: Vortex cores with streamlines
    ax5 = axes[1, 1]
    vel_mag = magnitude(data.u, data.v)
    c5 = ax5.contourf(data.X, data.Y, vel_mag, levels=30, cmap="viridis", alpha=0.7)
    plt.colorbar(c5, ax=ax5, label="|V|")
    ax5.contour(data.X, data.Y, cores.astype(int), levels=[0.5], colors="red", linewidths=2)
    ax5.streamplot(data.X, data.Y, data.u, data.v, density=1.5, color="white", linewidth=0.5)
    ax5.set_xlabel("x")
    ax5.set_ylabel("y")
    ax5.set_title("Velocity + Vortex Cores (red)")
    ax5.set_aspect("equal")

    # Plot 6: Combined visualization
    ax6 = axes[1, 2]
    # Show Q > 0 regions with vorticity coloring
    Q_positive = np.ma.masked_where(Q <= 0, omega)
    c6 = ax6.contourf(data.X, data.Y, Q_positive, levels=30, cmap="RdBu_r")
    plt.colorbar(c6, ax=ax6, label="omega (Q>0 only)")
    ax6.streamplot(data.X, data.Y, data.u, data.v, density=1.2, color="black", linewidth=0.5)
    # Mark circulation center
    circle = plt.Circle(vortex_center, 0.15, fill=False, color="lime", linewidth=2)
    ax6.add_patch(circle)
    ax6.set_xlabel("x")
    ax6.set_ylabel("y")
    ax6.set_title("Vortex Regions (Q>0) + Circulation Path")
    ax6.set_aspect("equal")

    plt.suptitle("Vortex Identification Analysis", fontsize=14)
    plt.tight_layout()

    output_file = "vortex_identification.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_file}")

    plt.show()


def main():
    """Run vortex identification example."""
    print("Vortex Identification Example")
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

    # Run analyses
    omega = compute_vorticity_analysis(data)
    Q = compute_q_criterion_analysis(data)
    lam2 = compute_lambda2_analysis(data)
    enst = compute_enstrophy_analysis(data)
    cores = detect_vortex_cores_analysis(omega, Q)
    vortex_center = compute_circulation_analysis(data)

    # Create visualizations
    create_visualizations(data, omega, Q, lam2, enst, cores, vortex_center)

    print("\n" + "=" * 40)
    print("Vortex identification complete!")
    print()
    print("Key functions demonstrated:")
    print("  - vorticity(): z-component of curl of velocity")
    print("  - q_criterion(): Q > 0 indicates rotation dominates")
    print("  - lambda2_criterion(): L2 < 0 indicates vortex cores")
    print("  - enstrophy(): 0.5 * omega^2 (vorticity energy)")
    print("  - detect_vortex_cores(): Combined omega + Q detection")
    print("  - circulation(): Line integral of velocity around path")


if __name__ == "__main__":
    main()
