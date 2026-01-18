#!/usr/bin/env python3
"""Field Comparison Example.

This example demonstrates field comparison and analysis capabilities:

1. Comparing two CFD solutions
2. Computing field differences
3. Computing error norms (L2, L-infinity)
4. Parameter sweep analysis
5. Visualizing differences

Usage:
    python examples/field_comparison.py

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
    compare_fields,
    compute_error_norms,
    compute_field_difference,
)
from cfd_viz.common import read_vtk_file
from cfd_viz.fields import magnitude


def run_case(nx, ny, steps, output_name):
    """Run a simulation case with specified parameters."""
    print(f"  Running {output_name} ({nx}x{ny}, {steps} steps)...")

    result = cfd_python.run_simulation_with_params(
        nx=nx,
        ny=ny,
        xmin=0.0,
        xmax=1.0,
        ymin=0.0,
        ymax=1.0,
        steps=steps,
        output_file=output_name,
    )

    return output_name


def compare_grid_resolutions():
    """Compare solutions at different grid resolutions."""
    print("\n1. Grid Resolution Comparison")
    print("-" * 40)

    # Run coarse and fine grid simulations
    coarse_file = run_case(40, 40, 300, "coarse_grid.vtk")
    fine_file = run_case(80, 80, 300, "fine_grid.vtk")

    # Load data
    coarse = read_vtk_file(coarse_file)
    fine = read_vtk_file(fine_file)

    if coarse is None or fine is None:
        print("Error loading VTK files")
        return None, None, None

    print(f"\nCoarse grid: {coarse.nx}x{coarse.ny}")
    print(f"Fine grid: {fine.nx}x{fine.ny}")

    # Interpolate coarse to fine grid for comparison
    from scipy.interpolate import RegularGridInterpolator

    # Create interpolators for coarse data
    interp_u = RegularGridInterpolator(
        (coarse.y, coarse.x), coarse.u,
        bounds_error=False, fill_value=0
    )
    interp_v = RegularGridInterpolator(
        (coarse.y, coarse.x), coarse.v,
        bounds_error=False, fill_value=0
    )

    # Interpolate to fine grid
    points = np.array([fine.Y.flatten(), fine.X.flatten()]).T
    u_coarse_interp = interp_u(points).reshape(fine.ny, fine.nx)
    v_coarse_interp = interp_v(points).reshape(fine.ny, fine.nx)

    # Compute differences
    u_diff = compute_field_difference(fine.u, u_coarse_interp)
    v_diff = compute_field_difference(fine.v, v_coarse_interp)

    print(f"\nU-velocity difference:")
    print(f"  Max absolute: {u_diff.max_abs_diff:.6f}")
    print(f"  RMS: {u_diff.rms_diff:.6f}")
    print(f"  Mean relative: {u_diff.mean_relative_diff * 100:.2f}%")

    print(f"\nV-velocity difference:")
    print(f"  Max absolute: {v_diff.max_abs_diff:.6f}")
    print(f"  RMS: {v_diff.rms_diff:.6f}")
    print(f"  Mean relative: {v_diff.mean_relative_diff * 100:.2f}%")

    return fine, u_coarse_interp, v_coarse_interp


def compare_iteration_counts():
    """Compare solutions at different iteration counts."""
    print("\n2. Iteration Count Comparison")
    print("-" * 40)

    # Run with different iteration counts
    early_file = run_case(60, 60, 100, "early_solution.vtk")
    converged_file = run_case(60, 60, 500, "converged_solution.vtk")

    # Load data
    early = read_vtk_file(early_file)
    converged = read_vtk_file(converged_file)

    if early is None or converged is None:
        print("Error loading VTK files")
        return None, None

    print(f"\nEarly solution: {100} iterations")
    print(f"Converged solution: {500} iterations")

    # Compute comprehensive comparison
    comparison = compare_fields(
        u1=early.u,
        v1=early.v,
        p1=early.get("p") if early.get("p") is not None else np.zeros_like(early.u),
        u2=converged.u,
        v2=converged.v,
        p2=converged.get("p") if converged.get("p") is not None else np.zeros_like(converged.u),
        dx=early.dx,
        dy=early.dy,
    )

    print(f"\nVelocity comparison:")
    print(f"  Early max velocity: {comparison.stats1['max_velocity']:.4f}")
    print(f"  Converged max velocity: {comparison.stats2['max_velocity']:.4f}")

    print(f"\nVelocity difference:")
    print(f"  Max: {comparison.velocity_diff.max_abs_diff:.6f}")
    print(f"  RMS: {comparison.velocity_diff.rms_diff:.6f}")

    return early, converged


def compute_error_norms_demo():
    """Demonstrate error norm computations."""
    print("\n3. Error Norm Computation")
    print("-" * 40)

    # Run reference (fine grid) and test (coarse grid)
    ref_file = run_case(80, 80, 400, "reference.vtk")
    test_file = run_case(60, 60, 400, "test.vtk")

    ref = read_vtk_file(ref_file)
    test = read_vtk_file(test_file)

    if ref is None or test is None:
        print("Error loading VTK files")
        return None

    # Interpolate test to reference grid
    from scipy.interpolate import RegularGridInterpolator

    interp_u = RegularGridInterpolator(
        (test.y, test.x), test.u,
        bounds_error=False, fill_value=0
    )
    interp_v = RegularGridInterpolator(
        (test.y, test.x), test.v,
        bounds_error=False, fill_value=0
    )

    points = np.array([ref.Y.flatten(), ref.X.flatten()]).T
    u_test_interp = interp_u(points).reshape(ref.ny, ref.nx)
    v_test_interp = interp_v(points).reshape(ref.ny, ref.nx)

    # Compute velocity magnitude for both
    vel_ref = magnitude(ref.u, ref.v)
    vel_test = magnitude(u_test_interp, v_test_interp)

    # Compute error norms
    norms = compute_error_norms(vel_ref, vel_test, ref.dx, ref.dy)

    print("Error norms (velocity magnitude):")
    print(f"  L1 norm: {norms['L1']:.6f}")
    print(f"  L2 norm: {norms['L2']:.6f}")
    print(f"  L-infinity norm: {norms['Linf']:.6f}")
    print(f"  Relative L2: {norms['relative_L2'] * 100:.2f}%")

    return ref, vel_ref, vel_test, norms


def create_comparison_visualizations(fine, u_coarse, v_coarse, early, converged, ref, vel_ref, vel_test):
    """Create visualization of comparisons."""
    print("\n4. Creating Comparison Visualizations")
    print("-" * 40)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot 1: Fine grid velocity magnitude
    ax1 = axes[0, 0]
    vel_fine = magnitude(fine.u, fine.v)
    c1 = ax1.contourf(fine.X, fine.Y, vel_fine, levels=30, cmap="viridis")
    plt.colorbar(c1, ax=ax1, label="|V|")
    ax1.set_title("Fine Grid (80x80)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_aspect("equal")

    # Plot 2: Coarse grid (interpolated) velocity magnitude
    ax2 = axes[0, 1]
    vel_coarse = magnitude(u_coarse, v_coarse)
    c2 = ax2.contourf(fine.X, fine.Y, vel_coarse, levels=30, cmap="viridis")
    plt.colorbar(c2, ax=ax2, label="|V|")
    ax2.set_title("Coarse Grid (40x40, interpolated)")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_aspect("equal")

    # Plot 3: Grid difference
    ax3 = axes[0, 2]
    vel_diff = vel_fine - vel_coarse
    max_diff = max(abs(vel_diff.min()), abs(vel_diff.max()))
    c3 = ax3.contourf(fine.X, fine.Y, vel_diff, levels=30,
                      cmap="RdBu_r", vmin=-max_diff, vmax=max_diff)
    plt.colorbar(c3, ax=ax3, label="Difference")
    ax3.set_title("Fine - Coarse Difference")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_aspect("equal")

    # Plot 4: Early solution
    ax4 = axes[1, 0]
    vel_early = magnitude(early.u, early.v)
    c4 = ax4.contourf(early.X, early.Y, vel_early, levels=30, cmap="viridis")
    plt.colorbar(c4, ax=ax4, label="|V|")
    ax4.set_title("Early (100 steps)")
    ax4.set_xlabel("x")
    ax4.set_ylabel("y")
    ax4.set_aspect("equal")

    # Plot 5: Converged solution
    ax5 = axes[1, 1]
    vel_converged = magnitude(converged.u, converged.v)
    c5 = ax5.contourf(converged.X, converged.Y, vel_converged, levels=30, cmap="viridis")
    plt.colorbar(c5, ax=ax5, label="|V|")
    ax5.set_title("Converged (500 steps)")
    ax5.set_xlabel("x")
    ax5.set_ylabel("y")
    ax5.set_aspect("equal")

    # Plot 6: Convergence difference
    ax6 = axes[1, 2]
    conv_diff = vel_converged - vel_early
    max_conv_diff = max(abs(conv_diff.min()), abs(conv_diff.max()))
    c6 = ax6.contourf(converged.X, converged.Y, conv_diff, levels=30,
                      cmap="RdBu_r", vmin=-max_conv_diff, vmax=max_conv_diff)
    plt.colorbar(c6, ax=ax6, label="Difference")
    ax6.set_title("Converged - Early Difference")
    ax6.set_xlabel("x")
    ax6.set_ylabel("y")
    ax6.set_aspect("equal")

    plt.suptitle("Field Comparison Analysis", fontsize=14)
    plt.tight_layout()

    output_file = "field_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_file}")

    plt.show()


def main():
    """Run field comparison example."""
    print("Field Comparison Example")
    print("=" * 40)
    print()

    # Run comparisons
    fine, u_coarse, v_coarse = compare_grid_resolutions()
    early, converged = compare_iteration_counts()
    ref, vel_ref, vel_test, norms = compute_error_norms_demo()

    if fine is None or early is None or ref is None:
        print("Error: Could not complete all comparisons")
        return

    # Create visualizations
    create_comparison_visualizations(
        fine, u_coarse, v_coarse,
        early, converged,
        ref, vel_ref, vel_test
    )

    print("\n" + "=" * 40)
    print("Field comparison complete!")
    print()
    print("Key functions demonstrated:")
    print("  - compare_fields(): Comprehensive case comparison")
    print("  - compute_field_difference(): Point-wise difference analysis")
    print("  - compute_error_norms(): L1, L2, L-inf error norms")
    print()
    print("Comparison types shown:")
    print("  - Grid resolution comparison (coarse vs fine)")
    print("  - Convergence comparison (early vs converged)")
    print("  - Error norm computation")


if __name__ == "__main__":
    main()
