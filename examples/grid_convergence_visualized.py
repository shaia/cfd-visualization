#!/usr/bin/env python3
"""
Grid Convergence Study with Visualization
==========================================

Runs simulations at multiple grid resolutions to study how the solution
converges as the mesh is refined. Creates comparison plots showing:
- Velocity magnitude fields at each resolution
- Convergence of maximum velocity with grid size

This is a standard CFD verification technique to ensure mesh-independent results.
"""

import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import cfd_python
import matplotlib.pyplot as plt

from config import DATA_DIR, PLOTS_DIR, ensure_dirs


def run_convergence_study() -> List[Dict[str, Any]]:
    """Run simulations at multiple grid resolutions.

    Returns:
        List of result dictionaries containing grid info and statistics.
    """
    ensure_dirs()
    cfd_python.set_output_dir(str(DATA_DIR))

    print("Grid Convergence Study")
    print("=" * 50)
    print()
    print("Running simulations at increasing grid resolutions")
    print("to verify solution convergence.")
    print()

    # Grid sizes to test (coarse to fine)
    grid_sizes = [20, 40, 60, 80, 100]
    steps = 200

    available_solvers = cfd_python.list_solvers()
    solver = 'projection' if 'projection' in available_solvers else available_solvers[0]

    results = []

    for nx in grid_sizes:
        ny = nx  # Square grid
        output_file = str(DATA_DIR / f"convergence_{nx}x{ny}.vtk")

        print(f"  Running {nx}x{ny} grid...", end=" ", flush=True)

        result = cfd_python.run_simulation_with_params(
            nx=nx,
            ny=ny,
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
            steps=steps,
            solver_type=solver,
            output_file=output_file
        )

        # Get statistics from result
        if isinstance(result, dict) and 'stats' in result:
            max_vel = result['stats'].get('max_velocity', 0)
        else:
            max_vel = 0

        results.append({
            'nx': nx,
            'ny': ny,
            'max_velocity': max_vel,
            'vtk_file': output_file
        })

        print(f"max velocity = {max_vel:.6f}")

    print()
    print("All simulations complete!")

    return results


def create_convergence_plot(results: List[Dict[str, Any]]) -> str:
    """Create plot showing convergence of maximum velocity with grid size.

    Args:
        results: List of result dictionaries from run_convergence_study.

    Returns:
        Path to the saved plot file.
    """
    print()
    print("Creating convergence plot...")

    grid_sizes = [r['nx'] for r in results]
    max_velocities = [r['max_velocity'] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Max velocity vs grid size
    ax1.plot(grid_sizes, max_velocities, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Grid Size (N x N)')
    ax1.set_ylabel('Maximum Velocity')
    ax1.set_title('Grid Convergence: Maximum Velocity')
    ax1.grid(True, alpha=0.3)

    # Add Richardson extrapolation estimate if we have enough points
    if len(grid_sizes) >= 3:
        # Estimate converged value using last 3 points
        h = [1.0 / n for n in grid_sizes[-3:]]
        f = max_velocities[-3:]

        # Simple extrapolation
        extrapolated = f[-1] + (f[-1] - f[-2]) * h[-1] / (h[-2] - h[-1])
        ax1.axhline(
            y=extrapolated, color='r', linestyle='--',
            label=f'Extrapolated: {extrapolated:.6f}'
        )
        ax1.legend()

    # Plot 2: Log-log plot for convergence rate
    h_values = [1.0 / n for n in grid_sizes]
    ax2.loglog(h_values, max_velocities, 'bo-', linewidth=2, markersize=8)
    ax2.set_xlabel('Grid Spacing (h = 1/N)')
    ax2.set_ylabel('Maximum Velocity')
    ax2.set_title('Grid Convergence: Log-Log Scale')
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()

    output_file = str(PLOTS_DIR / 'grid_convergence.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"  Saved: {output_file}")

    return output_file


def create_comparison_plot(results: List[Dict[str, Any]]) -> str:
    """Create side-by-side comparison of velocity fields at different resolutions.

    Args:
        results: List of result dictionaries from run_convergence_study.

    Returns:
        Path to the saved plot file.
    """
    from visualize_cfd import create_velocity_magnitude, read_vtk_file

    print()
    print("Creating comparison plot...")

    # Select subset of results to show (first, middle, last)
    indices = [0, len(results) // 2, -1]
    selected = [results[i] for i in indices]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, result in zip(axes, selected):
        vtk_file = result['vtk_file']
        nx = result['nx']

        if os.path.exists(vtk_file):
            X, Y, data = read_vtk_file(vtk_file)

            if 'u' in data and 'v' in data:
                vel_mag = create_velocity_magnitude(data['u'], data['v'])
                contour = ax.contourf(X, Y, vel_mag, levels=20, cmap='viridis')
                plt.colorbar(contour, ax=ax, label='Velocity')
            elif 'velocity_magnitude' in data:
                contour = ax.contourf(
                    X, Y, data['velocity_magnitude'], levels=20, cmap='viridis'
                )
                plt.colorbar(contour, ax=ax, label='Velocity')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'{nx}x{nx} Grid')
        ax.set_aspect('equal')

    plt.suptitle('Grid Convergence: Velocity Field Comparison', fontsize=14)
    plt.tight_layout()

    output_file = str(PLOTS_DIR / 'grid_comparison.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"  Saved: {output_file}")

    return output_file


def print_summary(results: List[Dict[str, Any]]) -> None:
    """Print summary of convergence study.

    Args:
        results: List of result dictionaries from run_convergence_study.
    """
    print()
    print("=" * 50)
    print("Grid Convergence Summary")
    print("=" * 50)
    print()
    print(f"{'Grid':>10} {'Max Velocity':>15} {'Change':>12}")
    print("-" * 40)

    prev_vel = None
    for r in results:
        change = ""
        if prev_vel is not None and prev_vel != 0:
            pct = abs(r['max_velocity'] - prev_vel) / prev_vel * 100
            change = f"{pct:.2f}%"
        print(f"{r['nx']:>4}x{r['ny']:<4} {r['max_velocity']:>15.6f} {change:>12}")
        prev_vel = r['max_velocity']

    # Check if converged (< 1% change between last two)
    if len(results) >= 2 and results[-2]['max_velocity'] != 0:
        last_change = abs(results[-1]['max_velocity'] - results[-2]['max_velocity'])
        pct_change = last_change / results[-2]['max_velocity'] * 100
        print()
        if pct_change < 1.0:
            print("Solution appears CONVERGED (< 1% change at finest grid)")
        else:
            print(f"Solution may need finer grid (>{pct_change:.1f}% change)")


def main() -> None:
    """Main entry point for grid convergence study."""
    ensure_dirs()

    # Run convergence study
    results = run_convergence_study()

    # Create visualizations
    create_convergence_plot(results)
    create_comparison_plot(results)

    # Print summary
    print_summary(results)

    print()
    print(f"Plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
