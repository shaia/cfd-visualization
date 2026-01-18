#!/usr/bin/env python3
"""CFD-Python Integration Example.

This example demonstrates the direct integration between cfd-python and
cfd-visualization, showcasing the new features in cfd-python 0.1.6:

1. System info and backend detection
2. Quick plotting for rapid visualization
3. Flow statistics with cfd-python acceleration
4. VTKData conversion utilities

Usage:
    python examples/cfd_python_integration.py

Requirements:
    - cfd_python >= 0.1.6
    - cfd_viz package
"""

import sys

import matplotlib.pyplot as plt

# Check for cfd_python
try:
    import cfd_python
except ImportError:
    print("Error: cfd_python package not installed.")
    print("Install with: pip install cfd-python>=0.1.6")
    sys.exit(1)

from cfd_viz import (
    print_system_info,
    get_system_info,
    get_recommended_settings,
    quick_plot_data,
    compute_flow_statistics,
    calculate_field_stats,
)
from cfd_viz.common import read_vtk_file


def demo_system_info():
    """Demonstrate system info and backend detection."""
    print("1. System Information")
    print("-" * 40)

    # Print formatted system info
    print_system_info()
    print()

    # Get info programmatically
    info = get_system_info()
    settings = get_recommended_settings()

    print("Programmatic access:")
    print(f"  cfd-python version: {info['cfd_python_version']}")
    print(f"  SIMD available: {info['has_simd']} ({info['simd']})")
    print(f"  Recommended chunk size: {settings['chunk_size']}")
    print()


def demo_quick_plotting():
    """Demonstrate quick plotting for rapid visualization."""
    print("2. Quick Plotting")
    print("-" * 40)

    # Run a simulation
    print("Running simulation...")
    vtk_file = "integration_demo.vtk"
    cfd_python.run_simulation_with_params(
        nx=50,
        ny=50,
        xmin=0.0,
        xmax=1.0,
        ymin=0.0,
        ymax=1.0,
        steps=200,
        output_file=vtk_file,
    )

    # Load and visualize with one-liner
    data = read_vtk_file(vtk_file)
    if data is None:
        print(f"Error: Could not read {vtk_file}")
        return None

    # Quick plot - velocity magnitude
    _fig, ax = quick_plot_data(data, field="velocity_magnitude")
    ax.set_title("Quick Plot: Velocity Magnitude")
    plt.savefig("integration_velocity.png", dpi=100)
    print("Saved: integration_velocity.png")

    # Quick plot - vorticity
    _fig, ax = quick_plot_data(data, field="vorticity")
    ax.set_title("Quick Plot: Vorticity")
    plt.savefig("integration_vorticity.png", dpi=100)
    print("Saved: integration_vorticity.png")

    plt.close("all")
    print()

    return data


def demo_flow_statistics(data):
    """Demonstrate flow statistics with cfd-python acceleration."""
    print("3. Flow Statistics")
    print("-" * 40)

    if data is None:
        print("No data available for statistics.")
        return

    # Compute comprehensive flow statistics
    # Uses cfd-python acceleration when available
    stats = compute_flow_statistics(data)

    print("Velocity magnitude:")
    print(f"  Min: {stats['velocity_magnitude']['min']:.6f}")
    print(f"  Max: {stats['velocity_magnitude']['max']:.6f}")
    print(f"  Avg: {stats['velocity_magnitude']['avg']:.6f}")
    print()

    print("U velocity:")
    print(f"  Min: {stats['u']['min']:.6f}")
    print(f"  Max: {stats['u']['max']:.6f}")
    print(f"  Avg: {stats['u']['avg']:.6f}")
    print()

    print("V velocity:")
    print(f"  Min: {stats['v']['min']:.6f}")
    print(f"  Max: {stats['v']['max']:.6f}")
    print(f"  Avg: {stats['v']['avg']:.6f}")
    print()

    # Also demonstrate single-field statistics
    u_stats = calculate_field_stats(data.u.flatten().tolist())
    print(f"U stats via calculate_field_stats: max={u_stats['max']:.6f}")
    print()


def demo_comparison_plot(data):
    """Create a comparison plot showing multiple fields."""
    print("4. Multi-Field Comparison")
    print("-" * 40)

    if data is None:
        print("No data available for comparison.")
        return

    # Create a 2x2 comparison grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    quick_plot_data(data, field="velocity_magnitude", ax=axes[0, 0])
    axes[0, 0].set_title("Velocity Magnitude")

    quick_plot_data(data, field="vorticity", ax=axes[0, 1])
    axes[0, 1].set_title("Vorticity")

    quick_plot_data(data, field="u", ax=axes[1, 0])
    axes[1, 0].set_title("U Velocity")

    quick_plot_data(data, field="v", ax=axes[1, 1])
    axes[1, 1].set_title("V Velocity")

    plt.suptitle("CFD-Python + CFD-Viz Integration", fontsize=14)
    plt.tight_layout()
    plt.savefig("integration_comparison.png", dpi=150)
    print("Saved: integration_comparison.png")
    plt.close(fig)
    print()


def main():
    """Run all integration demos."""
    print("CFD-Python Integration Demo")
    print("=" * 40)
    print()

    # Demo 1: System info
    demo_system_info()

    # Demo 2: Quick plotting
    data = demo_quick_plotting()

    # Demo 3: Flow statistics
    demo_flow_statistics(data)

    # Demo 4: Comparison plot
    demo_comparison_plot(data)

    print("=" * 40)
    print("Integration demo complete!")
    print()
    print("Key features demonstrated:")
    print("  - print_system_info(): Show backend capabilities")
    print("  - quick_plot_data(): One-liner visualization")
    print("  - compute_flow_statistics(): Accelerated statistics")
    print("  - Multi-field comparison plots")


if __name__ == "__main__":
    main()
