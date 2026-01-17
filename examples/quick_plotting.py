#!/usr/bin/env python3
"""Quick Plotting Example.

This example demonstrates the quick plotting functions in cfd_viz,
which provide one-liner visualization for rapid iteration during
development and prototyping.

Functions demonstrated:
1. quick_plot_data() - Takes a VTKData object (recommended)
2. quick_plot() - Takes raw velocity data as flat lists

Note: quick_plot_result() is also available for simulation result dicts
that contain 'u' and 'v' keys, but is not demonstrated here since
cfd_python returns VTK files rather than raw velocity arrays.

Usage:
    python examples/quick_plotting.py

Requirements:
    - cfd_python package (pip install -e ../cfd-python)
    - cfd_viz package
"""

import sys

import matplotlib.pyplot as plt

# Check for cfd_python
try:
    import cfd_python
except ImportError:
    print("Error: cfd_python package not installed.")
    print("Install with: pip install -e ../cfd-python")
    sys.exit(1)

from cfd_viz import quick_plot, quick_plot_data
from cfd_viz.common import read_vtk_file


def example_quick_plot_data():
    """Demonstrate quick_plot_data() with VTKData object.

    This is the recommended approach when using cfd_python - run the
    simulation with VTK output and visualize the VTKData object.
    """
    print("Example 1: quick_plot_data()")
    print("-" * 40)

    # Run simulation with VTK output
    vtk_file = "quick_example.vtk"
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

    # Load VTK data
    data = read_vtk_file(vtk_file)
    if data is None:
        print(f"Error: Could not read {vtk_file}")
        return

    # One-liner visualization!
    _fig, ax = quick_plot_data(data)
    ax.set_title("quick_plot_data() - Velocity Magnitude")
    plt.savefig("quick_plot_data_velocity.png", dpi=100)
    print("Saved: quick_plot_data_velocity.png")

    # Different field types
    _fig, ax = quick_plot_data(data, field="vorticity")
    plt.savefig("quick_plot_data_vorticity.png", dpi=100)
    print("Saved: quick_plot_data_vorticity.png")

    # U and V velocity components
    _fig, ax = quick_plot_data(data, field="u")
    plt.savefig("quick_plot_data_u.png", dpi=100)
    print("Saved: quick_plot_data_u.png")

    _fig, ax = quick_plot_data(data, field="v")
    plt.savefig("quick_plot_data_v.png", dpi=100)
    print("Saved: quick_plot_data_v.png")

    # Pressure field
    _fig, ax = quick_plot_data(data, field="p")
    plt.savefig("quick_plot_data_pressure.png", dpi=100)
    print("Saved: quick_plot_data_pressure.png")

    plt.close("all")


def example_quick_plot():
    """Demonstrate quick_plot() with raw velocity data.

    Useful when you have velocity arrays from any source,
    not just VTK files.
    """
    print("\nExample 2: quick_plot()")
    print("-" * 40)

    # Run simulation and get data from VTK
    vtk_file = "quick_raw_example.vtk"
    cfd_python.run_simulation_with_params(
        nx=60,
        ny=60,
        xmin=0.0,
        xmax=1.0,
        ymin=0.0,
        ymax=1.0,
        steps=300,
        output_file=vtk_file,
    )

    data = read_vtk_file(vtk_file)
    if data is None:
        print(f"Error: Could not read {vtk_file}")
        return

    # Convert to flat lists (simulating raw data from other sources)
    u = data.u.flatten().tolist()
    v = data.v.flatten().tolist()
    nx = data.nx
    ny = data.ny

    # Visualize with quick_plot
    _fig, ax = quick_plot(u, v, nx, ny)
    ax.set_title("quick_plot() - From Raw Data")
    plt.savefig("quick_plot_raw.png", dpi=100)
    print("Saved: quick_plot_raw.png")

    # With custom domain bounds
    _fig, ax = quick_plot(
        u, v, nx, ny,
        field="vorticity",
        xmin=-0.5,
        xmax=0.5,
        ymin=-0.5,
        ymax=0.5,
    )
    ax.set_title("quick_plot() - Custom Domain")
    plt.savefig("quick_plot_custom_domain.png", dpi=100)
    print("Saved: quick_plot_custom_domain.png")

    plt.close("all")


def example_comparison_grid():
    """Create a comparison grid showing all field types.

    Demonstrates how quick_plot can be used with existing
    matplotlib subplots for multi-panel figures.
    """
    print("\nExample 3: Comparison Grid")
    print("-" * 40)

    # Run simulation
    vtk_file = "quick_comparison.vtk"
    cfd_python.run_simulation_with_params(
        nx=80,
        ny=80,
        xmin=0.0,
        xmax=1.0,
        ymin=0.0,
        ymax=1.0,
        steps=400,
        output_file=vtk_file,
    )

    data = read_vtk_file(vtk_file)
    if data is None:
        print(f"Error: Could not read {vtk_file}")
        return

    # Create subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot different fields using existing axes
    quick_plot_data(data, field="velocity_magnitude", ax=axes[0, 0])
    axes[0, 0].set_title("Velocity Magnitude")

    quick_plot_data(data, field="vorticity", ax=axes[0, 1])
    axes[0, 1].set_title("Vorticity")

    quick_plot_data(data, field="u", ax=axes[1, 0])
    axes[1, 0].set_title("U-Velocity Component")

    quick_plot_data(data, field="v", ax=axes[1, 1])
    axes[1, 1].set_title("V-Velocity Component")

    plt.suptitle("Quick Plot - Field Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig("quick_plot_comparison.png", dpi=150)
    print("Saved: quick_plot_comparison.png")

    plt.close("all")


def example_custom_styling():
    """Demonstrate customization options.

    Shows how to customize figure size, colormap, levels, etc.
    """
    print("\nExample 4: Custom Styling")
    print("-" * 40)

    # Run simulation
    vtk_file = "quick_styled.vtk"
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

    data = read_vtk_file(vtk_file)
    if data is None:
        print(f"Error: Could not read {vtk_file}")
        return

    # Custom figure size
    _fig, ax = quick_plot_data(
        data,
        field="velocity_magnitude",
        figsize=(10, 8),
        levels=30,
        cmap="plasma",
    )
    ax.set_title("Custom: Large Figure, Plasma Colormap, 30 Levels")
    plt.savefig("quick_plot_styled.png", dpi=100)
    print("Saved: quick_plot_styled.png")

    plt.close("all")


def main():
    """Run all quick plotting examples."""
    print("Quick Plotting Examples")
    print("=" * 40)
    print()
    print("Quick plotting provides one-liner visualization")
    print("for rapid iteration during development.")
    print()

    example_quick_plot_data()
    example_quick_plot()
    example_comparison_grid()
    example_custom_styling()

    print()
    print("=" * 40)
    print("All examples complete!")
    print()
    print("Summary of quick_plot functions:")
    print("  - quick_plot_data(data): For VTKData objects (recommended)")
    print("  - quick_plot(u, v, nx, ny): For raw velocity arrays")
    print("  - quick_plot_result(result): For dicts with u/v keys (not shown)")


if __name__ == "__main__":
    main()
