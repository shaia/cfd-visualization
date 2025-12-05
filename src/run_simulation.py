#!/usr/bin/env python3
"""
Run CFD Simulation and Visualize Results
=========================================

Runs a CFD simulation using the cfd-python wrapper and creates visualizations.

Usage:
    python run_simulation.py [options]

    # Run simulation with default parameters
    python run_simulation.py

    # Run simulation and create visualizations
    python run_simulation.py --visualize

    # Custom grid size
    python run_simulation.py --nx 200 --ny 100

    # Specify solver
    python run_simulation.py --solver jacobi

Requirements:
    - cfd-python package must be installed (pip install cfd-python)
"""

import argparse
import os
import sys

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, ensure_dirs

CFD_AVAILABLE = False
CFD_ERROR = None

try:
    import cfd_python
    # Check if the C extension is actually available (not just the stub)
    if hasattr(cfd_python, 'run_simulation'):
        CFD_AVAILABLE = True
    else:
        CFD_ERROR = "cfd-python C extension not built. Run 'pip install -e .' in cfd-python directory."
except ImportError as e:
    CFD_ERROR = f"cfd-python package not installed: {e}"


def run_simulation(nx=100, ny=50, solver=None, num_iterations=1000):
    """Run CFD simulation using cfd-python wrapper"""

    if not CFD_AVAILABLE:
        print(f"Error: {CFD_ERROR}")
        print("\nTo build cfd-python:")
        print("  cd ../cfd-python")
        print("  pip install -e .")
        return False

    # Set output directory to our centralized data directory
    cfd_python.set_output_dir(str(DATA_DIR))

    print("CFD Simulation")
    print("==============")
    print(f"Grid: {nx} x {ny}")
    print(f"Iterations: {num_iterations}")
    print(f"Output directory: {DATA_DIR}")

    # List available solvers
    available_solvers = cfd_python.list_solvers()
    print(f"\nAvailable solvers: {', '.join(available_solvers)}")

    # Determine solver to use
    if solver:
        if not cfd_python.has_solver(solver):
            print(f"Error: Solver '{solver}' not available")
            print(f"Available solvers: {', '.join(available_solvers)}")
            return False
    else:
        # Use default (first available)
        solver = available_solvers[0] if available_solvers else "jacobi"

    print(f"Using solver: {solver}")
    print("\nRunning simulation...")

    try:
        # Generate output filename
        output_file = str(DATA_DIR / f"flow_field_{nx}x{ny}.vtk")

        # Run simulation with keyword arguments
        result = cfd_python.run_simulation_with_params(
            nx=nx,
            ny=ny,
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
            steps=num_iterations,
            solver_type=solver,
            output_file=output_file
        )

        print("\nSimulation complete!")
        if isinstance(result, dict):
            print(f"Output file: {result.get('output_file', output_file)}")
            print(f"Solver: {result.get('solver_name', 'N/A')}")
            if 'stats' in result:
                stats = result['stats']
                print(f"Max velocity: {stats.get('max_velocity', 'N/A')}")

        return True

    except Exception as e:
        print(f"Simulation failed: {e}")
        return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Run CFD simulation and create visualizations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_simulation.py                    # Run with defaults
  python run_simulation.py --visualize        # Run and visualize
  python run_simulation.py --nx 200 --ny 100  # Custom grid size
  python run_simulation.py --solver gauss_seidel
        """
    )

    parser.add_argument('--nx', type=int, default=100,
                       help='Grid points in x direction (default: 100)')
    parser.add_argument('--ny', type=int, default=50,
                       help='Grid points in y direction (default: 50)')
    parser.add_argument('--solver', '-s', type=str, default=None,
                       help='Solver type (jacobi, gauss_seidel, sor, etc.)')
    parser.add_argument('--iterations', '-i', type=int, default=1000,
                       help='Maximum iterations (default: 1000)')
    parser.add_argument('--visualize', '-v', action='store_true',
                       help='Create visualizations after simulation')
    parser.add_argument('--list-solvers', action='store_true',
                       help='List available solvers and exit')

    args = parser.parse_args()

    # Ensure output directories exist
    ensure_dirs()

    # List solvers if requested
    if args.list_solvers:
        if not CFD_AVAILABLE:
            print(f"Error: {CFD_ERROR}")
            return
        solvers = cfd_python.list_solvers()
        print("Available solvers:")
        for solver in solvers:
            info = cfd_python.get_solver_info(solver)
            print(f"  - {solver}: {info.get('description', 'No description')}")
        return

    # Run simulation
    success = run_simulation(
        nx=args.nx,
        ny=args.ny,
        solver=args.solver,
        num_iterations=args.iterations
    )

    if not success:
        sys.exit(1)

    # Create visualizations if requested
    if args.visualize:
        print("\nCreating visualizations...")
        try:
            from visualize_cfd import (
                create_animations,
                create_static_plots,
                find_vtk_files,
            )
            vtk_files = find_vtk_files()
            if vtk_files:
                create_static_plots(vtk_files)
                if len(vtk_files) > 1:
                    create_animations(vtk_files)
            else:
                print("No VTK files found for visualization")
        except ImportError as e:
            print(f"Could not import visualization module: {e}")
        except Exception as e:
            print(f"Visualization failed: {e}")


if __name__ == "__main__":
    main()
