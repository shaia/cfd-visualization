#!/usr/bin/env python3
"""
Solver Comparison
=================

Run the same problem with different solvers to compare
convergence rates and performance.
"""

import cfd_python

from cfd_viz.common import DATA_DIR, ensure_dirs


def run_with_solver(solver_name, nx=80, ny=80, iterations=2000):
    """Run simulation with a specific solver"""

    if not cfd_python.has_solver(solver_name):
        print(f"  Solver '{solver_name}' not available, skipping")
        return None

    solver_type = getattr(cfd_python, f"SOLVER_{solver_name.upper()}", None)
    if solver_type is None:
        print(f"  Could not find constant for '{solver_name}', skipping")
        return None

    # Configure simulation
    params = cfd_python.get_default_solver_params()
    params["nx"] = nx
    params["ny"] = ny
    params["max_iterations"] = iterations
    params["output_interval"] = iterations  # Only output final state
    params["Re"] = 100.0
    params["solver_type"] = solver_type

    print(f"  Running {solver_name}...")

    try:
        result = cfd_python.run_simulation_with_params(params)
        return {
            "solver": solver_name,
            "iterations": result.get("iterations", 0),
            "residual": result.get("final_residual", float("inf")),
            "converged": result.get("converged", False),
        }
    except Exception as e:
        print(f"  Error: {e}")
        return None


def main():
    ensure_dirs()

    print("Solver Comparison")
    print("=" * 40)
    print()

    # Set output directory
    cfd_python.set_output_dir(str(DATA_DIR))

    # Get available solvers
    solvers = cfd_python.list_solvers()
    print(f"Available solvers: {', '.join(solvers)}")
    print()

    # Run each solver
    results = []
    for solver in solvers:
        result = run_with_solver(solver)
        if result:
            results.append(result)

    # Print comparison table
    print()
    print("Results:")
    print("-" * 60)
    print(f"{'Solver':<20} {'Iterations':>12} {'Residual':>15} {'Converged':>10}")
    print("-" * 60)

    for r in sorted(results, key=lambda x: x["iterations"]):
        print(
            f"{r['solver']:<20} {r['iterations']:>12} {r['residual']:>15.2e} {str(r['converged']):>10}"
        )

    print("-" * 60)

    if results:
        fastest = min(results, key=lambda x: x["iterations"])
        print(
            f"\nFastest solver: {fastest['solver']} ({fastest['iterations']} iterations)"
        )


if __name__ == "__main__":
    main()
