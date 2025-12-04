#!/usr/bin/env python3
"""
Quick Test Simulation
=====================

Small, fast simulation to verify everything is working.
Good for testing the setup before running larger simulations.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import ensure_dirs
from run_simulation import run_simulation


def main():
    ensure_dirs()

    print("Quick Test Simulation")
    print("=" * 40)
    print()
    print("Small grid for fast verification.")
    print()

    # Small and fast
    success = run_simulation(
        nx=50,
        ny=50,
        solver=None,  # Use default
        num_iterations=500
    )

    if success:
        print("\nTest passed! CFD simulation working correctly.")
        print("\nTo visualize results:")
        print("  python src/visualize_cfd.py --all")
    else:
        print("\nTest failed. Check cfd-python installation.")

if __name__ == "__main__":
    main()
