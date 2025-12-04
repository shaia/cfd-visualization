#!/usr/bin/env python3
"""
Lid-Driven Cavity Flow Simulation
=================================

Classic benchmark problem: 2D square cavity with moving top lid.
The top wall moves at constant velocity while other walls are stationary.

This creates a primary vortex in the center and secondary vortices
in the corners at higher Reynolds numbers.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import ensure_dirs
from run_simulation import run_simulation


def main():
    ensure_dirs()

    print("Lid-Driven Cavity Flow")
    print("=" * 40)
    print()
    print("A classic CFD benchmark problem.")
    print("Top wall moves right, other walls stationary.")
    print()

    # Standard lid-driven cavity parameters
    # Square domain with fine mesh for accuracy
    run_simulation(
        nx=100,
        ny=100,
        solver='projection',
        num_iterations=5000
    )

if __name__ == "__main__":
    main()
