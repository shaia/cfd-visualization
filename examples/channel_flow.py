#!/usr/bin/env python3
"""
Channel Flow Simulation
=======================

Flow through a rectangular channel with inlet on the left
and outlet on the right. Develops into parabolic Poiseuille
flow profile at steady state.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import ensure_dirs
from run_simulation import run_simulation


def main():
    ensure_dirs()

    print("Channel Flow Simulation")
    print("=" * 40)
    print()
    print("Flow through a rectangular channel.")
    print("Develops parabolic velocity profile.")
    print()

    # Rectangular channel - longer in x direction
    run_simulation(
        nx=200,
        ny=50,
        solver='projection',
        num_iterations=3000
    )

if __name__ == "__main__":
    main()
