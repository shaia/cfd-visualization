#!/usr/bin/env python3
"""Error Handling Example.

This example demonstrates proper error handling when using cfd-python
with cfd-visualization:

1. Checking for cfd-python availability
2. Version requirements
3. Handling simulation errors
4. Graceful fallback when cfd-python is not available

Usage:
    python examples/error_handling.py

Requirements:
    - cfd_viz package
    - cfd_python >= 0.1.6 (optional, demonstrates fallback)
"""

from cfd_viz import (
    has_cfd_python,
    get_cfd_python_version,
    require_cfd_python,
    check_cfd_python_version,
)


def demo_availability_check():
    """Demonstrate checking cfd-python availability."""
    print("1. Availability Check")
    print("-" * 40)

    if has_cfd_python():
        version = get_cfd_python_version()
        print(f"cfd-python is available (version {version})")
    else:
        print("cfd-python is NOT available")
        print("Some features will use NumPy fallback")

    print()


def demo_version_check():
    """Demonstrate version checking."""
    print("2. Version Check")
    print("-" * 40)

    if not has_cfd_python():
        print("cfd-python not installed, skipping version check")
        print()
        return

    # Check if version meets requirements
    min_version = "0.1.6"
    if check_cfd_python_version(min_version):
        print(f"Version check passed: >= {min_version}")
    else:
        print(f"Version check FAILED: requires >= {min_version}")
        print(f"  Installed: {get_cfd_python_version()}")

    print()


def demo_require_function():
    """Demonstrate require_cfd_python function."""
    print("3. Require Function")
    print("-" * 40)

    def feature_requiring_cfd_python():
        """This function requires cfd-python."""
        # Call require_cfd_python to ensure it's available
        require_cfd_python("this feature")
        import cfd_python
        return f"Using cfd-python {cfd_python.__version__}"

    try:
        result = feature_requiring_cfd_python()
        print(f"Function executed successfully: {result}")
    except ImportError as e:
        print(f"Function failed (expected if cfd-python not installed): {e}")

    print()


def demo_cfd_error_handling():
    """Demonstrate cfd-python error handling."""
    print("4. CFD Error Handling")
    print("-" * 40)

    if not has_cfd_python():
        print("cfd-python not installed, skipping error handling demo")
        print()
        return

    import cfd_python
    from cfd_python import CFDError

    # Example 1: Successful operation
    print("Running valid simulation...")
    try:
        result = cfd_python.run_simulation_with_params(
            nx=10,
            ny=10,
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
            steps=10,
        )
        print(f"  Success! Steps completed: {result.get('steps', 'N/A')}")
    except CFDError as e:
        print(f"  CFD Error: {e}")

    # Example 2: Checking status
    print("\nChecking last operation status...")
    status = cfd_python.get_last_status()
    if status == cfd_python.CFD_SUCCESS:
        print("  Status: Success")
    else:
        error_msg = cfd_python.get_error_string(status)
        print(f"  Status: Error - {error_msg}")

    print()


def demo_graceful_fallback():
    """Demonstrate graceful fallback for statistics."""
    print("5. Graceful Fallback")
    print("-" * 40)

    import numpy as np
    from cfd_viz import calculate_field_stats

    # Create sample data
    data = np.random.randn(100).tolist()

    # calculate_field_stats will use cfd-python if available,
    # otherwise falls back to NumPy
    stats = calculate_field_stats(data)

    backend = "cfd-python" if has_cfd_python() else "NumPy"
    print(f"Statistics computed using: {backend}")
    print(f"  Min: {stats['min']:.4f}")
    print(f"  Max: {stats['max']:.4f}")
    print(f"  Avg: {stats['avg']:.4f}")
    print(f"  Sum: {stats['sum']:.4f}")

    print()


def main():
    """Run all error handling demos."""
    print("Error Handling Demo")
    print("=" * 40)
    print()

    demo_availability_check()
    demo_version_check()
    demo_require_function()
    demo_cfd_error_handling()
    demo_graceful_fallback()

    print("=" * 40)
    print("Error handling demo complete!")
    print()
    print("Key patterns demonstrated:")
    print("  - has_cfd_python(): Check availability before use")
    print("  - check_cfd_python_version(): Verify version requirements")
    print("  - require_cfd_python(): Raise error if not available")
    print("  - CFDError: Exception handling for simulation errors")
    print("  - Graceful fallback: NumPy when cfd-python unavailable")


if __name__ == "__main__":
    main()
