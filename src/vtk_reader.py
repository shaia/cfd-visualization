#!/usr/bin/env python3
"""
VTK File Reader Utility
=======================

Shared VTK file parsing for CFD visualization scripts.
Supports STRUCTURED_POINTS and RECTILINEAR_GRID formats.

This module provides a unified interface for reading VTK files
across all visualization scripts in the project.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


class VTKData:
    """Container for VTK file data with convenient access patterns."""

    def __init__(
        self,
        x: NDArray,
        y: NDArray,
        X: NDArray,
        Y: NDArray,
        fields: Dict[str, NDArray],
        nx: int,
        ny: int,
        dx: float,
        dy: float
    ):
        self.x = x
        self.y = y
        self.X = X
        self.Y = Y
        self.fields = fields
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy

    @property
    def u(self) -> Optional[NDArray]:
        """X-velocity component."""
        return self.fields.get('u')

    @property
    def v(self) -> Optional[NDArray]:
        """Y-velocity component."""
        return self.fields.get('v')

    def __getitem__(self, key: str) -> NDArray:
        """Access fields by name."""
        return self.fields[key]

    def get(self, key: str, default: Any = None) -> Any:
        """Get field with default value."""
        return self.fields.get(key, default)

    def keys(self):
        """Return field names."""
        return self.fields.keys()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for backward compatibility."""
        result = {
            'x': self.x,
            'y': self.y,
            'X': self.X,
            'Y': self.Y,
            'nx': self.nx,
            'ny': self.ny,
            'dx': self.dx,
            'dy': self.dy,
        }
        result.update(self.fields)
        return result


def read_vtk_file(filename: str) -> Optional[VTKData]:
    """Read a VTK structured points or rectilinear grid file.

    Supports both STRUCTURED_POINTS and RECTILINEAR_GRID formats.
    Handles SCALARS and VECTORS data fields.

    Args:
        filename: Path to the VTK file.

    Returns:
        VTKData object containing grid coordinates and field data,
        or None if the file cannot be read.

    Raises:
        ValueError: If the VTK file format is invalid.
    """
    try:
        with open(filename) as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: File {filename} not found")
        return None

    # Parse header
    nx, ny = 0, 0
    origin = [0.0, 0.0, 0.0]
    spacing = [1.0, 1.0, 1.0]
    x_coords = None
    y_coords = None
    fields: Dict[str, NDArray] = {}

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith('DIMENSIONS'):
            parts = line.split()
            nx, ny = int(parts[1]), int(parts[2])

        elif line.startswith('ORIGIN'):
            parts = line.split()
            origin = [float(parts[1]), float(parts[2]), float(parts[3])]

        elif line.startswith('SPACING'):
            parts = line.split()
            spacing = [float(parts[1]), float(parts[2]), float(parts[3])]

        elif line.startswith('X_COORDINATES'):
            n = int(line.split()[1])
            i += 1
            x_coords = []
            while len(x_coords) < n and i < len(lines):
                x_coords.extend([float(val) for val in lines[i].strip().split()])
                i += 1
            i -= 1

        elif line.startswith('Y_COORDINATES'):
            n = int(line.split()[1])
            i += 1
            y_coords = []
            while len(y_coords) < n and i < len(lines):
                y_coords.extend([float(val) for val in lines[i].strip().split()])
                i += 1
            i -= 1

        elif line.startswith('POINT_DATA'):
            pass  # POINT_DATA marks start of field data, processed by subsequent sections

        elif line.startswith('VECTORS'):
            # Parse VECTORS format: each line has 3 components (u, v, w)
            i += 1
            u_values = []
            v_values = []
            while i < len(lines) and len(u_values) < nx * ny:
                parts = lines[i].strip().split()
                # VTK VECTORS have 3 components (x, y, z)
                if len(parts) >= 3:
                    u_values.append(float(parts[0]))
                    v_values.append(float(parts[1]))
                    i += 1
                elif len(parts) == 0:
                    i += 1
                else:
                    break
            if u_values:
                fields['u'] = np.array(u_values).reshape((ny, nx))
                fields['v'] = np.array(v_values).reshape((ny, nx))
            continue

        elif line.startswith('SCALARS'):
            field_name = line.split()[1]
            i += 1
            # Skip LOOKUP_TABLE line if present
            if i < len(lines) and lines[i].strip().startswith('LOOKUP_TABLE'):
                i += 1

            values = []
            while i < len(lines) and len(values) < nx * ny:
                line_content = lines[i].strip()
                # Stop if we hit another VTK keyword
                if not line_content:
                    i += 1
                    continue
                if line_content.split()[0].isalpha():
                    break
                try:
                    values.extend([float(x) for x in line_content.split()])
                except ValueError:
                    break
                i += 1

            if len(values) == nx * ny:
                fields[field_name] = np.array(values).reshape((ny, nx))
            continue

        i += 1

    # Validate required data
    if nx == 0 or ny == 0:
        raise ValueError(f"Invalid VTK file format: dimensions not found in {filename}")

    # Create coordinate arrays
    if x_coords is not None and y_coords is not None:
        # RECTILINEAR_GRID format
        x = np.array(x_coords)
        y = np.array(y_coords)
        dx = x[1] - x[0] if len(x) > 1 else 1.0
        dy = y[1] - y[0] if len(y) > 1 else 1.0
    else:
        # STRUCTURED_POINTS format
        x = np.linspace(origin[0], origin[0] + (nx - 1) * spacing[0], nx)
        y = np.linspace(origin[1], origin[1] + (ny - 1) * spacing[1], ny)
        dx = spacing[0]
        dy = spacing[1]

    X, Y = np.meshgrid(x, y)

    return VTKData(
        x=x, y=y, X=X, Y=Y,
        fields=fields,
        nx=nx, ny=ny, dx=dx, dy=dy
    )


def read_vtk_velocity(filename: str) -> Tuple[
    Optional[NDArray], Optional[NDArray],
    Optional[NDArray], Optional[NDArray]
]:
    """Read VTK file and return grid coordinates and velocity components.

    Convenience function for scripts that only need velocity data.

    Args:
        filename: Path to the VTK file.

    Returns:
        Tuple of (X, Y, u, v) where X, Y are coordinate meshgrids
        and u, v are velocity components. Returns (None, None, None, None)
        if the file cannot be read.
    """
    data = read_vtk_file(filename)
    if data is None:
        return None, None, None, None
    return data.X, data.Y, data.u, data.v
