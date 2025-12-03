#!/usr/bin/env python3
"""
Interactive HTML Output
=======================

Generates an interactive HTML visualization of CFD simulation results
using Plotly. The output can be opened in any web browser without
requiring Python or any dependencies.

Features:
- Interactive zoom, pan, and hover
- Velocity magnitude contour plot
- Velocity vector field (quiver)
- Self-contained HTML file
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import DATA_DIR, OUTPUT_DIR, ensure_dirs

try:
    import cfd_python
    CFD_AVAILABLE = True
except ImportError:
    CFD_AVAILABLE = False

import numpy as np
from numpy.typing import NDArray

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def run_simulation() -> str | None:
    """Run a simulation and return the output file path."""
    if not CFD_AVAILABLE:
        print("Error: cfd-python not available")
        return None

    ensure_dirs()
    cfd_python.set_output_dir(str(DATA_DIR))

    output_file = str(DATA_DIR / "interactive_output.vtk")

    print("Running simulation...")
    result = cfd_python.run_simulation_with_params(
        nx=50,
        ny=50,
        xmin=0.0,
        xmax=1.0,
        ymin=0.0,
        ymax=1.0,
        steps=200,
        solver_type='projection',
        output_file=output_file
    )

    print(f"Simulation complete: {output_file}")
    return output_file


def read_vtk_file(filepath: str) -> tuple[NDArray, NDArray, dict[str, NDArray]]:
    """Read VTK file and extract grid and data.

    Args:
        filepath: Path to the VTK file.

    Returns:
        Tuple of (X, Y, data) where X and Y are coordinate meshgrids
        and data is a dict mapping field names to 2D arrays.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    nx, ny, nz = 0, 0, 0
    origin = [0.0, 0.0, 0.0]
    spacing = [1.0, 1.0, 1.0]
    data = {}

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith('DIMENSIONS'):
            parts = line.split()
            nx, ny, nz = int(parts[1]), int(parts[2]), int(parts[3])

        elif line.startswith('ORIGIN'):
            parts = line.split()
            origin = [float(parts[1]), float(parts[2]), float(parts[3])]

        elif line.startswith('SPACING'):
            parts = line.split()
            spacing = [float(parts[1]), float(parts[2]), float(parts[3])]

        elif line.startswith('VECTORS'):
            name = line.split()[1]
            values = []
            i += 1
            while i < len(lines) and len(values) < nx * ny:
                line_content = lines[i].strip()
                # Stop if we hit another VTK keyword
                if not line_content or line_content.split()[0].isalpha():
                    break
                parts = line_content.split()
                # VTK VECTORS have 3 components (x, y, z); we only use x, y for 2D
                if len(parts) >= 3:
                    values.append((float(parts[0]), float(parts[1])))
                    i += 1
                else:
                    break
            if values:
                u = np.array([v[0] for v in values]).reshape((ny, nx))
                v = np.array([v[1] for v in values]).reshape((ny, nx))
                data['u'] = u
                data['v'] = v
            continue

        elif line.startswith('SCALARS'):
            name = line.split()[1]
            i += 2  # Skip LOOKUP_TABLE line
            values = []
            while i < len(lines) and len(values) < nx * ny:
                line_content = lines[i].strip()
                # Stop if we hit another VTK keyword (non-numeric line)
                if not line_content or line_content[0].isalpha():
                    break
                try:
                    values.append(float(line_content))
                except ValueError:
                    break
                i += 1
            if values:
                data[name] = np.array(values).reshape((ny, nx))
            continue

        i += 1

    # Create coordinate grids
    x = origin[0] + np.arange(nx) * spacing[0]
    y = origin[1] + np.arange(ny) * spacing[1]
    X, Y = np.meshgrid(x, y)

    return X, Y, data


def create_html_visualization(vtk_file: str, output_file: str) -> str | None:
    """Create interactive HTML visualization from VTK data.

    Args:
        vtk_file: Path to the input VTK file.
        output_file: Path for the output HTML file.

    Returns:
        Path to the created HTML file, or None on error.
    """
    if not PLOTLY_AVAILABLE:
        print("Error: plotly not available. Install with: pip install plotly")
        return None

    print("Reading VTK file...")
    X, Y, data = read_vtk_file(vtk_file)

    # Calculate velocity magnitude
    if 'u' in data and 'v' in data:
        vel_mag = np.sqrt(data['u']**2 + data['v']**2)
    elif 'velocity_magnitude' in data:
        vel_mag = data['velocity_magnitude']
    else:
        print("Error: No velocity data found")
        return None

    print("Creating interactive plot...")

    # Create figure with subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Velocity Magnitude', 'Velocity Vectors'),
        horizontal_spacing=0.1
    )

    # Velocity magnitude contour
    fig.add_trace(
        go.Contour(
            x=X[0, :],
            y=Y[:, 0],
            z=vel_mag,
            colorscale='Viridis',
            colorbar=dict(title='Velocity', x=0.45),
            name='Velocity Magnitude'
        ),
        row=1, col=1
    )

    # Velocity vectors (subsample for clarity)
    skip = 3
    x_sub = X[::skip, ::skip].flatten()
    y_sub = Y[::skip, ::skip].flatten()
    u_sub = data['u'][::skip, ::skip].flatten()
    v_sub = data['v'][::skip, ::skip].flatten()

    # Normalize for display
    mag_sub = np.sqrt(u_sub**2 + v_sub**2)
    scale = 0.02
    u_norm = u_sub / (mag_sub.max() + 1e-10) * scale
    v_norm = v_sub / (mag_sub.max() + 1e-10) * scale

    # Create quiver-like arrows using annotations
    # First add the base contour
    fig.add_trace(
        go.Contour(
            x=X[0, :],
            y=Y[:, 0],
            z=vel_mag,
            colorscale='Viridis',
            showscale=False,
            opacity=0.5,
            name='Background'
        ),
        row=1, col=2
    )

    # Add arrows as scatter with lines
    for xi, yi, ui, vi in zip(x_sub, y_sub, u_norm, v_norm):
        fig.add_annotation(
            x=xi + ui,
            y=yi + vi,
            ax=xi,
            ay=yi,
            xref='x2',
            yref='y2',
            axref='x2',
            ayref='y2',
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1,
            arrowcolor='black'
        )

    # Update layout
    fig.update_layout(
        title=dict(
            text='CFD Simulation Results - Interactive Visualization',
            font=dict(size=20)
        ),
        height=600,
        width=1200,
        showlegend=False
    )

    # Set equal aspect ratio
    fig.update_xaxes(title_text='X', row=1, col=1, scaleanchor='y', scaleratio=1)
    fig.update_yaxes(title_text='Y', row=1, col=1)
    fig.update_xaxes(title_text='X', row=1, col=2, scaleanchor='y2', scaleratio=1)
    fig.update_yaxes(title_text='Y', row=1, col=2)

    # Save to HTML
    fig.write_html(output_file, include_plotlyjs=True, full_html=True)
    print(f"Saved: {output_file}")

    return output_file


def main() -> None:
    """Main entry point for interactive HTML visualization."""
    if not PLOTLY_AVAILABLE:
        print("Error: plotly is required for this example")
        print("Install with: pip install plotly")
        return

    ensure_dirs()

    print("Interactive HTML Output")
    print("=" * 40)
    print()

    # Run simulation
    vtk_file = run_simulation()
    if vtk_file is None:
        return

    # Create HTML output
    html_file = str(OUTPUT_DIR / "interactive_cfd_visualization.html")
    result = create_html_visualization(vtk_file, html_file)

    if result:
        print()
        print("Done!")
        print(f"Open in browser: {html_file}")


if __name__ == "__main__":
    main()
