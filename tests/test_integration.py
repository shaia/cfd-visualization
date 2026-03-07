"""End-to-end integration tests for cfd-visualization pipeline."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from cfd_viz.common import read_vtk_file
from cfd_viz.convert import from_cfd_python
from cfd_viz.fields import magnitude, vorticity
from cfd_viz.plotting import plot_contour_field


def _write_structured_points(
    path, nx, ny, vectors=None, scalars=None, origin=(0, 0, 0), spacing=(1, 1, 1)
):
    """Write a minimal STRUCTURED_POINTS VTK file."""
    lines = [
        "# vtk DataFile Version 3.0",
        "Test",
        "ASCII",
        "DATASET STRUCTURED_POINTS",
        f"DIMENSIONS {nx} {ny} 1",
        f"ORIGIN {origin[0]} {origin[1]} {origin[2]}",
        f"SPACING {spacing[0]} {spacing[1]} {spacing[2]}",
        "",
        f"POINT_DATA {nx * ny}",
    ]
    if vectors is not None:
        lines.append("VECTORS velocity float")
        for u, v in zip(vectors[0].ravel(), vectors[1].ravel()):
            lines.append(f"{u} {v} 0.0")
    if scalars is not None:
        for name, data in scalars.items():
            lines.append(f"SCALARS {name} float 1")
            lines.append("LOOKUP_TABLE default")
            for val in data.ravel():
                lines.append(str(val))
    path.write_text("\n".join(lines) + "\n")


@pytest.mark.integration
class TestEndToEnd:
    def test_vtk_read_compute_plot(self, tmp_path):
        """Read VTK file -> compute derived fields -> plot -> no exceptions."""
        nx, ny = 50, 50
        rng = np.random.default_rng(0)
        u = rng.standard_normal((ny, nx))
        v = rng.standard_normal((ny, nx))
        vtk_file = tmp_path / "flow_field.vtk"
        _write_structured_points(vtk_file, nx=nx, ny=ny, vectors=(u, v))

        data = read_vtk_file(str(vtk_file))
        assert data is not None

        speed = magnitude(data.u, data.v)
        omega = vorticity(data.u, data.v, data.dx, data.dy)

        assert speed.shape == (data.ny, data.nx)
        assert omega.shape == (data.ny, data.nx)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        plot_contour_field(data.X, data.Y, speed, ax=axes[0], title="Speed")
        plot_contour_field(data.X, data.Y, omega, ax=axes[1], title="Vorticity")
        plt.close(fig)

    def test_cfd_python_convert_plot(self):
        """from_cfd_python -> compute -> plot -> no exceptions."""
        nx, ny = 20, 20
        rng = np.random.default_rng(42)
        u_flat = rng.standard_normal(nx * ny).tolist()
        v_flat = rng.standard_normal(nx * ny).tolist()
        p_flat = rng.standard_normal(nx * ny).tolist()

        data = from_cfd_python(u_flat, v_flat, nx=nx, ny=ny, p=p_flat)

        speed = magnitude(data.u, data.v)
        fig, ax = plt.subplots()
        plot_contour_field(data.X, data.Y, speed, ax=ax)
        plt.close(fig)

    def test_field_alias_in_pipeline(self, tmp_path):
        """Verify aliases work through the full pipeline."""
        nx, ny = 10, 8
        u = np.ones((ny, nx))
        v = np.zeros((ny, nx))
        p = np.arange(nx * ny, dtype=float).reshape(ny, nx)
        vtk_file = tmp_path / "animated_flow.vtk"
        _write_structured_points(
            vtk_file, nx=nx, ny=ny, vectors=(u, v), scalars={"pressure": p}
        )

        data = read_vtk_file(str(vtk_file))
        assert data is not None

        # "pressure" was normalized to "p" at read time
        p_field = data.get("pressure")
        assert p_field is not None

        fig, ax = plt.subplots()
        plot_contour_field(data.X, data.Y, p_field, ax=ax)
        plt.close(fig)

    def test_version_accessible(self):
        """__version__ should be a non-empty string."""
        from cfd_viz import __version__

        assert isinstance(__version__, str)
        assert len(__version__) > 0

    def test_vtk_data_repr(self, tmp_path):
        """VTKData repr should be informative."""
        nx, ny = 50, 50
        u = np.ones((ny, nx))
        v = np.zeros((ny, nx))
        vtk_file = tmp_path / "flow_field.vtk"
        _write_structured_points(vtk_file, nx=nx, ny=ny, vectors=(u, v))

        data = read_vtk_file(str(vtk_file))
        r = repr(data)
        assert "VTKData" in r
        assert "50" in r
        assert "u" in r
