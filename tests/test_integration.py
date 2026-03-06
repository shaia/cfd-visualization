"""End-to-end integration tests for cfd-visualization pipeline."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from cfd_viz.common import read_vtk_file
from cfd_viz.convert import from_cfd_python
from cfd_viz.fields import magnitude, vorticity
from cfd_viz.plotting import plot_contour_field

SAMPLE_VTK_DIR = Path(__file__).parent.parent / "data" / "vtk_files"


@pytest.mark.integration
class TestEndToEnd:
    def test_vtk_read_compute_plot(self):
        """Read VTK file -> compute derived fields -> plot -> no exceptions."""
        data = read_vtk_file(str(SAMPLE_VTK_DIR / "flow_field_50x50_Re100.vtk"))
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

    def test_field_alias_in_pipeline(self):
        """Verify aliases work through the full pipeline."""
        data = read_vtk_file(str(SAMPLE_VTK_DIR / "animated_flow_0050.vtk"))
        assert data is not None

        # "pressure" was normalized to "p" at read time
        p = data.get("pressure")
        assert p is not None

        fig, ax = plt.subplots()
        plot_contour_field(data.X, data.Y, p, ax=ax)
        plt.close(fig)

    def test_version_accessible(self):
        """__version__ should be a non-empty string."""
        from cfd_viz import __version__

        assert isinstance(__version__, str)
        assert len(__version__) > 0

    def test_vtk_data_repr(self):
        """VTKData repr should be informative."""
        data = read_vtk_file(str(SAMPLE_VTK_DIR / "flow_field_50x50_Re100.vtk"))
        r = repr(data)
        assert "VTKData" in r
        assert "50" in r
        assert "u" in r
