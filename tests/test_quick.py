"""Tests for cfd_viz.quick module."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

matplotlib.use("Agg")  # Use non-interactive backend for tests

from cfd_viz.common import VTKData
from cfd_viz.convert import from_cfd_python
from cfd_viz.quick import quick_plot, quick_plot_data, quick_plot_result


@pytest.fixture
def sample_data():
    """Create sample simulation data."""
    nx, ny = 10, 10
    u = [float(i) for i in range(nx * ny)]
    v = [float(i * 0.5) for i in range(nx * ny)]
    p = [float(i * 100) for i in range(nx * ny)]
    return {"u": u, "v": v, "p": p, "nx": nx, "ny": ny}


@pytest.fixture
def sample_vtk_data():
    """Create sample VTKData object."""
    return from_cfd_python(
        u=[1.0] * 100,
        v=[2.0] * 100,
        nx=10,
        ny=10,
        p=[101325.0] * 100,
    )


class TestQuickPlot:
    """Tests for quick_plot function."""

    def test_returns_figure_and_axes(self, sample_data):
        """Should return figure and axes tuple."""
        fig, ax = quick_plot(
            sample_data["u"],
            sample_data["v"],
            sample_data["nx"],
            sample_data["ny"],
        )

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_velocity_magnitude_default(self, sample_data):
        """Should plot velocity magnitude by default."""
        fig, ax = quick_plot(
            sample_data["u"],
            sample_data["v"],
            sample_data["nx"],
            sample_data["ny"],
        )

        assert ax.get_title() == "Velocity Magnitude"
        plt.close(fig)

    def test_vorticity_field(self, sample_data):
        """Should plot vorticity when specified."""
        fig, ax = quick_plot(
            sample_data["u"],
            sample_data["v"],
            sample_data["nx"],
            sample_data["ny"],
            field="vorticity",
        )

        assert ax.get_title() == "Vorticity"
        plt.close(fig)

    def test_u_velocity_field(self, sample_data):
        """Should plot u velocity when specified."""
        fig, ax = quick_plot(
            sample_data["u"],
            sample_data["v"],
            sample_data["nx"],
            sample_data["ny"],
            field="u",
        )

        assert ax.get_title() == "U Velocity"
        plt.close(fig)

    def test_v_velocity_field(self, sample_data):
        """Should plot v velocity when specified."""
        fig, ax = quick_plot(
            sample_data["u"],
            sample_data["v"],
            sample_data["nx"],
            sample_data["ny"],
            field="v",
        )

        assert ax.get_title() == "V Velocity"
        plt.close(fig)

    def test_pressure_field(self, sample_data):
        """Should plot pressure when specified and provided."""
        fig, ax = quick_plot(
            sample_data["u"],
            sample_data["v"],
            sample_data["nx"],
            sample_data["ny"],
            field="p",
            p=sample_data["p"],
        )

        assert ax.get_title() == "Pressure"
        plt.close(fig)

    def test_pressure_without_data_raises(self, sample_data):
        """Should raise ValueError when pressure requested but not provided."""
        with pytest.raises(ValueError, match="Pressure field required"):
            quick_plot(
                sample_data["u"],
                sample_data["v"],
                sample_data["nx"],
                sample_data["ny"],
                field="p",
            )

    def test_unknown_field_raises(self, sample_data):
        """Should raise ValueError for unknown field type."""
        with pytest.raises(ValueError, match="Unknown field"):
            quick_plot(
                sample_data["u"],
                sample_data["v"],
                sample_data["nx"],
                sample_data["ny"],
                field="invalid",  # type: ignore
            )

    def test_custom_title(self, sample_data):
        """Should use custom title when provided."""
        fig, ax = quick_plot(
            sample_data["u"],
            sample_data["v"],
            sample_data["nx"],
            sample_data["ny"],
            title="Custom Title",
        )

        assert ax.get_title() == "Custom Title"
        plt.close(fig)

    def test_existing_axes(self, sample_data):
        """Should plot on existing axes when provided."""
        fig, existing_ax = plt.subplots()

        returned_fig, returned_ax = quick_plot(
            sample_data["u"],
            sample_data["v"],
            sample_data["nx"],
            sample_data["ny"],
            ax=existing_ax,
        )

        assert returned_ax is existing_ax
        assert returned_fig is fig
        plt.close(fig)

    def test_custom_figsize(self, sample_data):
        """Should use custom figsize when creating figure."""
        fig, ax = quick_plot(
            sample_data["u"],
            sample_data["v"],
            sample_data["nx"],
            sample_data["ny"],
            figsize=(12, 8),
        )

        assert fig.get_figwidth() == pytest.approx(12)
        assert fig.get_figheight() == pytest.approx(8)
        plt.close(fig)

    def test_custom_domain_bounds(self, sample_data):
        """Should use custom domain bounds."""
        fig, ax = quick_plot(
            sample_data["u"],
            sample_data["v"],
            sample_data["nx"],
            sample_data["ny"],
            xmin=-1.0,
            xmax=1.0,
            ymin=-2.0,
            ymax=2.0,
        )

        # Check that the axes limits are approximately correct
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        assert xlim[0] <= -1.0
        assert xlim[1] >= 1.0
        assert ylim[0] <= -2.0
        assert ylim[1] >= 2.0
        plt.close(fig)


class TestQuickPlotResult:
    """Tests for quick_plot_result function."""

    def test_returns_figure_and_axes(self, sample_data):
        """Should return figure and axes tuple."""
        fig, ax = quick_plot_result(sample_data)

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_velocity_magnitude_default(self, sample_data):
        """Should plot velocity magnitude by default."""
        fig, ax = quick_plot_result(sample_data)

        assert ax.get_title() == "Velocity Magnitude"
        plt.close(fig)

    def test_vorticity_field(self, sample_data):
        """Should plot vorticity when specified."""
        fig, ax = quick_plot_result(sample_data, field="vorticity")

        assert ax.get_title() == "Vorticity"
        plt.close(fig)

    def test_pressure_field(self, sample_data):
        """Should plot pressure when available in result."""
        fig, ax = quick_plot_result(sample_data, field="p")

        assert ax.get_title() == "Pressure"
        plt.close(fig)

    def test_passes_kwargs(self, sample_data):
        """Should pass kwargs to quick_plot."""
        fig, ax = quick_plot_result(sample_data, title="Custom", levels=10)

        assert ax.get_title() == "Custom"
        plt.close(fig)

    def test_uses_domain_from_result(self):
        """Should use domain bounds from result dict."""
        result = {
            "u": [1.0] * 100,
            "v": [0.5] * 100,
            "nx": 10,
            "ny": 10,
            "xmin": -5.0,
            "xmax": 5.0,
            "ymin": -3.0,
            "ymax": 3.0,
        }
        fig, ax = quick_plot_result(result)

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        assert xlim[0] <= -5.0
        assert xlim[1] >= 5.0
        assert ylim[0] <= -3.0
        assert ylim[1] >= 3.0
        plt.close(fig)


class TestQuickPlotData:
    """Tests for quick_plot_data function."""

    def test_returns_figure_and_axes(self, sample_vtk_data):
        """Should return figure and axes tuple."""
        fig, ax = quick_plot_data(sample_vtk_data)

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_velocity_magnitude_default(self, sample_vtk_data):
        """Should plot velocity magnitude by default."""
        fig, ax = quick_plot_data(sample_vtk_data)

        assert ax.get_title() == "Velocity Magnitude"
        plt.close(fig)

    def test_vorticity_field(self, sample_vtk_data):
        """Should plot vorticity when specified."""
        fig, ax = quick_plot_data(sample_vtk_data, field="vorticity")

        assert ax.get_title() == "Vorticity"
        plt.close(fig)

    def test_pressure_field(self, sample_vtk_data):
        """Should plot pressure when available."""
        fig, ax = quick_plot_data(sample_vtk_data, field="p")

        assert ax.get_title() == "Pressure"
        plt.close(fig)

    def test_missing_u_raises(self):
        """Should raise ValueError when u field is missing."""
        data = VTKData(
            x=np.linspace(0, 1, 10),
            y=np.linspace(0, 1, 10),
            X=np.zeros((10, 10)),
            Y=np.zeros((10, 10)),
            fields={"v": np.ones((10, 10))},
            nx=10,
            ny=10,
            dx=0.1,
            dy=0.1,
        )

        with pytest.raises(ValueError, match="must have both u and v"):
            quick_plot_data(data)

    def test_missing_v_raises(self):
        """Should raise ValueError when v field is missing."""
        data = VTKData(
            x=np.linspace(0, 1, 10),
            y=np.linspace(0, 1, 10),
            X=np.zeros((10, 10)),
            Y=np.zeros((10, 10)),
            fields={"u": np.ones((10, 10))},
            nx=10,
            ny=10,
            dx=0.1,
            dy=0.1,
        )

        with pytest.raises(ValueError, match="must have both u and v"):
            quick_plot_data(data)

    def test_missing_pressure_raises(self, sample_vtk_data):
        """Should raise ValueError when pressure requested but not available."""
        data = from_cfd_python([1.0] * 100, [2.0] * 100, nx=10, ny=10)

        with pytest.raises(ValueError, match="Pressure field required"):
            quick_plot_data(data, field="p")

    def test_existing_axes(self, sample_vtk_data):
        """Should plot on existing axes when provided."""
        fig, existing_ax = plt.subplots()

        returned_fig, returned_ax = quick_plot_data(sample_vtk_data, ax=existing_ax)

        assert returned_ax is existing_ax
        assert returned_fig is fig
        plt.close(fig)


class TestModuleLevelExports:
    """Test that quick functions are available from cfd_viz package."""

    def test_quick_plot_exported(self):
        """quick_plot should be importable from cfd_viz."""
        from cfd_viz import quick_plot as fn

        assert callable(fn)

    def test_quick_plot_result_exported(self):
        """quick_plot_result should be importable from cfd_viz."""
        from cfd_viz import quick_plot_result as fn

        assert callable(fn)

    def test_quick_plot_data_exported(self):
        """quick_plot_data should be importable from cfd_viz."""
        from cfd_viz import quick_plot_data as fn

        assert callable(fn)
