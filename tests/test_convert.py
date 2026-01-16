"""Tests for cfd_viz.convert module."""

import numpy as np
import pytest

from cfd_viz.convert import from_cfd_python, from_simulation_result, to_cfd_python


class TestFromCfdPython:
    """Tests for from_cfd_python function."""

    def test_basic_conversion(self):
        """Should create VTKData from flat lists."""
        u = [1.0] * 100
        v = [0.5] * 100
        data = from_cfd_python(u, v, nx=10, ny=10)

        assert data.nx == 10
        assert data.ny == 10
        assert data.u.shape == (10, 10)
        assert data.v.shape == (10, 10)

    def test_with_pressure(self):
        """Should include pressure field when provided."""
        u = [1.0] * 100
        v = [0.5] * 100
        p = [101325.0] * 100
        data = from_cfd_python(u, v, nx=10, ny=10, p=p)

        assert data.get("p") is not None
        assert data.get("p").shape == (10, 10)

    def test_custom_domain(self):
        """Should respect custom domain bounds."""
        u = [1.0] * 25
        v = [0.5] * 25
        data = from_cfd_python(
            u, v, nx=5, ny=5, xmin=-1.0, xmax=1.0, ymin=-1.0, ymax=1.0
        )

        assert data.x[0] == pytest.approx(-1.0)
        assert data.x[-1] == pytest.approx(1.0)
        assert data.y[0] == pytest.approx(-1.0)
        assert data.y[-1] == pytest.approx(1.0)

    def test_dx_dy_calculated(self):
        """Should calculate correct grid spacing."""
        u = [1.0] * 100
        v = [0.5] * 100
        data = from_cfd_python(u, v, nx=10, ny=10, xmin=0.0, xmax=1.0, ymin=0.0, ymax=2.0)

        assert data.dx == pytest.approx(1.0 / 9.0)  # (1.0 - 0.0) / (10 - 1)
        assert data.dy == pytest.approx(2.0 / 9.0)  # (2.0 - 0.0) / (10 - 1)

    def test_meshgrid_created(self):
        """Should create X, Y meshgrid arrays."""
        u = [1.0] * 25
        v = [0.5] * 25
        data = from_cfd_python(u, v, nx=5, ny=5)

        assert data.X.shape == (5, 5)
        assert data.Y.shape == (5, 5)

    def test_invalid_grid_dimensions_zero(self):
        """Should raise ValueError for zero grid dimensions."""
        with pytest.raises(ValueError, match="Invalid grid dimensions"):
            from_cfd_python([1.0], [1.0], nx=0, ny=10)

    def test_invalid_grid_dimensions_negative(self):
        """Should raise ValueError for negative grid dimensions."""
        with pytest.raises(ValueError, match="Invalid grid dimensions"):
            from_cfd_python([1.0], [1.0], nx=-5, ny=10)

    def test_invalid_x_bounds_inverted(self):
        """Should raise ValueError when xmin >= xmax."""
        with pytest.raises(ValueError, match="Invalid x bounds"):
            from_cfd_python([1.0] * 100, [1.0] * 100, nx=10, ny=10, xmin=1.0, xmax=0.0)

    def test_invalid_x_bounds_equal(self):
        """Should raise ValueError when xmin == xmax."""
        with pytest.raises(ValueError, match="Invalid x bounds"):
            from_cfd_python([1.0] * 100, [1.0] * 100, nx=10, ny=10, xmin=0.5, xmax=0.5)

    def test_invalid_y_bounds_inverted(self):
        """Should raise ValueError when ymin >= ymax."""
        with pytest.raises(ValueError, match="Invalid y bounds"):
            from_cfd_python([1.0] * 100, [1.0] * 100, nx=10, ny=10, ymin=1.0, ymax=0.0)

    def test_invalid_y_bounds_equal(self):
        """Should raise ValueError when ymin == ymax."""
        with pytest.raises(ValueError, match="Invalid y bounds"):
            from_cfd_python([1.0] * 100, [1.0] * 100, nx=10, ny=10, ymin=0.5, ymax=0.5)

    def test_u_size_mismatch(self):
        """Should raise ValueError when u size doesn't match grid."""
        with pytest.raises(ValueError, match="u has .* elements, expected"):
            from_cfd_python([1.0] * 50, [1.0] * 100, nx=10, ny=10)

    def test_v_size_mismatch(self):
        """Should raise ValueError when v size doesn't match grid."""
        with pytest.raises(ValueError, match="v has .* elements, expected"):
            from_cfd_python([1.0] * 100, [1.0] * 50, nx=10, ny=10)

    def test_p_size_mismatch(self):
        """Should raise ValueError when p size doesn't match grid."""
        with pytest.raises(ValueError, match="p has .* elements, expected"):
            from_cfd_python([1.0] * 100, [1.0] * 100, nx=10, ny=10, p=[1.0] * 50)

    def test_single_cell_grid(self):
        """Should handle 1x1 grid edge case."""
        data = from_cfd_python([1.0], [0.5], nx=1, ny=1)

        assert data.nx == 1
        assert data.ny == 1
        assert data.dx == 1.0  # Default when nx=1
        assert data.dy == 1.0  # Default when ny=1

    def test_values_preserved(self):
        """Should preserve input values in correct positions."""
        u = list(range(12))
        v = list(range(12, 24))
        data = from_cfd_python(u, v, nx=4, ny=3)

        # Check that values are reshaped correctly (row-major order)
        assert data.u[0, 0] == 0
        assert data.u[0, 3] == 3
        assert data.u[2, 3] == 11
        assert data.v[0, 0] == 12
        assert data.v[2, 3] == 23


class TestFromSimulationResult:
    """Tests for from_simulation_result function."""

    def test_basic_conversion(self):
        """Should convert simulation result dict to VTKData."""
        result = {
            "u": [1.0] * 100,
            "v": [0.5] * 100,
            "nx": 10,
            "ny": 10,
        }
        data = from_simulation_result(result)

        assert data.nx == 10
        assert data.ny == 10

    def test_with_optional_fields(self):
        """Should handle optional fields in result dict."""
        result = {
            "u": [1.0] * 100,
            "v": [0.5] * 100,
            "p": [101325.0] * 100,
            "nx": 10,
            "ny": 10,
            "xmin": -1.0,
            "xmax": 1.0,
            "ymin": -1.0,
            "ymax": 1.0,
        }
        data = from_simulation_result(result)

        assert data.get("p") is not None
        assert data.x[0] == pytest.approx(-1.0)
        assert data.y[-1] == pytest.approx(1.0)

    def test_default_domain_bounds(self):
        """Should use default bounds [0,1] when not specified."""
        result = {
            "u": [1.0] * 100,
            "v": [0.5] * 100,
            "nx": 10,
            "ny": 10,
        }
        data = from_simulation_result(result)

        assert data.x[0] == pytest.approx(0.0)
        assert data.x[-1] == pytest.approx(1.0)
        assert data.y[0] == pytest.approx(0.0)
        assert data.y[-1] == pytest.approx(1.0)


class TestToCfdPython:
    """Tests for to_cfd_python function."""

    def test_basic_conversion(self):
        """Should convert VTKData to cfd_python dict format."""
        u = [1.0] * 100
        v = [0.5] * 100
        data = from_cfd_python(u, v, nx=10, ny=10)

        result = to_cfd_python(data)

        assert result["nx"] == 10
        assert result["ny"] == 10
        assert isinstance(result["u"], list)
        assert isinstance(result["v"], list)
        assert len(result["u"]) == 100
        assert len(result["v"]) == 100

    def test_domain_bounds_extracted(self):
        """Should extract domain bounds from VTKData."""
        data = from_cfd_python(
            [1.0] * 25,
            [0.5] * 25,
            nx=5,
            ny=5,
            xmin=-2.0,
            xmax=2.0,
            ymin=-1.0,
            ymax=1.0,
        )

        result = to_cfd_python(data)

        assert result["xmin"] == pytest.approx(-2.0)
        assert result["xmax"] == pytest.approx(2.0)
        assert result["ymin"] == pytest.approx(-1.0)
        assert result["ymax"] == pytest.approx(1.0)

    def test_pressure_included_when_present(self):
        """Should include pressure when available."""
        data = from_cfd_python(
            [1.0] * 100, [0.5] * 100, nx=10, ny=10, p=[101325.0] * 100
        )

        result = to_cfd_python(data)

        assert result["p"] is not None
        assert len(result["p"]) == 100

    def test_pressure_none_when_absent(self):
        """Should return None for pressure when not available."""
        data = from_cfd_python([1.0] * 100, [0.5] * 100, nx=10, ny=10)

        result = to_cfd_python(data)

        assert result["p"] is None


class TestRoundtrip:
    """Tests for roundtrip conversion."""

    def test_roundtrip_preserves_data(self):
        """Should preserve data through roundtrip conversion."""
        original_u = list(range(100))
        original_v = list(range(100, 200))

        data = from_cfd_python(original_u, original_v, nx=10, ny=10)
        result = to_cfd_python(data)

        assert result["nx"] == 10
        assert result["ny"] == 10
        assert result["u"] == original_u
        assert result["v"] == original_v

    def test_roundtrip_with_pressure(self):
        """Should preserve pressure through roundtrip."""
        original_p = list(range(100))

        data = from_cfd_python(
            [1.0] * 100, [0.5] * 100, nx=10, ny=10, p=original_p
        )
        result = to_cfd_python(data)

        assert result["p"] == original_p

    def test_roundtrip_with_custom_domain(self):
        """Should preserve domain bounds through roundtrip."""
        data = from_cfd_python(
            [1.0] * 100,
            [0.5] * 100,
            nx=10,
            ny=10,
            xmin=-5.0,
            xmax=5.0,
            ymin=-2.5,
            ymax=2.5,
        )
        result = to_cfd_python(data)

        assert result["xmin"] == pytest.approx(-5.0)
        assert result["xmax"] == pytest.approx(5.0)
        assert result["ymin"] == pytest.approx(-2.5)
        assert result["ymax"] == pytest.approx(2.5)


class TestModuleLevelExports:
    """Test that conversion functions are available from cfd_viz package."""

    def test_from_cfd_python_exported(self):
        """from_cfd_python should be importable from cfd_viz."""
        from cfd_viz import from_cfd_python as fn

        assert callable(fn)

    def test_from_simulation_result_exported(self):
        """from_simulation_result should be importable from cfd_viz."""
        from cfd_viz import from_simulation_result as fn

        assert callable(fn)

    def test_to_cfd_python_exported(self):
        """to_cfd_python should be importable from cfd_viz."""
        from cfd_viz import to_cfd_python as fn

        assert callable(fn)
