"""Tests for cfd_viz.common.vtk_reader module."""

import numpy as np
import pytest

from cfd_viz.common.vtk_reader import (
    FIELD_ALIASES,
    VTKData,
    read_vtk_file,
    read_vtk_velocity,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_vtk_data(nx=4, ny=3, fields=None):
    """Create a minimal VTKData for testing."""
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    dx = x[1] - x[0] if nx > 1 else 1.0
    dy = y[1] - y[0] if ny > 1 else 1.0
    if fields is None:
        fields = {
            "u": np.ones((ny, nx)),
            "v": np.zeros((ny, nx)),
        }
    return VTKData(x=x, y=y, X=X, Y=Y, fields=fields, nx=nx, ny=ny, dx=dx, dy=dy)


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


# ---------------------------------------------------------------------------
# VTKData construction
# ---------------------------------------------------------------------------


class TestVTKDataConstruction:
    def test_basic_construction(self):
        data = _make_vtk_data()
        assert data.nx == 4
        assert data.ny == 3
        assert data.X.shape == (3, 4)

    def test_u_v_properties(self):
        data = _make_vtk_data()
        assert data.u is not None
        np.testing.assert_array_equal(data.u, np.ones((3, 4)))
        np.testing.assert_array_equal(data.v, np.zeros((3, 4)))

    def test_u_v_properties_missing(self):
        data = _make_vtk_data(fields={"p": np.zeros((3, 4))})
        assert data.u is None
        assert data.v is None

    def test_field_access_by_name(self):
        data = _make_vtk_data()
        np.testing.assert_array_equal(data["u"], np.ones((3, 4)))

    def test_field_access_by_alias(self):
        data = _make_vtk_data(fields={"p": np.ones((3, 4))})
        np.testing.assert_array_equal(data["pressure"], np.ones((3, 4)))

    def test_get_with_default(self):
        data = _make_vtk_data()
        assert data.get("nonexistent") is None
        assert data.get("nonexistent", 42) == 42

    def test_get_alias(self):
        data = _make_vtk_data(fields={"p": np.ones((3, 4))})
        result = data.get("pressure")
        assert result is not None
        np.testing.assert_array_equal(result, np.ones((3, 4)))

    def test_has_field(self):
        data = _make_vtk_data()
        assert data.has_field("u")
        assert not data.has_field("nonexistent")

    def test_has_field_alias(self):
        data = _make_vtk_data(fields={"p": np.zeros((3, 4))})
        assert data.has_field("pressure")

    def test_keys_returns_canonical_names(self):
        data = _make_vtk_data(fields={"u": np.zeros((3, 4)), "p": np.ones((3, 4))})
        assert set(data.keys()) == {"u", "p"}

    def test_to_dict(self):
        data = _make_vtk_data()
        d = data.to_dict()
        assert "u" in d
        assert "X" in d
        assert d["nx"] == 4

    def test_repr(self):
        data = _make_vtk_data()
        r = repr(data)
        assert "VTKData" in r
        assert "nx=4" in r
        assert "ny=3" in r
        assert "u" in r

    def test_invalid_field_shape_raises(self):
        with pytest.raises(ValueError, match="Field 'u' has shape"):
            _make_vtk_data(fields={"u": np.ones((5, 5))})

    def test_nan_warning(self):
        fields = {"u": np.array([[1, float("nan")], [3, 4]])}
        with pytest.warns(UserWarning, match="NaN"):
            _make_vtk_data(nx=2, ny=2, fields=fields)

    def test_inf_warning(self):
        fields = {"u": np.array([[1, float("inf")], [3, 4]])}
        with pytest.warns(UserWarning, match="infinite"):
            _make_vtk_data(nx=2, ny=2, fields=fields)

    def test_getitem_missing_raises_keyerror(self):
        data = _make_vtk_data()
        with pytest.raises(KeyError):
            data["nonexistent"]


# ---------------------------------------------------------------------------
# Reading STRUCTURED_POINTS
# ---------------------------------------------------------------------------


class TestReadVTKFileStructuredPoints:
    def test_read_real_sample_file(self, tmp_path):
        nx, ny = 50, 50
        u = np.ones((ny, nx))
        v = np.zeros((ny, nx))
        vtk_file = tmp_path / "flow_field_50x50_Re100.vtk"
        _write_structured_points(vtk_file, nx=nx, ny=ny, vectors=(u, v))
        data = read_vtk_file(str(vtk_file))
        assert data is not None
        assert data.nx == nx
        assert data.ny == ny
        assert data.u is not None
        assert data.v is not None
        assert data.u.shape == (ny, nx)

    def test_real_file_with_scalars(self, tmp_path):
        nx, ny = 10, 8
        u = np.ones((ny, nx))
        v = np.zeros((ny, nx))
        p = np.arange(nx * ny, dtype=float).reshape(ny, nx)
        vtk_file = tmp_path / "animated_flow_0050.vtk"
        _write_structured_points(
            vtk_file,
            nx=nx,
            ny=ny,
            vectors=(u, v),
            scalars={"pressure": p},
        )
        data = read_vtk_file(str(vtk_file))
        assert data is not None
        # "pressure" should be normalized to "p"
        assert "p" in data
        assert data["p"].shape == (data.ny, data.nx)

    def test_origin_and_spacing(self, tmp_path):
        nx, ny = 50, 50
        dx_expected = 0.020408
        u = np.ones((ny, nx))
        v = np.zeros((ny, nx))
        vtk_file = tmp_path / "flow_field_50x50_Re100.vtk"
        _write_structured_points(
            vtk_file,
            nx=nx,
            ny=ny,
            vectors=(u, v),
            origin=(0.0, 0.0, 0.0),
            spacing=(dx_expected, dx_expected, 1.0),
        )
        data = read_vtk_file(str(vtk_file))
        assert data.dx == pytest.approx(dx_expected, rel=1e-3)
        assert data.x[0] == pytest.approx(0.0)

    def test_synthetic_minimal(self, tmp_path):
        u = np.array([[1.0, 2.0], [3.0, 4.0]])
        v = np.array([[0.1, 0.2], [0.3, 0.4]])
        vtk_file = tmp_path / "test.vtk"
        _write_structured_points(vtk_file, nx=2, ny=2, vectors=(u, v))
        data = read_vtk_file(str(vtk_file))
        assert data is not None
        assert data.nx == 2
        assert data.ny == 2
        np.testing.assert_allclose(data.u, u)
        np.testing.assert_allclose(data.v, v)

    def test_scalars_with_lookup_table(self, tmp_path):
        u = np.ones((3, 3))
        v = np.zeros((3, 3))
        p = np.arange(9, dtype=float).reshape(3, 3)
        vtk_file = tmp_path / "test.vtk"
        _write_structured_points(
            vtk_file,
            nx=3,
            ny=3,
            vectors=(u, v),
            scalars={"pressure": p},
        )
        data = read_vtk_file(str(vtk_file))
        assert data is not None
        # "pressure" normalized to "p"
        assert "p" in data
        np.testing.assert_allclose(data["p"], p)

    def test_custom_origin_and_spacing(self, tmp_path):
        u = np.ones((2, 3))
        v = np.zeros((2, 3))
        vtk_file = tmp_path / "test.vtk"
        _write_structured_points(
            vtk_file,
            nx=3,
            ny=2,
            vectors=(u, v),
            origin=(1.0, 2.0, 0.0),
            spacing=(0.5, 0.25, 1.0),
        )
        data = read_vtk_file(str(vtk_file))
        assert data.x[0] == pytest.approx(1.0)
        assert data.x[-1] == pytest.approx(2.0)
        assert data.y[0] == pytest.approx(2.0)
        assert data.dx == pytest.approx(0.5)
        assert data.dy == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Reading RECTILINEAR_GRID
# ---------------------------------------------------------------------------


class TestReadVTKFileRectilinearGrid:
    def test_synthetic_rectilinear(self, tmp_path):
        nx, ny = 3, 3
        content = (
            "# vtk DataFile Version 3.0\n"
            "Rectilinear Test\n"
            "ASCII\n"
            "DATASET RECTILINEAR_GRID\n"
            f"DIMENSIONS {nx} {ny} 1\n"
            "X_COORDINATES 3 float\n"
            "0.0 0.5 1.0\n"
            "Y_COORDINATES 3 float\n"
            "0.0 0.5 1.0\n"
            "Z_COORDINATES 1 float\n"
            "0.0\n"
            "\n"
            f"POINT_DATA {nx * ny}\n"
            "VECTORS velocity float\n"
        )
        u_vals = np.arange(1, 10, dtype=float)
        v_vals = np.zeros(9)
        vec_lines = "\n".join(f"{u} {v} 0.0" for u, v in zip(u_vals, v_vals))
        vtk_file = tmp_path / "rect.vtk"
        vtk_file.write_text(content + vec_lines + "\n")

        data = read_vtk_file(str(vtk_file))
        assert data is not None
        assert data.nx == 3
        assert data.ny == 3
        assert data.x[0] == pytest.approx(0.0)
        assert data.x[-1] == pytest.approx(1.0)
        np.testing.assert_allclose(data.u.ravel(), u_vals)

    def test_non_uniform_spacing(self, tmp_path):
        nx, ny = 3, 2
        content = (
            "# vtk DataFile Version 3.0\n"
            "Non-uniform\n"
            "ASCII\n"
            "DATASET RECTILINEAR_GRID\n"
            f"DIMENSIONS {nx} {ny} 1\n"
            "X_COORDINATES 3 float\n"
            "0.0 0.3 1.0\n"
            "Y_COORDINATES 2 float\n"
            "0.0 1.0\n"
            "Z_COORDINATES 1 float\n"
            "0.0\n"
            "\n"
            f"POINT_DATA {nx * ny}\n"
            "VECTORS velocity float\n"
        )
        vec_lines = "\n".join("1.0 0.0 0.0" for _ in range(nx * ny))
        vtk_file = tmp_path / "nonuniform.vtk"
        vtk_file.write_text(content + vec_lines + "\n")

        data = read_vtk_file(str(vtk_file))
        assert data is not None
        # Non-uniform: dx is computed from first two coordinates
        assert data.x[1] == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Malformed files
# ---------------------------------------------------------------------------


class TestReadVTKFileMalformed:
    def test_missing_dimensions(self, tmp_path):
        content = (
            "# vtk DataFile Version 3.0\n"
            "No dims\n"
            "ASCII\n"
            "DATASET STRUCTURED_POINTS\n"
            "POINT_DATA 4\n"
            "VECTORS velocity float\n"
            "1.0 0.0 0.0\n"
            "1.0 0.0 0.0\n"
            "1.0 0.0 0.0\n"
            "1.0 0.0 0.0\n"
        )
        vtk_file = tmp_path / "nodims.vtk"
        vtk_file.write_text(content)
        with pytest.raises(ValueError, match="dimensions not found"):
            read_vtk_file(str(vtk_file))

    def test_file_not_found(self):
        result = read_vtk_file("/nonexistent/path/to/file.vtk")
        assert result is None

    def test_empty_file(self, tmp_path):
        vtk_file = tmp_path / "empty.vtk"
        vtk_file.write_text("")
        with pytest.raises(ValueError, match="dimensions not found"):
            read_vtk_file(str(vtk_file))

    def test_truncated_vector_data(self, tmp_path):
        content = (
            "# vtk DataFile Version 3.0\n"
            "Truncated\n"
            "ASCII\n"
            "DATASET STRUCTURED_POINTS\n"
            "DIMENSIONS 3 3 1\n"
            "ORIGIN 0 0 0\n"
            "SPACING 1 1 1\n"
            "\n"
            "POINT_DATA 9\n"
            "VECTORS velocity float\n"
            "1.0 0.0 0.0\n"
            "1.0 0.0 0.0\n"
        )
        vtk_file = tmp_path / "truncated.vtk"
        vtk_file.write_text(content)
        # Truncated data causes reshape error
        with pytest.raises(ValueError, match="cannot reshape"):
            read_vtk_file(str(vtk_file))


# ---------------------------------------------------------------------------
# read_vtk_velocity convenience function
# ---------------------------------------------------------------------------


class TestReadVTKVelocity:
    def test_returns_tuple(self, tmp_path):
        nx, ny = 50, 50
        u_in = np.ones((ny, nx))
        v_in = np.zeros((ny, nx))
        vtk_file = tmp_path / "flow_field.vtk"
        _write_structured_points(vtk_file, nx=nx, ny=ny, vectors=(u_in, v_in))
        result = read_vtk_velocity(str(vtk_file))
        X, Y, u, v = result
        assert X is not None
        assert X.shape == (ny, nx)
        assert u.shape == (ny, nx)

    def test_file_not_found_returns_nones(self):
        result = read_vtk_velocity("/nonexistent/file.vtk")
        assert all(r is None for r in result)


# ---------------------------------------------------------------------------
# Field aliases
# ---------------------------------------------------------------------------


class TestFieldAliases:
    def test_pressure_alias_maps_to_p(self):
        assert FIELD_ALIASES["pressure"] == "p"

    def test_velocity_aliases(self):
        assert FIELD_ALIASES["velocity_x"] == "u"
        assert FIELD_ALIASES["velocity_y"] == "v"

    def test_canonical_names_not_in_aliases_as_values(self):
        # Canonical names should not map to themselves
        assert "u" not in FIELD_ALIASES
        assert "v" not in FIELD_ALIASES
        assert "p" not in FIELD_ALIASES

    def test_canonical_name_normalization_at_read_time(self, tmp_path):
        """SCALARS 'pressure' should be stored as 'p' in fields."""
        u = np.ones((2, 2))
        v = np.zeros((2, 2))
        p = np.array([[1.0, 2.0], [3.0, 4.0]])
        vtk_file = tmp_path / "test.vtk"
        _write_structured_points(
            vtk_file,
            nx=2,
            ny=2,
            vectors=(u, v),
            scalars={"pressure": p},
        )
        data = read_vtk_file(str(vtk_file))
        # Stored under canonical name "p", not "pressure"
        assert "p" in data.fields
        assert "pressure" not in data.fields
        # But accessible via alias through __contains__ and __getitem__
        assert "pressure" in data
        np.testing.assert_allclose(data["pressure"], p)

    def test_unknown_scalar_name_preserved(self, tmp_path):
        """SCALARS with unknown names are stored as-is."""
        u = np.ones((2, 2))
        v = np.zeros((2, 2))
        temp = np.array([[300.0, 301.0], [302.0, 303.0]])
        vtk_file = tmp_path / "test.vtk"
        _write_structured_points(
            vtk_file,
            nx=2,
            ny=2,
            vectors=(u, v),
            scalars={"temperature": temp},
        )
        data = read_vtk_file(str(vtk_file))
        assert "temperature" in data
        np.testing.assert_allclose(data["temperature"], temp)
