"""Tests for cfd_viz.defaults module."""

import pytest

from cfd_viz.defaults import (
    UNSET,
    PlotDefaults,
    _UnsetType,
    get_defaults,
    load_config_file,
    plot_context,
    reset_defaults,
    resolve,
    set_defaults,
)


@pytest.fixture(autouse=True)
def _clean_defaults():
    """Reset global defaults after every test."""
    yield
    reset_defaults()


class TestPlotDefaults:
    """Tests for the PlotDefaults dataclass."""

    def test_default_values(self):
        d = PlotDefaults()
        assert d.cmap == "viridis"
        assert d.diverging_cmap == "RdBu_r"
        assert d.sequential_cmap == "plasma"
        assert d.figsize == (10, 8)
        assert d.dpi == 150
        assert d.levels == 20
        assert d.font_size == 12
        assert d.colorscale == "Viridis"
        assert d.diverging_colorscale == "RdBu"

    def test_custom_values(self):
        d = PlotDefaults(cmap="coolwarm", dpi=300)
        assert d.cmap == "coolwarm"
        assert d.dpi == 300
        assert d.levels == 20  # unchanged


class TestGetDefaults:
    """Tests for get_defaults()."""

    def test_returns_plot_defaults_instance(self):
        assert isinstance(get_defaults(), PlotDefaults)

    def test_returns_copy(self):
        d = get_defaults()
        d.cmap = "modified"
        assert get_defaults().cmap == "viridis"


class TestSetDefaults:
    """Tests for set_defaults()."""

    def test_updates_values(self):
        set_defaults(cmap="coolwarm")
        assert get_defaults().cmap == "coolwarm"

    def test_partial_update(self):
        set_defaults(cmap="coolwarm")
        assert get_defaults().dpi == 150  # unchanged
        assert get_defaults().levels == 20  # unchanged

    def test_multiple_fields(self):
        set_defaults(cmap="hot", dpi=300, levels=30)
        d = get_defaults()
        assert d.cmap == "hot"
        assert d.dpi == 300
        assert d.levels == 30

    def test_rejects_unknown_fields(self):
        with pytest.raises(TypeError, match="Unknown defaults.*bogus"):
            set_defaults(bogus="x")

    def test_rejects_multiple_unknown_fields(self):
        with pytest.raises(TypeError, match="Unknown defaults"):
            set_defaults(bogus="x", also_bogus="y")


class TestResetDefaults:
    """Tests for reset_defaults()."""

    def test_restores_originals(self):
        set_defaults(cmap="hot", dpi=300)
        reset_defaults()
        d = get_defaults()
        assert d.cmap == "viridis"
        assert d.dpi == 150


class TestResolve:
    """Tests for the resolve() helper."""

    def test_returns_explicit_value(self):
        assert resolve("coolwarm", "cmap") == "coolwarm"

    def test_returns_default_for_unset(self):
        assert resolve(UNSET, "cmap") == "viridis"

    def test_respects_set_defaults(self):
        set_defaults(cmap="hot")
        assert resolve(UNSET, "cmap") == "hot"

    def test_explicit_overrides_set_defaults(self):
        set_defaults(cmap="hot")
        assert resolve("coolwarm", "cmap") == "coolwarm"

    def test_resolves_different_fields(self):
        assert resolve(UNSET, "dpi") == 150
        assert resolve(UNSET, "levels") == 20
        assert resolve(UNSET, "figsize") == (10, 8)


class TestUnset:
    """Tests for the UNSET sentinel."""

    def test_is_singleton(self):
        assert _UnsetType() is UNSET

    def test_is_falsy(self):
        assert not UNSET

    def test_repr(self):
        assert repr(UNSET) == "UNSET"


class TestPlotContext:
    """Tests for the plot_context() context manager."""

    def test_overrides_inside_context(self):
        with plot_context(cmap="hot"):
            assert get_defaults().cmap == "hot"

    def test_restores_on_exit(self):
        set_defaults(cmap="coolwarm")
        with plot_context(cmap="hot"):
            pass
        assert get_defaults().cmap == "coolwarm"

    def test_restores_on_exception(self):
        set_defaults(cmap="coolwarm")
        with pytest.raises(ValueError), plot_context(cmap="hot"):
            raise ValueError("boom")
        assert get_defaults().cmap == "coolwarm"

    def test_nested_contexts(self):
        assert get_defaults().cmap == "viridis"
        with plot_context(cmap="hot"):
            assert get_defaults().cmap == "hot"
            with plot_context(cmap="coolwarm"):
                assert get_defaults().cmap == "coolwarm"
            assert get_defaults().cmap == "hot"
        assert get_defaults().cmap == "viridis"

    def test_yields_defaults(self):
        with plot_context(cmap="hot", dpi=300) as d:
            assert d.cmap == "hot"
            assert d.dpi == 300

    def test_rejects_unknown_fields(self):
        with (
            pytest.raises(TypeError, match="Unknown defaults"),
            plot_context(bogus="x"),
        ):
            pass


class TestLoadConfigFile:
    """Tests for load_config_file()."""

    def test_nonexistent_returns_false(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert load_config_file() is False

    def test_load_cfd_viz_toml(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "cfd_viz.toml").write_text(
            '[defaults]\ncmap = "coolwarm"\ndpi = 200\n'
        )
        assert load_config_file() is True
        d = get_defaults()
        assert d.cmap == "coolwarm"
        assert d.dpi == 200

    def test_load_pyproject_toml(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "pyproject.toml").write_text(
            '[tool.cfd_viz.defaults]\ncmap = "hot"\nlevels = 30\n'
        )
        assert load_config_file() is True
        d = get_defaults()
        assert d.cmap == "hot"
        assert d.levels == 30

    def test_cfd_viz_toml_takes_priority(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "cfd_viz.toml").write_text('[defaults]\ncmap = "coolwarm"\n')
        (tmp_path / "pyproject.toml").write_text(
            '[tool.cfd_viz.defaults]\ncmap = "hot"\n'
        )
        load_config_file()
        assert get_defaults().cmap == "coolwarm"

    def test_figsize_tuple_conversion(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "cfd_viz.toml").write_text("[defaults]\nfigsize = [12, 8]\n")
        load_config_file()
        assert get_defaults().figsize == (12, 8)

    def test_explicit_path(self, tmp_path):
        config = tmp_path / "custom.toml"
        config.write_text('cmap = "magma"\ndpi = 72\n')
        assert load_config_file(str(config)) is True
        d = get_defaults()
        assert d.cmap == "magma"
        assert d.dpi == 72

    def test_pyproject_without_section_returns_false(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "pyproject.toml").write_text("[tool.other]\nfoo = 1\n")
        assert load_config_file() is False
