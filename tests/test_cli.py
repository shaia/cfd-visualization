"""Tests for the unified cfd-viz CLI."""

import os
import textwrap
from unittest import mock

import pytest

from cfd_viz.cli import _build_parser, main


class TestParserStructure:
    """Verify parser accepts expected arguments without executing anything."""

    def test_no_args_exits_zero(self):
        with pytest.raises(SystemExit) as exc:
            main([])
        assert exc.value.code == 0

    def test_help_exits(self):
        with pytest.raises(SystemExit) as exc:
            main(["--help"])
        assert exc.value.code == 0

    def test_info_subcommand_parsed(self):
        parser = _build_parser()
        args = parser.parse_args(["info"])
        assert args.command == "info"

    def test_animate_subcommand_parsed(self):
        parser = _build_parser()
        args = parser.parse_args(["animate", "a.vtk", "--type", "velocity"])
        assert args.command == "animate"
        assert args.vtk_files == ["a.vtk"]
        assert args.type == "velocity"

    def test_animate_defaults(self):
        parser = _build_parser()
        args = parser.parse_args(["animate", "a.vtk"])
        assert args.type == "velocity"
        assert args.field == "velocity_mag"
        assert args.fps == 5
        assert args.output is None
        assert args.export_frames is False

    def test_animate_all_options(self):
        parser = _build_parser()
        args = parser.parse_args(
            [
                "animate",
                "a.vtk",
                "b.vtk",
                "--type",
                "streamlines",
                "--field",
                "pressure",
                "--output",
                "out.gif",
                "--fps",
                "10",
                "--export-frames",
                "--all",
            ]
        )
        assert args.vtk_files == ["a.vtk", "b.vtk"]
        assert args.type == "streamlines"
        assert args.field == "pressure"
        assert args.output == "out.gif"
        assert args.fps == 10
        assert args.export_frames is True

    def test_dashboard_subcommand_parsed(self):
        parser = _build_parser()
        args = parser.parse_args(
            [
                "dashboard",
                "--vtk-pattern",
                "data/*.vtk",
                "--auto-open",
            ]
        )
        assert args.command == "dashboard"
        assert args.vtk_pattern == "data/*.vtk"
        assert args.auto_open is True

    def test_dashboard_defaults(self):
        parser = _build_parser()
        args = parser.parse_args(["dashboard"])
        assert args.vtk_pattern == "output/output_optimized_*.vtk"
        assert args.output_dir == "visualization_output"
        assert args.auto_open is False

    def test_vorticity_subcommand_parsed(self):
        parser = _build_parser()
        args = parser.parse_args(["vorticity", "flow.vtk", "-o", "out"])
        assert args.command == "vorticity"
        assert args.input_file == "flow.vtk"
        assert args.output == "out"

    def test_vorticity_latest_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["vorticity", "--latest"])
        assert args.latest is True
        assert args.input_file is None

    def test_profiles_subcommand_parsed(self):
        parser = _build_parser()
        args = parser.parse_args(["profiles", "f.vtk", "--interactive"])
        assert args.command == "profiles"
        assert args.input_file == "f.vtk"
        assert args.interactive is True

    def test_monitor_subcommand_parsed(self):
        parser = _build_parser()
        args = parser.parse_args(
            [
                "monitor",
                "--watch-dir",
                "/tmp/data",
                "--interval",
                "5",
            ]
        )
        assert args.command == "monitor"
        assert args.watch_dir == "/tmp/data"
        assert args.interval == 5.0

    def test_monitor_defaults(self):
        parser = _build_parser()
        args = parser.parse_args(["monitor"])
        assert args.watch_dir is None
        assert args.output is None
        assert args.interval == 2.0
        assert args.manual is False

    def test_batch_requires_config(self):
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["batch"])

    def test_batch_subcommand_parsed(self):
        parser = _build_parser()
        args = parser.parse_args(["batch", "--config", "batch.toml"])
        assert args.command == "batch"
        assert args.config == "batch.toml"


class TestInfoCommand:
    """Test that ``cfd-viz info`` calls print_system_info."""

    def test_info_dispatches(self):
        with mock.patch("cfd_viz.info.print_system_info") as mock_print:
            main(["info"])
            mock_print.assert_called_once()


class TestBatchProcessing:
    """Test batch config loading and execution."""

    def test_empty_batch(self, tmp_path):
        config = tmp_path / "batch.toml"
        config.write_text("[batch]\njobs = []\n")

        from cfd_viz._batch import run_batch

        # Should print "No jobs" and return without error
        run_batch(str(config))

    def test_batch_missing_vtk(self, tmp_path):
        config = tmp_path / "batch.toml"
        config.write_text(
            textwrap.dedent("""\
            [batch]
            output_dir = "{out}"

            [[batch.jobs]]
            vtk = "nonexistent.vtk"
            analyses = ["vorticity"]
        """).format(out=str(tmp_path / "out").replace("\\", "/"))
        )

        from cfd_viz._batch import run_batch

        # Should handle missing file gracefully (warning, not crash)
        run_batch(str(config))

    def test_batch_unknown_analysis(self, tmp_path, capsys):
        config = tmp_path / "batch.toml"
        vtk_file = tmp_path / "dummy.vtk"
        vtk_file.write_text("")  # empty file

        config.write_text(
            textwrap.dedent("""\
            [batch]
            output_dir = "{out}"

            [[batch.jobs]]
            vtk = "{vtk}"
            analyses = ["unknown_analysis"]
        """).format(
                out=str(tmp_path / "out").replace("\\", "/"),
                vtk=str(vtk_file).replace("\\", "/"),
            )
        )

        from cfd_viz._batch import run_batch

        run_batch(str(config))
        captured = capsys.readouterr()
        assert (
            "unknown analysis" in captured.out.lower()
            or "unknown_analysis" in captured.out
        )

    def test_batch_vtk_pattern(self, tmp_path):
        config = tmp_path / "batch.toml"
        config.write_text(
            textwrap.dedent("""\
            [batch]
            output_dir = "{out}"

            [[batch.jobs]]
            vtk_pattern = "{pattern}"
            analyses = ["vorticity"]
        """).format(
                out=str(tmp_path / "out").replace("\\", "/"),
                pattern=str(tmp_path / "*.vtk").replace("\\", "/"),
            )
        )

        from cfd_viz._batch import run_batch

        # No matching files — should skip gracefully
        run_batch(str(config))


class TestProgressIndicator:
    """Test the progress bar helper."""

    def test_progress_writes_to_stderr(self, capsys):
        from cfd_viz._batch import _progress

        _progress(1, 2, "test")
        captured = capsys.readouterr()
        assert "test" in captured.err
        assert "%" in captured.err

    def test_progress_complete(self, capsys):
        from cfd_viz._batch import _progress

        _progress(5, 5, "done")
        captured = capsys.readouterr()
        assert "100" in captured.err


class TestEntryPoint:
    """Verify the entry point registration in pyproject.toml."""

    def test_entry_point_in_pyproject(self):
        # Read pyproject.toml to verify entry point is registered
        try:
            import tomllib
        except ModuleNotFoundError:
            import tomli as tomllib

        pyproject_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "pyproject.toml"
        )
        with open(pyproject_path, "rb") as f:
            config = tomllib.load(f)

        scripts = config.get("project", {}).get("scripts", {})
        assert "cfd-viz" in scripts
        assert scripts["cfd-viz"] == "cfd_viz.cli:main"
