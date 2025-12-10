"""Tests for cfd_viz.animation.export module."""

import os
import tempfile

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

matplotlib.use("Agg")  # Non-interactive backend for testing

from cfd_viz.animation.export import (
    create_comprehensive_frame_figure,
    export_animation_frames,
    export_frame_to_image,
    save_animation,
)
from cfd_viz.animation.frames import create_animation_frames, create_frame_data
from cfd_viz.animation.renderers import create_field_animation


@pytest.fixture
def sample_grid():
    """Create a sample grid for testing."""
    x = np.linspace(0, 1, 30)
    y = np.linspace(0, 0.5, 15)
    X, Y = np.meshgrid(x, y)
    return X, Y


@pytest.fixture
def uniform_velocity(sample_grid):
    """Create uniform velocity field."""
    X, Y = sample_grid
    u = np.ones_like(X) * 1.0
    v = np.zeros_like(X)
    return u, v


@pytest.fixture
def sample_pressure(sample_grid):
    """Create sample pressure field."""
    X, Y = sample_grid
    p = X + Y
    return p


@pytest.fixture
def sample_frame(sample_grid, uniform_velocity, sample_pressure):
    """Create a sample frame for testing."""
    X, Y = sample_grid
    u, v = uniform_velocity
    p = sample_pressure
    return create_frame_data(X, Y, u, v, p=p, time_index=0)


@pytest.fixture
def sample_animation_frames(sample_grid, uniform_velocity, sample_pressure):
    """Create sample animation frames."""
    X, Y = sample_grid
    u, v = uniform_velocity
    p = sample_pressure

    frames_list = [
        (X, Y, u, v, p),
        (X, Y, u * 1.1, v, p * 1.1),
        (X, Y, u * 1.2, v, p * 1.2),
    ]
    return create_animation_frames(frames_list)


class TestExportFrameToImage:
    """Tests for export_frame_to_image function."""

    def test_creates_file(self, sample_frame):
        """Should create an image file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_frame.png")
            export_frame_to_image(sample_frame, output_path)

            assert os.path.exists(output_path)

    def test_respects_dpi(self, sample_frame):
        """Should use specified DPI."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_frame.png")
            export_frame_to_image(sample_frame, output_path, dpi=72)

            assert os.path.exists(output_path)

    def test_without_vectors(self, sample_frame):
        """Should work without vector subplot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_frame.png")
            export_frame_to_image(sample_frame, output_path, include_vectors=False)

            assert os.path.exists(output_path)

    def test_without_streamlines(self, sample_frame):
        """Should work without streamlines subplot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_frame.png")
            export_frame_to_image(sample_frame, output_path, include_streamlines=False)

            assert os.path.exists(output_path)

    def test_without_pressure(self, sample_grid, uniform_velocity):
        """Should work without pressure data."""
        X, Y = sample_grid
        u, v = uniform_velocity
        frame = create_frame_data(X, Y, u, v, time_index=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_frame.png")
            export_frame_to_image(frame, output_path, include_pressure=True)

            assert os.path.exists(output_path)


class TestExportAnimationFrames:
    """Tests for export_animation_frames function."""

    def test_creates_all_files(self, sample_animation_frames):
        """Should create image file for each frame."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exported = export_animation_frames(
                sample_animation_frames, tmpdir, prefix="test"
            )

            assert len(exported) == len(sample_animation_frames)
            for path in exported:
                assert os.path.exists(path)

    def test_uses_prefix(self, sample_animation_frames):
        """Should use specified prefix."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exported = export_animation_frames(
                sample_animation_frames, tmpdir, prefix="myprefix"
            )

            for path in exported:
                assert "myprefix" in os.path.basename(path)

    def test_creates_output_dir(self, sample_animation_frames):
        """Should create output directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "subdir", "frames")
            exported = export_animation_frames(sample_animation_frames, output_dir)

            assert os.path.exists(output_dir)
            assert len(exported) > 0


class TestSaveAnimation:
    """Tests for save_animation function."""

    def test_saves_gif(self, sample_animation_frames):
        """Should save animation as GIF."""
        fig, anim = create_field_animation(sample_animation_frames, "velocity_mag")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.gif")
            save_animation(anim, output_path, writer="pillow", fps=2)

            assert os.path.exists(output_path)

        plt.close(fig)

    def test_creates_parent_dirs(self, sample_animation_frames):
        """Should create parent directories."""
        fig, anim = create_field_animation(sample_animation_frames, "velocity_mag")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "subdir", "nested", "test.gif")
            save_animation(anim, output_path, writer="pillow", fps=2)

            assert os.path.exists(output_path)

        plt.close(fig)


class TestCreateComprehensiveFrameFigure:
    """Tests for create_comprehensive_frame_figure function."""

    def test_returns_figure(self, sample_frame):
        """Should return a matplotlib figure."""
        fig = create_comprehensive_frame_figure(sample_frame)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_has_six_panels(self, sample_frame):
        """Should create 6 subplots."""
        fig = create_comprehensive_frame_figure(sample_frame)

        # 2x3 layout = 6 axes
        assert len(fig.axes) >= 6
        plt.close(fig)

    def test_custom_figsize(self, sample_frame):
        """Should respect custom figure size."""
        fig = create_comprehensive_frame_figure(sample_frame, figsize=(10, 6))

        assert fig.get_figwidth() == 10
        assert fig.get_figheight() == 6
        plt.close(fig)

    def test_works_without_pressure(self, sample_grid, uniform_velocity):
        """Should work without pressure field."""
        X, Y = sample_grid
        u, v = uniform_velocity
        frame = create_frame_data(X, Y, u, v, time_index=0)

        fig = create_comprehensive_frame_figure(frame)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)
