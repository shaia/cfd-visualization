"""Tests for cfd_viz.animation.renderers module."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

matplotlib.use("Agg")  # Non-interactive backend for testing

from cfd_viz.animation.frames import (
    AnimationFrames,
    advect_particles_through_frames,
    create_animation_frames,
    create_frame_data,
)
from cfd_viz.animation.renderers import (
    create_3d_surface_animation,
    create_cfd_colormap,
    create_field_animation,
    create_multi_panel_animation,
    create_particle_trace_animation,
    create_streamline_animation,
    create_vector_animation,
    create_velocity_colormap,
    create_vorticity_analysis_animation,
    render_contour_frame,
    render_streamline_frame,
    render_vector_frame,
)


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
def vortex_velocity(sample_grid):
    """Create vortex velocity field."""
    X, Y = sample_grid
    cx, cy = 0.5, 0.25
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    r = np.maximum(r, 0.01)
    u = -(Y - cy) / r
    v = (X - cx) / r
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


class TestColormaps:
    """Tests for colormap creation functions."""

    def test_create_cfd_colormap(self):
        """Should create a valid colormap."""
        cmap = create_cfd_colormap()

        assert cmap is not None
        assert cmap.name == "cfd_custom"
        assert cmap.N == 256

    def test_create_velocity_colormap(self):
        """Should create a velocity colormap."""
        cmap = create_velocity_colormap()

        assert cmap is not None
        assert cmap.name == "velocity"


class TestFrameRenderers:
    """Tests for single frame rendering functions."""

    def test_render_contour_frame(self, sample_frame):
        """Should render contour plot on axes."""
        fig, ax = plt.subplots()

        render_contour_frame(ax, sample_frame, field_name="velocity_mag")

        # Check that something was plotted
        assert len(ax.collections) > 0 or len(ax.images) > 0
        plt.close(fig)

    def test_render_contour_frame_missing_field(self, sample_frame):
        """Should handle missing field gracefully."""
        fig, ax = plt.subplots()

        render_contour_frame(ax, sample_frame, field_name="nonexistent")

        # Should show error text
        assert len(ax.texts) > 0
        plt.close(fig)

    def test_render_contour_frame_with_title(self, sample_frame):
        """Should set custom title."""
        fig, ax = plt.subplots()

        render_contour_frame(ax, sample_frame, title="Custom Title")

        assert ax.get_title() == "Custom Title"
        plt.close(fig)

    def test_render_vector_frame(self, sample_frame):
        """Should render vector field."""
        fig, ax = plt.subplots()

        render_vector_frame(ax, sample_frame, subsample=2)

        # Quiver creates collections
        assert len(ax.collections) > 0
        plt.close(fig)

    def test_render_streamline_frame(
        self, sample_grid, vortex_velocity, sample_pressure
    ):
        """Should render streamlines."""
        X, Y = sample_grid
        u, v = vortex_velocity
        p = sample_pressure
        frame = create_frame_data(X, Y, u, v, p=p, time_index=0)

        fig, ax = plt.subplots()
        render_streamline_frame(ax, frame, density=1.0)

        # Streamplot creates line collections
        assert len(ax.collections) > 0 or len(ax.patches) > 0
        plt.close(fig)


class TestFieldAnimation:
    """Tests for create_field_animation function."""

    def test_creates_animation(self, sample_animation_frames):
        """Should create matplotlib animation."""
        fig, anim = create_field_animation(sample_animation_frames, "velocity_mag")

        assert fig is not None
        assert anim is not None
        plt.close(fig)

    def test_empty_frames_raises(self):
        """Should raise error for empty frames."""
        empty_frames = AnimationFrames(frames=[])

        with pytest.raises(ValueError, match="No frames"):
            create_field_animation(empty_frames, "velocity_mag")

    def test_missing_field_raises(self, sample_animation_frames):
        """Should raise error for missing field."""
        with pytest.raises(ValueError, match="not found"):
            create_field_animation(sample_animation_frames, "nonexistent")

    def test_custom_parameters(self, sample_animation_frames):
        """Should accept custom parameters."""
        fig, anim = create_field_animation(
            sample_animation_frames,
            "velocity_mag",
            figsize=(8, 4),
            cmap="plasma",
            interval=100,
        )

        assert fig is not None
        plt.close(fig)


class TestStreamlineAnimation:
    """Tests for create_streamline_animation function."""

    def test_creates_animation(self, sample_grid, vortex_velocity, sample_pressure):
        """Should create streamline animation."""
        X, Y = sample_grid
        u, v = vortex_velocity
        p = sample_pressure

        frames_list = [(X, Y, u, v, p), (X, Y, u * 1.1, v * 1.1, p)]
        anim_frames = create_animation_frames(frames_list)

        fig, anim = create_streamline_animation(anim_frames)

        assert fig is not None
        assert anim is not None
        plt.close(fig)


class TestVectorAnimation:
    """Tests for create_vector_animation function."""

    def test_creates_animation(self, sample_animation_frames):
        """Should create vector field animation."""
        fig, anim = create_vector_animation(sample_animation_frames)

        assert fig is not None
        assert anim is not None
        plt.close(fig)

    def test_custom_subsample(self, sample_animation_frames):
        """Should accept custom subsample parameter."""
        fig, anim = create_vector_animation(sample_animation_frames, subsample=3)

        assert fig is not None
        plt.close(fig)


class TestMultiPanelAnimation:
    """Tests for create_multi_panel_animation function."""

    def test_creates_animation(self, sample_animation_frames):
        """Should create multi-panel animation."""
        fig, anim = create_multi_panel_animation(sample_animation_frames)

        assert fig is not None
        assert anim is not None
        plt.close(fig)

    def test_custom_title(self, sample_animation_frames):
        """Should accept custom title."""
        fig, anim = create_multi_panel_animation(
            sample_animation_frames, title="Custom Dashboard"
        )

        assert "Custom Dashboard" in fig._suptitle.get_text()
        plt.close(fig)


class TestParticleTraceAnimation:
    """Tests for create_particle_trace_animation function."""

    def test_creates_animation(self, sample_animation_frames):
        """Should create particle trace animation."""
        traces = advect_particles_through_frames(
            sample_animation_frames, n_particles=5, seed=42
        )

        fig, anim = create_particle_trace_animation(sample_animation_frames, traces)

        assert fig is not None
        assert anim is not None
        plt.close(fig)


class TestVorticityAnalysisAnimation:
    """Tests for create_vorticity_analysis_animation function."""

    def test_creates_animation(self, sample_grid, vortex_velocity, sample_pressure):
        """Should create vorticity analysis animation."""
        X, Y = sample_grid
        u, v = vortex_velocity
        p = sample_pressure

        frames_list = [(X, Y, u, v, p), (X, Y, u * 1.1, v * 1.1, p)]
        anim_frames = create_animation_frames(frames_list)

        fig, anim = create_vorticity_analysis_animation(anim_frames)

        assert fig is not None
        assert anim is not None
        plt.close(fig)


class TestSurfaceAnimation:
    """Tests for create_3d_surface_animation function."""

    def test_creates_animation(self, sample_animation_frames):
        """Should create 3D surface animation."""
        fig, anim = create_3d_surface_animation(sample_animation_frames, "velocity_mag")

        assert fig is not None
        assert anim is not None
        plt.close(fig)

    def test_no_rotation(self, sample_animation_frames):
        """Should work without camera rotation."""
        fig, anim = create_3d_surface_animation(
            sample_animation_frames, "velocity_mag", rotate_camera=False
        )

        assert fig is not None
        plt.close(fig)


class TestAnimationIntegration:
    """Integration tests for animation workflows."""

    def test_full_workflow(self, sample_grid, uniform_velocity, sample_pressure):
        """Test complete animation workflow."""
        X, Y = sample_grid
        u, v = uniform_velocity
        p = sample_pressure

        # Create frames
        frames_list = [
            (X, Y, u * factor, v, p * factor) for factor in [1.0, 1.1, 1.2, 1.3]
        ]
        anim_frames = create_animation_frames(frames_list)

        # Verify frame data
        assert len(anim_frames) == 4
        vmin, vmax = anim_frames.get_field_range("velocity_mag")
        assert vmax > vmin

        # Create animation
        fig, anim = create_field_animation(anim_frames, "velocity_mag")
        assert anim is not None

        plt.close(fig)

    def test_particle_trace_workflow(
        self, sample_grid, uniform_velocity, sample_pressure
    ):
        """Test particle trace workflow."""
        X, Y = sample_grid
        u, v = uniform_velocity
        p = sample_pressure

        frames_list = [(X, Y, u, v, p) for _ in range(5)]
        anim_frames = create_animation_frames(frames_list)

        # Initialize and advect particles
        traces = advect_particles_through_frames(
            anim_frames, n_particles=10, dt=0.01, steps_per_frame=3, seed=42
        )

        # Create animation
        fig, anim = create_particle_trace_animation(anim_frames, traces)
        assert anim is not None

        plt.close(fig)
