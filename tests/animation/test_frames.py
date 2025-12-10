"""Tests for cfd_viz.animation.frames module."""

import numpy as np
import pytest

from cfd_viz.animation.frames import (
    AnimationFrames,
    FrameData,
    ParticleTraces,
    advect_particles_through_frames,
    compute_particle_positions,
    create_animation_frames,
    create_frame_data,
    initialize_particles,
    subsample_frame,
)


@pytest.fixture
def sample_grid():
    """Create a sample grid for testing."""
    x = np.linspace(0, 1, 50)
    y = np.linspace(0, 0.5, 25)
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
    # Simple vortex pattern
    cx, cy = 0.5, 0.25
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    r = np.maximum(r, 0.01)  # Avoid division by zero
    u = -(Y - cy) / r
    v = (X - cx) / r
    return u, v


@pytest.fixture
def sample_pressure(sample_grid):
    """Create sample pressure field."""
    X, Y = sample_grid
    p = X + Y  # Simple linear pressure
    return p


class TestFrameData:
    """Tests for FrameData dataclass."""

    def test_create_frame_data_basic(self, sample_grid, uniform_velocity):
        """Should create FrameData with basic fields."""
        X, Y = sample_grid
        u, v = uniform_velocity

        frame = create_frame_data(X, Y, u, v, time_index=0)

        assert isinstance(frame, FrameData)
        assert frame.X is X
        assert frame.Y is Y
        assert "u" in frame.fields
        assert "v" in frame.fields
        assert frame.time_index == 0

    def test_create_frame_data_with_pressure(
        self, sample_grid, uniform_velocity, sample_pressure
    ):
        """Should include pressure field when provided."""
        X, Y = sample_grid
        u, v = uniform_velocity
        p = sample_pressure

        frame = create_frame_data(X, Y, u, v, p=p, time_index=0)

        assert "p" in frame.fields
        assert np.array_equal(frame.fields["p"], p)

    def test_create_frame_data_computes_derived(self, sample_grid, uniform_velocity):
        """Should compute derived fields (velocity_mag, vorticity)."""
        X, Y = sample_grid
        u, v = uniform_velocity

        frame = create_frame_data(X, Y, u, v, time_index=0, compute_derived=True)

        assert "velocity_mag" in frame.fields
        assert "vorticity" in frame.fields
        # For uniform flow, velocity magnitude should be 1.0
        assert np.allclose(frame.fields["velocity_mag"], 1.0)

    def test_create_frame_data_no_derived(self, sample_grid, uniform_velocity):
        """Should skip derived fields when compute_derived=False."""
        X, Y = sample_grid
        u, v = uniform_velocity

        frame = create_frame_data(X, Y, u, v, time_index=0, compute_derived=False)

        assert "velocity_mag" not in frame.fields
        assert "vorticity" not in frame.fields

    def test_create_frame_data_with_timestamp(self, sample_grid, uniform_velocity):
        """Should include timestamp when provided."""
        X, Y = sample_grid
        u, v = uniform_velocity

        frame = create_frame_data(X, Y, u, v, time_index=5, timestamp=0.5)

        assert frame.time_index == 5
        assert frame.timestamp == 0.5


class TestAnimationFrames:
    """Tests for AnimationFrames dataclass."""

    def test_create_animation_frames(
        self, sample_grid, uniform_velocity, sample_pressure
    ):
        """Should create AnimationFrames from list of data."""
        X, Y = sample_grid
        u, v = uniform_velocity
        p = sample_pressure

        frames_list = [
            (X, Y, u, v, p),
            (X, Y, u * 1.1, v, p * 1.1),
            (X, Y, u * 1.2, v, p * 1.2),
        ]

        anim_frames = create_animation_frames(frames_list)

        assert isinstance(anim_frames, AnimationFrames)
        assert len(anim_frames) == 3
        assert anim_frames[0].time_index == 0
        assert anim_frames[2].time_index == 2

    def test_animation_frames_with_time_indices(
        self, sample_grid, uniform_velocity, sample_pressure
    ):
        """Should use provided time indices."""
        X, Y = sample_grid
        u, v = uniform_velocity
        p = sample_pressure

        frames_list = [(X, Y, u, v, p), (X, Y, u, v, p)]
        time_indices = [100, 200]

        anim_frames = create_animation_frames(frames_list, time_indices=time_indices)

        assert anim_frames[0].time_index == 100
        assert anim_frames[1].time_index == 200

    def test_animation_frames_get_field_range(
        self, sample_grid, uniform_velocity, sample_pressure
    ):
        """Should compute field range across all frames."""
        X, Y = sample_grid
        u, v = uniform_velocity
        p = sample_pressure

        frames_list = [
            (X, Y, u, v, p),
            (X, Y, u * 2.0, v, p),  # Max velocity will be 2.0
        ]

        anim_frames = create_animation_frames(frames_list)
        vmin, vmax = anim_frames.get_field_range("velocity_mag")

        assert vmin == pytest.approx(1.0)
        assert vmax == pytest.approx(2.0)

    def test_animation_frames_getitem(
        self, sample_grid, uniform_velocity, sample_pressure
    ):
        """Should support indexing."""
        X, Y = sample_grid
        u, v = uniform_velocity
        p = sample_pressure

        frames_list = [(X, Y, u, v, p), (X, Y, u, v, p)]
        anim_frames = create_animation_frames(frames_list)

        frame = anim_frames[1]
        assert isinstance(frame, FrameData)
        assert frame.time_index == 1


class TestSubsampleFrame:
    """Tests for subsample_frame function."""

    def test_subsample_reduces_size(self, sample_grid, uniform_velocity):
        """Should reduce array sizes by step factor."""
        X, Y = sample_grid
        u, v = uniform_velocity

        frame = create_frame_data(X, Y, u, v, time_index=0)
        subsampled = subsample_frame(frame, step=2)

        # Original shape is (25, 50), subsampled should be (13, 25) for step=2
        assert subsampled.X.shape[0] < frame.X.shape[0]
        assert subsampled.X.shape[1] < frame.X.shape[1]

    def test_subsample_preserves_time_index(self, sample_grid, uniform_velocity):
        """Should preserve time index."""
        X, Y = sample_grid
        u, v = uniform_velocity

        frame = create_frame_data(X, Y, u, v, time_index=42)
        subsampled = subsample_frame(frame, step=2)

        assert subsampled.time_index == 42

    def test_subsample_selective_fields(self, sample_grid, uniform_velocity):
        """Should only subsample specified fields."""
        X, Y = sample_grid
        u, v = uniform_velocity

        frame = create_frame_data(X, Y, u, v, time_index=0)
        subsampled = subsample_frame(frame, step=2, fields_to_subsample=["u", "v"])

        # u and v should be subsampled
        assert subsampled.fields["u"].shape[0] < frame.fields["u"].shape[0]
        # velocity_mag should NOT be subsampled (not in list)
        assert (
            subsampled.fields["velocity_mag"].shape
            == frame.fields["velocity_mag"].shape
        )


class TestParticleTracking:
    """Tests for particle tracking functions."""

    def test_initialize_particles(self, sample_grid):
        """Should initialize particles near inlet."""
        X, Y = sample_grid
        n_particles = 20

        px, py = initialize_particles(X, Y, n_particles=n_particles, seed=42)

        assert len(px) == n_particles
        assert len(py) == n_particles
        # Particles should be near the left edge
        assert np.all(px <= X.min() + 0.1 * (X.max() - X.min()))
        assert np.all(py >= Y.min())
        assert np.all(py <= Y.max())

    def test_initialize_particles_reproducible(self, sample_grid):
        """Should be reproducible with seed."""
        X, Y = sample_grid

        px1, py1 = initialize_particles(X, Y, n_particles=10, seed=123)
        px2, py2 = initialize_particles(X, Y, n_particles=10, seed=123)

        assert np.allclose(px1, px2)
        assert np.allclose(py1, py2)

    def test_compute_particle_positions_advects(self, sample_grid, uniform_velocity):
        """Should advect particles in flow direction."""
        X, Y = sample_grid
        u, v = uniform_velocity

        # Initial positions
        px = np.array([0.1, 0.2, 0.3])
        py = np.array([0.25, 0.25, 0.25])

        new_px, new_py = compute_particle_positions(X, Y, u, v, px, py, dt=0.1)

        # For uniform flow u=1.0, v=0, particles should move right
        assert np.all(new_px > px)
        assert np.allclose(new_py, py)  # y should not change

    def test_advect_particles_through_frames(
        self, sample_grid, uniform_velocity, sample_pressure
    ):
        """Should create particle traces through frames."""
        X, Y = sample_grid
        u, v = uniform_velocity
        p = sample_pressure

        frames_list = [(X, Y, u, v, p), (X, Y, u, v, p), (X, Y, u, v, p)]
        anim_frames = create_animation_frames(frames_list)

        traces = advect_particles_through_frames(
            anim_frames, n_particles=10, dt=0.01, steps_per_frame=5, seed=42
        )

        assert isinstance(traces, ParticleTraces)
        assert traces.n_particles == 10
        # Should have positions for each frame plus initial
        assert len(traces.positions_x) >= len(anim_frames)

    def test_particle_traces_get_current(
        self, sample_grid, uniform_velocity, sample_pressure
    ):
        """Should return current particle positions."""
        X, Y = sample_grid
        u, v = uniform_velocity
        p = sample_pressure

        frames_list = [(X, Y, u, v, p)]
        anim_frames = create_animation_frames(frames_list)

        traces = advect_particles_through_frames(anim_frames, n_particles=5, seed=42)

        px, py = traces.get_current_positions()

        assert len(px) == 5
        assert len(py) == 5


class TestVorticityComputation:
    """Tests for vorticity computation in frame data."""

    def test_vorticity_uniform_flow(self, sample_grid, uniform_velocity):
        """Uniform flow should have zero vorticity."""
        X, Y = sample_grid
        u, v = uniform_velocity

        frame = create_frame_data(X, Y, u, v, time_index=0)
        vorticity = frame.fields["vorticity"]

        # Uniform flow has zero vorticity
        assert np.allclose(vorticity, 0, atol=1e-10)

    def test_vorticity_vortex_flow(self, sample_grid, vortex_velocity):
        """Vortex flow should have non-zero vorticity."""
        X, Y = sample_grid
        u, v = vortex_velocity

        frame = create_frame_data(X, Y, u, v, time_index=0)
        vorticity = frame.fields["vorticity"]

        # Vortex has non-zero vorticity
        assert np.max(np.abs(vorticity)) > 0


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_frames_list(self):
        """Should handle empty frames list."""
        anim_frames = create_animation_frames([])

        assert len(anim_frames) == 0

    def test_single_frame(self, sample_grid, uniform_velocity, sample_pressure):
        """Should handle single frame."""
        X, Y = sample_grid
        u, v = uniform_velocity
        p = sample_pressure

        frames_list = [(X, Y, u, v, p)]
        anim_frames = create_animation_frames(frames_list)

        assert len(anim_frames) == 1

    def test_field_range_missing_field(
        self, sample_grid, uniform_velocity, sample_pressure
    ):
        """Should return default range for missing field."""
        X, Y = sample_grid
        u, v = uniform_velocity
        p = sample_pressure

        frames_list = [(X, Y, u, v, p)]
        anim_frames = create_animation_frames(frames_list)

        vmin, vmax = anim_frames.get_field_range("nonexistent_field")

        assert vmin == 0.0
        assert vmax == 1.0
