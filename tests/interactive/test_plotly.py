"""Tests for cfd_viz.interactive.plotly module."""

import numpy as np
import plotly.graph_objects as go
import pytest

from cfd_viz.interactive.plotly import (
    InteractiveFrameCollection,
    InteractiveFrameData,
    create_animated_dashboard,
    create_contour_figure,
    create_convergence_figure,
    create_dashboard_figure,
    create_heatmap_figure,
    create_interactive_frame,
    create_interactive_frame_collection,
    create_surface_figure,
    create_time_series_figure,
    create_vector_figure,
)


@pytest.fixture
def sample_coords():
    """Create sample coordinate arrays."""
    x = np.linspace(0, 1, 30)
    y = np.linspace(0, 0.5, 15)
    return x, y


@pytest.fixture
def uniform_velocity(sample_coords):
    """Create uniform velocity field."""
    x, y = sample_coords
    X, Y = np.meshgrid(x, y)
    u = np.ones_like(X) * 1.0
    v = np.zeros_like(X)
    return u, v


@pytest.fixture
def sample_pressure(sample_coords):
    """Create sample pressure field."""
    x, y = sample_coords
    X, Y = np.meshgrid(x, y)
    p = X + Y
    return p


@pytest.fixture
def sample_frame(sample_coords, uniform_velocity, sample_pressure):
    """Create a sample interactive frame."""
    x, y = sample_coords
    u, v = uniform_velocity
    p = sample_pressure
    return create_interactive_frame(x, y, u, v, p=p, time_index=0)


@pytest.fixture
def sample_frame_collection(sample_coords, uniform_velocity, sample_pressure):
    """Create sample frame collection."""
    x, y = sample_coords
    u, v = uniform_velocity
    p = sample_pressure

    frames_list = [
        (x, y, u, v, p),
        (x, y, u * 1.1, v, p * 1.1),
        (x, y, u * 1.2, v, p * 1.2),
    ]
    return create_interactive_frame_collection(frames_list, time_indices=[0, 100, 200])


class TestInteractiveFrameData:
    """Tests for InteractiveFrameData creation."""

    def test_create_frame_basic(self, sample_coords, uniform_velocity):
        """Should create frame with basic fields."""
        x, y = sample_coords
        u, v = uniform_velocity

        frame = create_interactive_frame(x, y, u, v, time_index=0)

        assert isinstance(frame, InteractiveFrameData)
        assert np.array_equal(frame.x, x)
        assert np.array_equal(frame.y, y)
        assert "u" in frame.fields
        assert "v" in frame.fields
        assert frame.time_index == 0

    def test_create_frame_with_pressure(
        self, sample_coords, uniform_velocity, sample_pressure
    ):
        """Should include pressure field."""
        x, y = sample_coords
        u, v = uniform_velocity
        p = sample_pressure

        frame = create_interactive_frame(x, y, u, v, p=p, time_index=0)

        assert "p" in frame.fields
        assert np.array_equal(frame.fields["p"], p)

    def test_create_frame_computes_derived(self, sample_coords, uniform_velocity):
        """Should compute velocity magnitude and vorticity."""
        x, y = sample_coords
        u, v = uniform_velocity

        frame = create_interactive_frame(x, y, u, v, compute_derived=True)

        assert "velocity_mag" in frame.fields
        assert "vorticity" in frame.fields
        assert np.allclose(frame.fields["velocity_mag"], 1.0)

    def test_create_frame_no_derived(self, sample_coords, uniform_velocity):
        """Should skip derived fields when compute_derived=False."""
        x, y = sample_coords
        u, v = uniform_velocity

        frame = create_interactive_frame(x, y, u, v, compute_derived=False)

        assert "velocity_mag" not in frame.fields
        assert "vorticity" not in frame.fields


class TestInteractiveFrameCollection:
    """Tests for InteractiveFrameCollection."""

    def test_create_collection(self, sample_coords, uniform_velocity, sample_pressure):
        """Should create collection from list of data."""
        x, y = sample_coords
        u, v = uniform_velocity
        p = sample_pressure

        frames_list = [(x, y, u, v, p), (x, y, u, v, p)]
        collection = create_interactive_frame_collection(frames_list)

        assert isinstance(collection, InteractiveFrameCollection)
        assert len(collection) == 2

    def test_collection_with_time_indices(
        self, sample_coords, uniform_velocity, sample_pressure
    ):
        """Should use provided time indices."""
        x, y = sample_coords
        u, v = uniform_velocity
        p = sample_pressure

        frames_list = [(x, y, u, v, p), (x, y, u, v, p)]
        collection = create_interactive_frame_collection(
            frames_list, time_indices=[100, 200]
        )

        assert collection.time_indices == [100, 200]
        assert collection[0].time_index == 100
        assert collection[1].time_index == 200

    def test_collection_indexing(self, sample_frame_collection):
        """Should support indexing."""
        frame = sample_frame_collection[1]

        assert isinstance(frame, InteractiveFrameData)
        assert frame.time_index == 100

    def test_collection_add_frame(
        self, sample_coords, uniform_velocity, sample_pressure
    ):
        """Should support adding frames."""
        x, y = sample_coords
        u, v = uniform_velocity
        p = sample_pressure

        collection = InteractiveFrameCollection()
        frame = create_interactive_frame(x, y, u, v, p=p, time_index=50)
        collection.add_frame(frame)

        assert len(collection) == 1
        assert collection.time_indices == [50]


class TestHeatmapFigure:
    """Tests for create_heatmap_figure."""

    def test_returns_figure(self, sample_coords, uniform_velocity):
        """Should return Plotly Figure."""
        x, y = sample_coords
        u, _ = uniform_velocity

        fig = create_heatmap_figure(x, y, u, title="Test")

        assert isinstance(fig, go.Figure)

    def test_has_heatmap_trace(self, sample_coords, uniform_velocity):
        """Should contain heatmap trace."""
        x, y = sample_coords
        u, _ = uniform_velocity

        fig = create_heatmap_figure(x, y, u)

        assert len(fig.data) == 1
        assert isinstance(fig.data[0], go.Heatmap)

    def test_custom_colorscale(self, sample_coords, uniform_velocity):
        """Should use custom colorscale."""
        x, y = sample_coords
        u, _ = uniform_velocity

        fig = create_heatmap_figure(x, y, u, colorscale="RdBu")

        # Plotly expands colorscale names to RGB values, so check it's not the default
        assert fig.data[0].colorscale is not None


class TestContourFigure:
    """Tests for create_contour_figure."""

    def test_returns_figure(self, sample_coords, uniform_velocity):
        """Should return Plotly Figure."""
        x, y = sample_coords
        u, _ = uniform_velocity

        fig = create_contour_figure(x, y, u)

        assert isinstance(fig, go.Figure)

    def test_has_contour_trace(self, sample_coords, uniform_velocity):
        """Should contain contour trace."""
        x, y = sample_coords
        u, _ = uniform_velocity

        fig = create_contour_figure(x, y, u)

        assert len(fig.data) == 1
        assert isinstance(fig.data[0], go.Contour)


class TestVectorFigure:
    """Tests for create_vector_figure."""

    def test_returns_figure(self, sample_coords, uniform_velocity):
        """Should return Plotly Figure."""
        x, y = sample_coords
        u, v = uniform_velocity

        fig = create_vector_figure(x, y, u, v, subsample=5)

        assert isinstance(fig, go.Figure)

    def test_has_scatter_trace(self, sample_coords, uniform_velocity):
        """Should contain scatter trace for markers."""
        x, y = sample_coords
        u, v = uniform_velocity

        fig = create_vector_figure(x, y, u, v, subsample=5)

        # Should have line traces and one scatter trace
        scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
        assert len(scatter_traces) > 0


class TestSurfaceFigure:
    """Tests for create_surface_figure."""

    def test_returns_figure(self, sample_coords, uniform_velocity):
        """Should return Plotly Figure."""
        x, y = sample_coords
        u, _ = uniform_velocity

        fig = create_surface_figure(x, y, u)

        assert isinstance(fig, go.Figure)

    def test_has_surface_trace(self, sample_coords, uniform_velocity):
        """Should contain surface trace."""
        x, y = sample_coords
        u, _ = uniform_velocity

        fig = create_surface_figure(x, y, u)

        assert len(fig.data) == 1
        assert isinstance(fig.data[0], go.Surface)


class TestTimeSeriesFigure:
    """Tests for create_time_series_figure."""

    def test_returns_figure(self):
        """Should return Plotly Figure."""
        times = [0, 100, 200, 300]
        metrics = {"max_vel": [1.0, 1.1, 1.2, 1.3], "avg_vel": [0.5, 0.6, 0.7, 0.8]}

        fig = create_time_series_figure(times, metrics)

        assert isinstance(fig, go.Figure)

    def test_has_traces_for_each_metric(self):
        """Should have one trace per metric."""
        times = [0, 100, 200]
        metrics = {"metric1": [1, 2, 3], "metric2": [4, 5, 6], "metric3": [7, 8, 9]}

        fig = create_time_series_figure(times, metrics)

        assert len(fig.data) == 3


class TestConvergenceFigure:
    """Tests for create_convergence_figure."""

    def test_returns_figure(self, sample_frame_collection):
        """Should return Plotly Figure."""
        fig = create_convergence_figure(sample_frame_collection)

        assert isinstance(fig, go.Figure)

    def test_has_velocity_traces(self, sample_frame_collection):
        """Should include velocity traces."""
        fig = create_convergence_figure(sample_frame_collection)

        trace_names = [t.name for t in fig.data if t.name]
        assert "Max Velocity" in trace_names
        assert "Avg Velocity" in trace_names


class TestDashboardFigure:
    """Tests for create_dashboard_figure."""

    def test_returns_figure(self, sample_frame):
        """Should return Plotly Figure."""
        fig = create_dashboard_figure(sample_frame)

        assert isinstance(fig, go.Figure)

    def test_has_multiple_traces(self, sample_frame):
        """Should have multiple traces for panels."""
        fig = create_dashboard_figure(sample_frame)

        assert len(fig.data) >= 6  # At least 6 panels


class TestAnimatedDashboard:
    """Tests for create_animated_dashboard."""

    def test_returns_figure(self, sample_frame_collection):
        """Should return Plotly Figure."""
        fig = create_animated_dashboard(sample_frame_collection)

        assert isinstance(fig, go.Figure)

    def test_has_animation_frames(self, sample_frame_collection):
        """Should have animation frames."""
        fig = create_animated_dashboard(sample_frame_collection)

        assert len(fig.frames) == len(sample_frame_collection)

    def test_has_play_button(self, sample_frame_collection):
        """Should have play/pause buttons."""
        fig = create_animated_dashboard(sample_frame_collection)

        assert fig.layout.updatemenus is not None
        assert len(fig.layout.updatemenus) > 0

    def test_has_slider(self, sample_frame_collection):
        """Should have iteration slider."""
        fig = create_animated_dashboard(sample_frame_collection)

        assert fig.layout.sliders is not None
        assert len(fig.layout.sliders) > 0

    def test_empty_collection_raises(self):
        """Should raise error for empty collection."""
        empty_collection = InteractiveFrameCollection()

        with pytest.raises(ValueError, match="No frames"):
            create_animated_dashboard(empty_collection)


class TestIntegration:
    """Integration tests for interactive workflow."""

    def test_full_workflow(self, sample_coords, uniform_velocity, sample_pressure):
        """Test complete interactive workflow."""
        x, y = sample_coords
        u, v = uniform_velocity
        p = sample_pressure

        # Create frames
        frames_list = [(x, y, u * factor, v, p * factor) for factor in [1.0, 1.1, 1.2]]
        collection = create_interactive_frame_collection(
            frames_list, time_indices=[0, 100, 200]
        )

        assert len(collection) == 3

        # Create various figures
        heatmap = create_heatmap_figure(x, y, u, title="Velocity")
        assert isinstance(heatmap, go.Figure)

        dashboard = create_dashboard_figure(collection[0])
        assert isinstance(dashboard, go.Figure)

        animated = create_animated_dashboard(collection)
        assert len(animated.frames) == 3
