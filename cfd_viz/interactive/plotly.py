"""Interactive Figure Creation using Plotly.

Pure functions for creating Plotly figures from CFD data.
Functions accept numpy arrays and return Plotly Figure objects.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
from numpy.typing import NDArray


@dataclass
class InteractiveFrameData:
    """Data for a single interactive frame.

    Attributes:
        x: 1D array of x-coordinates.
        y: 1D array of y-coordinates.
        fields: Dictionary of field name -> 2D array.
        time_index: Frame index or iteration number.
    """

    x: NDArray
    y: NDArray
    fields: Dict[str, NDArray]
    time_index: int


@dataclass
class InteractiveFrameCollection:
    """Collection of frames for interactive visualization.

    Attributes:
        frames: List of InteractiveFrameData objects.
        time_indices: List of time/iteration indices.
    """

    frames: List[InteractiveFrameData] = field(default_factory=list)
    time_indices: List[int] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> InteractiveFrameData:
        return self.frames[index]

    def add_frame(self, frame: InteractiveFrameData) -> None:
        """Add a frame to the collection."""
        self.frames.append(frame)
        self.time_indices.append(frame.time_index)


def create_interactive_frame(
    x: NDArray,
    y: NDArray,
    u: NDArray,
    v: NDArray,
    p: Optional[NDArray] = None,
    T: Optional[NDArray] = None,
    time_index: int = 0,
    compute_derived: bool = True,
) -> InteractiveFrameData:
    """Create interactive frame data from velocity and pressure fields.

    Args:
        x: 1D array of x-coordinates.
        y: 1D array of y-coordinates.
        u: 2D array of x-velocity component.
        v: 2D array of y-velocity component.
        p: Optional 2D array of pressure.
        T: Optional 2D array of temperature.
        time_index: Frame index or iteration number.
        compute_derived: Whether to compute derived fields.

    Returns:
        InteractiveFrameData with all field data.
    """
    fields: Dict[str, NDArray] = {
        "u": u,
        "v": v,
    }

    if p is not None:
        fields["p"] = p

    if T is not None:
        fields["T"] = T

    if compute_derived:
        fields["velocity_mag"] = np.sqrt(u**2 + v**2)
        fields["vorticity"] = np.gradient(v, axis=1) - np.gradient(u, axis=0)

    return InteractiveFrameData(
        x=x,
        y=y,
        fields=fields,
        time_index=time_index,
    )


def create_interactive_frame_collection(
    frames_list: List[Tuple[NDArray, NDArray, NDArray, NDArray, Optional[NDArray]]],
    time_indices: Optional[List[int]] = None,
) -> InteractiveFrameCollection:
    """Create interactive frame collection from list of field data.

    Args:
        frames_list: List of (x, y, u, v, p) tuples for each frame.
        time_indices: Optional list of time indices for each frame.

    Returns:
        InteractiveFrameCollection containing all frame data.
    """
    collection = InteractiveFrameCollection()

    for i, frame_tuple in enumerate(frames_list):
        x, y, u, v, p = frame_tuple
        time_idx = time_indices[i] if time_indices else i

        frame = create_interactive_frame(x=x, y=y, u=u, v=v, p=p, time_index=time_idx)
        collection.add_frame(frame)

    return collection


# =============================================================================
# Single-Panel Figures
# =============================================================================


def create_heatmap_figure(
    x: NDArray,
    y: NDArray,
    field: NDArray,
    title: str = "Field",
    colorscale: str = "Viridis",
    height: int = 500,
    width: int = 700,
) -> go.Figure:
    """Create an interactive heatmap figure.

    Args:
        x: 1D array of x-coordinates.
        y: 1D array of y-coordinates.
        field: 2D array of field values.
        title: Plot title.
        colorscale: Plotly colorscale name.
        height: Figure height in pixels.
        width: Figure width in pixels.

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure(
        data=go.Heatmap(
            z=field,
            x=x,
            y=y,
            colorscale=colorscale,
            colorbar=dict(title=title),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="X",
        yaxis_title="Y",
        height=height,
        width=width,
        yaxis=dict(scaleanchor="x", scaleratio=1),
    )

    return fig


def create_contour_figure(
    x: NDArray,
    y: NDArray,
    field: NDArray,
    title: str = "Contour",
    colorscale: str = "Viridis",
    ncontours: int = 20,
    height: int = 500,
    width: int = 700,
) -> go.Figure:
    """Create an interactive contour figure.

    Args:
        x: 1D array of x-coordinates.
        y: 1D array of y-coordinates.
        field: 2D array of field values.
        title: Plot title.
        colorscale: Plotly colorscale name.
        ncontours: Number of contour levels.
        height: Figure height in pixels.
        width: Figure width in pixels.

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure(
        data=go.Contour(
            z=field,
            x=x,
            y=y,
            colorscale=colorscale,
            ncontours=ncontours,
            colorbar=dict(title=title),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="X",
        yaxis_title="Y",
        height=height,
        width=width,
        yaxis=dict(scaleanchor="x", scaleratio=1),
    )

    return fig


def create_vector_figure(
    x: NDArray,
    y: NDArray,
    u: NDArray,
    v: NDArray,
    subsample: int = 5,
    title: str = "Vector Field",
    colorscale: str = "Viridis",
    height: int = 500,
    width: int = 700,
) -> go.Figure:
    """Create an interactive vector field figure using quiver-like markers.

    Args:
        x: 1D array of x-coordinates.
        y: 1D array of y-coordinates.
        u: 2D array of x-velocity component.
        v: 2D array of y-velocity component.
        subsample: Subsampling step for vectors.
        title: Plot title.
        colorscale: Plotly colorscale name.
        height: Figure height in pixels.
        width: Figure width in pixels.

    Returns:
        Plotly Figure object.
    """
    X, Y = np.meshgrid(x, y)
    X_sub = X[::subsample, ::subsample]
    Y_sub = Y[::subsample, ::subsample]
    u_sub = u[::subsample, ::subsample]
    v_sub = v[::subsample, ::subsample]
    speed = np.sqrt(u_sub**2 + v_sub**2)

    # Normalize vectors for display
    max_speed = np.max(speed) if np.max(speed) > 0 else 1
    scale = (x.max() - x.min()) / (X_sub.shape[1] * 2)

    fig = go.Figure()

    # Add arrows as lines with markers
    for i in range(X_sub.shape[0]):
        for j in range(X_sub.shape[1]):
            x0, y0 = X_sub[i, j], Y_sub[i, j]
            dx = u_sub[i, j] / max_speed * scale
            dy = v_sub[i, j] / max_speed * scale

            fig.add_trace(
                go.Scatter(
                    x=[x0, x0 + dx],
                    y=[y0, y0 + dy],
                    mode="lines",
                    line=dict(color="rgba(0,0,0,0.5)", width=1),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    # Add colored markers at vector origins
    fig.add_trace(
        go.Scatter(
            x=X_sub.flatten(),
            y=Y_sub.flatten(),
            mode="markers",
            marker=dict(
                size=6,
                color=speed.flatten(),
                colorscale=colorscale,
                colorbar=dict(title="Speed"),
            ),
            name="Vectors",
            hovertemplate="x: %{x:.3f}<br>y: %{y:.3f}<br>speed: %{marker.color:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="X",
        yaxis_title="Y",
        height=height,
        width=width,
        yaxis=dict(scaleanchor="x", scaleratio=1),
        showlegend=False,
    )

    return fig


def create_surface_figure(
    x: NDArray,
    y: NDArray,
    field: NDArray,
    title: str = "3D Surface",
    colorscale: str = "Viridis",
    height: int = 600,
    width: int = 800,
) -> go.Figure:
    """Create an interactive 3D surface figure.

    Args:
        x: 1D array of x-coordinates.
        y: 1D array of y-coordinates.
        field: 2D array of field values.
        title: Plot title.
        colorscale: Plotly colorscale name.
        height: Figure height in pixels.
        width: Figure width in pixels.

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure(
        data=go.Surface(
            z=field,
            x=x,
            y=y,
            colorscale=colorscale,
            colorbar=dict(title=title),
        )
    )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Value",
        ),
        height=height,
        width=width,
    )

    return fig


# =============================================================================
# Time Series Figures
# =============================================================================


def create_time_series_figure(
    times: List[int],
    metrics: Dict[str, List[float]],
    title: str = "Time Series",
    height: int = 500,
    width: int = 800,
) -> go.Figure:
    """Create an interactive time series figure.

    Args:
        times: List of time/iteration values.
        metrics: Dictionary mapping metric names to lists of values.
        title: Plot title.
        height: Figure height in pixels.
        width: Figure width in pixels.

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure()

    colors = ["red", "blue", "green", "orange", "purple", "cyan"]

    for i, (name, values) in enumerate(metrics.items()):
        fig.add_trace(
            go.Scatter(
                x=times,
                y=values,
                mode="lines+markers",
                name=name,
                line=dict(color=colors[i % len(colors)], width=2),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Iteration",
        yaxis_title="Value",
        height=height,
        width=width,
        hovermode="x unified",
        legend=dict(x=0.02, y=0.98),
    )

    return fig


def create_convergence_figure(
    frames: InteractiveFrameCollection,
    title: str = "Convergence History",
    height: int = 500,
    width: int = 800,
) -> go.Figure:
    """Create convergence history figure from frame collection.

    Args:
        frames: InteractiveFrameCollection with velocity/pressure data.
        title: Plot title.
        height: Figure height in pixels.
        width: Figure width in pixels.

    Returns:
        Plotly Figure object.
    """
    times = frames.time_indices
    max_velocities = []
    avg_velocities = []
    max_pressures = []
    min_pressures = []

    for frame in frames.frames:
        vel_mag = frame.fields.get("velocity_mag")
        if vel_mag is None:
            u = frame.fields.get("u", np.zeros((1, 1)))
            v = frame.fields.get("v", np.zeros((1, 1)))
            vel_mag = np.sqrt(u**2 + v**2)

        max_velocities.append(float(np.max(vel_mag)))
        avg_velocities.append(float(np.mean(vel_mag)))

        p = frame.fields.get("p")
        if p is not None:
            max_pressures.append(float(np.max(p)))
            min_pressures.append(float(np.min(p)))

    fig = go.Figure()

    # Velocity traces (left y-axis)
    fig.add_trace(
        go.Scatter(
            x=times,
            y=max_velocities,
            mode="lines+markers",
            name="Max Velocity",
            line=dict(color="red", width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=times,
            y=avg_velocities,
            mode="lines+markers",
            name="Avg Velocity",
            line=dict(color="blue", width=2),
        )
    )

    # Pressure traces (right y-axis)
    if max_pressures:
        fig.add_trace(
            go.Scatter(
                x=times,
                y=max_pressures,
                mode="lines+markers",
                name="Max Pressure",
                line=dict(color="green", width=2, dash="dash"),
                yaxis="y2",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=times,
                y=min_pressures,
                mode="lines+markers",
                name="Min Pressure",
                line=dict(color="orange", width=2, dash="dash"),
                yaxis="y2",
            )
        )

    fig.update_layout(
        title=title,
        xaxis=dict(title="Iteration"),
        yaxis=dict(title="Velocity", side="left"),
        yaxis2=dict(title="Pressure", side="right", overlaying="y"),
        height=height,
        width=width,
        hovermode="x unified",
        legend=dict(x=0.02, y=0.98),
    )

    return fig


# =============================================================================
# Multi-Panel Dashboard
# =============================================================================


def create_dashboard_figure(
    frame: InteractiveFrameData,
    title: str = "CFD Dashboard",
    height: int = 900,
    width: int = 1200,
) -> go.Figure:
    """Create a multi-panel dashboard for a single frame.

    Args:
        frame: InteractiveFrameData to visualize.
        title: Dashboard title.
        height: Figure height in pixels.
        width: Figure width in pixels.

    Returns:
        Plotly Figure object with 6 panels.
    """
    x, y = frame.x, frame.y
    u = frame.fields.get("u", np.zeros((len(y), len(x))))
    v = frame.fields.get("v", np.zeros((len(y), len(x))))
    p = frame.fields.get("p", np.zeros((len(y), len(x))))
    vel_mag = frame.fields.get("velocity_mag", np.sqrt(u**2 + v**2))
    vorticity = frame.fields.get("vorticity", np.zeros((len(y), len(x))))
    T = frame.fields.get("T", np.ones((len(y), len(x))) * 300)

    fig = sp.make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "Velocity Magnitude",
            "Pressure Field",
            "Vorticity",
            "Vector Field",
            "Temperature",
            "3D Surface",
        ),
        specs=[
            [{"type": "heatmap"}, {"type": "heatmap"}, {"type": "heatmap"}],
            [{"type": "scatter"}, {"type": "heatmap"}, {"type": "surface"}],
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    # Row 1: Velocity magnitude
    fig.add_trace(
        go.Heatmap(z=vel_mag, x=x, y=y, colorscale="Viridis", showscale=True),
        row=1,
        col=1,
    )

    # Row 1: Pressure
    fig.add_trace(
        go.Heatmap(z=p, x=x, y=y, colorscale="RdBu", showscale=True),
        row=1,
        col=2,
    )

    # Row 1: Vorticity
    fig.add_trace(
        go.Heatmap(z=vorticity, x=x, y=y, colorscale="RdBu", showscale=True),
        row=1,
        col=3,
    )

    # Row 2: Vector field (simplified as scatter)
    X, Y = np.meshgrid(x, y)
    skip = max(1, min(len(x), len(y)) // 15)
    X_sub = X[::skip, ::skip]
    Y_sub = Y[::skip, ::skip]
    speed_sub = vel_mag[::skip, ::skip]

    fig.add_trace(
        go.Scatter(
            x=X_sub.flatten(),
            y=Y_sub.flatten(),
            mode="markers",
            marker=dict(
                size=8,
                color=speed_sub.flatten(),
                colorscale="Viridis",
                showscale=False,
            ),
        ),
        row=2,
        col=1,
    )

    # Row 2: Temperature
    fig.add_trace(
        go.Heatmap(z=T, x=x, y=y, colorscale="Plasma", showscale=True),
        row=2,
        col=2,
    )

    # Row 2: 3D Surface
    fig.add_trace(
        go.Surface(z=vel_mag, x=x, y=y, colorscale="Viridis", showscale=False),
        row=2,
        col=3,
    )

    fig.update_layout(
        title=dict(text=f"{title} - Step {frame.time_index}", x=0.5),
        height=height,
        width=width,
        showlegend=False,
    )

    return fig


def create_animated_dashboard(
    frames: InteractiveFrameCollection,
    title: str = "Interactive CFD Dashboard",
    height: int = 1000,
    width: int = 1400,
) -> go.Figure:
    """Create an animated multi-panel dashboard.

    Args:
        frames: InteractiveFrameCollection with multiple frames.
        title: Dashboard title.
        height: Figure height in pixels.
        width: Figure width in pixels.

    Returns:
        Plotly Figure object with animation controls.
    """
    if not frames.frames:
        raise ValueError("No frames provided")

    # Create subplot structure
    fig = sp.make_subplots(
        rows=3,
        cols=3,
        subplot_titles=(
            "Velocity Magnitude",
            "Pressure Field",
            "Vorticity",
            "Vector Field",
            "3D Flow",
            "Temperature",
            "Combined",
            "Time Series",
            "3D Surface",
        ),
        specs=[
            [{"type": "heatmap"}, {"type": "heatmap"}, {"type": "heatmap"}],
            [{"type": "scatter"}, {"type": "scatter3d"}, {"type": "heatmap"}],
            [{"type": "heatmap"}, {"type": "scatter"}, {"type": "surface"}],
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.05,
    )

    # Build animation frames
    plotly_frames = []
    avg_velocities = []

    for frame_idx, frame in enumerate(frames.frames):
        x, y = frame.x, frame.y
        X, Y = np.meshgrid(x, y)

        u = frame.fields.get("u", np.zeros_like(X))
        v = frame.fields.get("v", np.zeros_like(X))
        p = frame.fields.get("p", np.zeros_like(X))
        vel_mag = frame.fields.get("velocity_mag", np.sqrt(u**2 + v**2))
        vorticity = frame.fields.get("vorticity", np.zeros_like(X))
        T = frame.fields.get("T", np.ones_like(X) * 300)

        avg_velocities.append(float(np.mean(vel_mag)))

        frame_data = []

        # 1. Velocity Magnitude
        frame_data.append(
            go.Heatmap(z=vel_mag, x=x, y=y, colorscale="Viridis", showscale=False)
        )

        # 2. Pressure
        frame_data.append(go.Heatmap(z=p, x=x, y=y, colorscale="RdBu", showscale=False))

        # 3. Vorticity
        frame_data.append(
            go.Heatmap(z=vorticity, x=x, y=y, colorscale="RdBu", showscale=False)
        )

        # 4. Vector Field
        skip = max(1, min(len(x), len(y)) // 10)
        X_sub = X[::skip, ::skip]
        Y_sub = Y[::skip, ::skip]
        speed_sub = vel_mag[::skip, ::skip]

        frame_data.append(
            go.Scatter(
                x=X_sub.flatten(),
                y=Y_sub.flatten(),
                mode="markers",
                marker=dict(
                    size=8,
                    color=speed_sub.flatten(),
                    colorscale="Viridis",
                    showscale=False,
                ),
                showlegend=False,
            )
        )

        # 5. 3D Flow
        frame_data.append(
            go.Scatter3d(
                x=X.flatten(),
                y=Y.flatten(),
                z=vel_mag.flatten(),
                mode="markers",
                marker=dict(
                    size=2,
                    color=vel_mag.flatten(),
                    colorscale="Viridis",
                    showscale=False,
                ),
                showlegend=False,
            )
        )

        # 6. Temperature
        frame_data.append(
            go.Heatmap(z=T, x=x, y=y, colorscale="Plasma", showscale=False)
        )

        # 7. Combined
        frame_data.append(
            go.Heatmap(
                z=vel_mag, x=x, y=y, colorscale="Viridis", opacity=0.7, showscale=False
            )
        )

        # 8. Time Series
        frame_data.append(
            go.Scatter(
                x=frames.time_indices[: frame_idx + 1],
                y=avg_velocities[: frame_idx + 1],
                mode="lines",
                showlegend=False,
            )
        )

        # 9. 3D Surface
        frame_data.append(
            go.Surface(z=vel_mag, x=x, y=y, colorscale="Viridis", showscale=False)
        )

        plotly_frames.append(go.Frame(data=frame_data, name=str(frame_idx)))

    # Add initial traces (first frame)
    first_frame = frames.frames[0]
    x, y = first_frame.x, first_frame.y
    X, Y = np.meshgrid(x, y)

    u = first_frame.fields.get("u", np.zeros_like(X))
    v = first_frame.fields.get("v", np.zeros_like(X))
    p = first_frame.fields.get("p", np.zeros_like(X))
    vel_mag = first_frame.fields.get("velocity_mag", np.sqrt(u**2 + v**2))
    vorticity = first_frame.fields.get("vorticity", np.zeros_like(X))
    T = first_frame.fields.get("T", np.ones_like(X) * 300)

    # Row 1
    fig.add_trace(go.Heatmap(z=vel_mag, x=x, y=y, colorscale="Viridis"), row=1, col=1)
    fig.add_trace(go.Heatmap(z=p, x=x, y=y, colorscale="RdBu"), row=1, col=2)
    fig.add_trace(go.Heatmap(z=vorticity, x=x, y=y, colorscale="RdBu"), row=1, col=3)

    # Row 2
    skip = max(1, min(len(x), len(y)) // 10)
    X_sub = X[::skip, ::skip]
    Y_sub = Y[::skip, ::skip]
    speed_sub = vel_mag[::skip, ::skip]

    fig.add_trace(
        go.Scatter(
            x=X_sub.flatten(),
            y=Y_sub.flatten(),
            mode="markers",
            marker=dict(size=8, color=speed_sub.flatten(), colorscale="Viridis"),
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter3d(
            x=X.flatten(),
            y=Y.flatten(),
            z=vel_mag.flatten(),
            mode="markers",
            marker=dict(size=2, color=vel_mag.flatten(), colorscale="Viridis"),
        ),
        row=2,
        col=2,
    )

    fig.add_trace(go.Heatmap(z=T, x=x, y=y, colorscale="Plasma"), row=2, col=3)

    # Row 3
    fig.add_trace(
        go.Heatmap(z=vel_mag, x=x, y=y, colorscale="Viridis", opacity=0.7),
        row=3,
        col=1,
    )

    fig.add_trace(
        go.Scatter(x=frames.time_indices, y=avg_velocities, mode="lines"),
        row=3,
        col=2,
    )

    fig.add_trace(
        go.Surface(z=vel_mag, x=x, y=y, colorscale="Viridis"),
        row=3,
        col=3,
    )

    # Add frames and animation controls
    fig.frames = plotly_frames

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=20)),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=500, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=300),
                            ),
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=False),
                                mode="immediate",
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                ],
                x=0.1,
                y=0.02,
            )
        ],
        sliders=[
            dict(
                active=0,
                yanchor="top",
                xanchor="left",
                currentvalue=dict(
                    font=dict(size=16),
                    prefix="Iteration: ",
                    visible=True,
                    xanchor="right",
                ),
                transition=dict(duration=300),
                pad=dict(b=10, t=50),
                len=0.9,
                x=0.1,
                y=0,
                steps=[
                    dict(
                        args=[
                            [frame.name],
                            dict(
                                frame=dict(duration=300, redraw=True),
                                mode="immediate",
                                transition=dict(duration=300),
                            ),
                        ],
                        label=str(frames.time_indices[i]),
                        method="animate",
                    )
                    for i, frame in enumerate(plotly_frames)
                ],
            )
        ],
        height=height,
        width=width,
        showlegend=False,
    )

    return fig
