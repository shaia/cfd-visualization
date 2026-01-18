"""Animation Rendering Functions.

Pure functions for creating matplotlib animations from frame data.
These functions accept FrameData/AnimationFrames and return
matplotlib Animation objects.
"""

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from .frames import AnimationFrames, FrameData, ParticleTraces

# =============================================================================
# Colormaps
# =============================================================================


def create_cfd_colormap() -> LinearSegmentedColormap:
    """Create a custom colormap for CFD visualization."""
    colors = [
        "#000080",
        "#0000FF",
        "#00FFFF",
        "#FFFF00",
        "#FF8000",
        "#FF0000",
        "#800000",
    ]
    return LinearSegmentedColormap.from_list("cfd_custom", colors, N=256)


def create_velocity_colormap() -> LinearSegmentedColormap:
    """Create a velocity-specific colormap."""
    colors = ["#000080", "#0080FF", "#00FFFF", "#FFFF00", "#FF8000", "#FF0000"]
    return LinearSegmentedColormap.from_list("velocity", colors, N=256)


# =============================================================================
# Single Frame Rendering
# =============================================================================


def render_contour_frame(
    ax: Axes,
    frame: FrameData,
    field_name: str = "velocity_mag",
    levels: int = 20,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    colorbar: bool = True,
    title: Optional[str] = None,
) -> None:
    """Render a contour plot for a single frame.

    Args:
        ax: Matplotlib axes to render on.
        frame: Frame data to render.
        field_name: Name of field to plot.
        levels: Number of contour levels.
        cmap: Colormap name.
        vmin: Minimum value for colormap.
        vmax: Maximum value for colormap.
        colorbar: Whether to add colorbar.
        title: Plot title.
    """
    ax.clear()

    field = frame.fields.get(field_name)
    if field is None:
        ax.text(0.5, 0.5, f"Field '{field_name}' not found", ha="center", va="center")
        return

    cs = ax.contourf(
        frame.X,
        frame.Y,
        field,
        levels=levels,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    if colorbar:
        plt.colorbar(cs, ax=ax, label=field_name.replace("_", " ").title())

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal")

    if title:
        ax.set_title(title)
    else:
        ax.set_title(
            f"{field_name.replace('_', ' ').title()} - Step {frame.time_index}"
        )


def render_vector_frame(
    ax: Axes,
    frame: FrameData,
    subsample: int = 5,
    color_by: str = "velocity_mag",
    cmap: str = "viridis",
    scale: float = 15,
    width: float = 0.003,
    title: Optional[str] = None,
) -> None:
    """Render a vector field for a single frame.

    Args:
        ax: Matplotlib axes to render on.
        frame: Frame data to render.
        subsample: Subsampling step for vectors.
        color_by: Field name to use for coloring vectors.
        cmap: Colormap name.
        scale: Arrow scale factor.
        width: Arrow width.
        title: Plot title.
    """
    ax.clear()

    u = frame.fields.get("u")
    v = frame.fields.get("v")
    if u is None or v is None:
        ax.text(0.5, 0.5, "Velocity fields not found", ha="center", va="center")
        return

    X_sub = frame.X[::subsample, ::subsample]
    Y_sub = frame.Y[::subsample, ::subsample]
    u_sub = u[::subsample, ::subsample]
    v_sub = v[::subsample, ::subsample]

    color_field = frame.fields.get(color_by)
    if color_field is not None:
        c_sub = color_field[::subsample, ::subsample]
    else:
        c_sub = np.sqrt(u_sub**2 + v_sub**2)

    ax.quiver(
        X_sub,
        Y_sub,
        u_sub,
        v_sub,
        c_sub,
        cmap=cmap,
        scale=scale,
        width=width,
    )

    ax.set_xlim(frame.X.min(), frame.X.max())
    ax.set_ylim(frame.Y.min(), frame.Y.max())
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal")

    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Vector Field - Step {frame.time_index}")


def render_streamline_frame(
    ax: Axes,
    frame: FrameData,
    color_by: str = "velocity_mag",
    cmap: str = "viridis",
    density: float = 2.0,
    linewidth: float = 1.0,
    title: Optional[str] = None,
) -> None:
    """Render streamlines for a single frame.

    Args:
        ax: Matplotlib axes to render on.
        frame: Frame data to render.
        color_by: Field name to use for coloring streamlines.
        cmap: Colormap name.
        density: Streamline density.
        linewidth: Line width.
        title: Plot title.
    """
    ax.clear()

    u = frame.fields.get("u")
    v = frame.fields.get("v")
    if u is None or v is None:
        ax.text(0.5, 0.5, "Velocity fields not found", ha="center", va="center")
        return

    color_field = frame.fields.get(color_by)
    if color_field is None:
        color_field = np.sqrt(u**2 + v**2)

    ax.streamplot(
        frame.X,
        frame.Y,
        u,
        v,
        color=color_field,
        cmap=cmap,
        density=density,
        linewidth=linewidth,
    )

    ax.set_xlim(frame.X.min(), frame.X.max())
    ax.set_ylim(frame.Y.min(), frame.Y.max())
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal")

    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Streamlines - Step {frame.time_index}")


# =============================================================================
# Animation Creation
# =============================================================================


def create_field_animation(
    animation_frames: AnimationFrames,
    field_name: str = "velocity_mag",
    figsize: Tuple[int, int] = (12, 6),
    cmap: str = "viridis",
    levels: int = 20,
    interval: int = 200,
    title_prefix: str = "CFD Simulation",
) -> Tuple[Figure, animation.FuncAnimation]:
    """Create animation of a scalar field.

    Args:
        animation_frames: AnimationFrames containing data.
        field_name: Field to animate.
        figsize: Figure size.
        cmap: Colormap name.
        levels: Number of contour levels.
        interval: Animation interval in milliseconds.
        title_prefix: Prefix for animation title.

    Returns:
        Tuple of (figure, animation) objects.
    """
    if not animation_frames.frames:
        raise ValueError("No frames to animate")

    fig, ax = plt.subplots(figsize=figsize)

    # Get consistent color range
    vmin, vmax = animation_frames.get_field_range(field_name)

    # Initial frame
    first_frame = animation_frames.frames[0]
    field_data = first_frame.fields.get(field_name)
    if field_data is None:
        raise ValueError(f"Field '{field_name}' not found in frames")

    im = ax.imshow(
        field_data,
        extent=[
            first_frame.X.min(),
            first_frame.X.max(),
            first_frame.Y.min(),
            first_frame.Y.max(),
        ],
        origin="lower",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(field_name.replace("_", " ").title())

    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    time_text = ax.text(
        0.02,
        0.95,
        "",
        transform=ax.transAxes,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    def animate(frame_idx: int) -> List:
        frame = animation_frames.frames[frame_idx]
        field_data = frame.fields.get(field_name)
        if field_data is not None:
            im.set_array(field_data)
        time_text.set_text(f"Step: {frame.time_index}")
        ax.set_title(f"{title_prefix} - {field_name.replace('_', ' ').title()}")
        return [im, time_text]

    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=len(animation_frames),
        interval=interval,
        blit=True,
        repeat=True,
    )

    return fig, anim


def create_streamline_animation(
    animation_frames: AnimationFrames,
    figsize: Tuple[int, int] = (12, 6),
    cmap: str = "viridis",
    density: float = 1.5,
    interval: int = 200,
    title_prefix: str = "CFD Streamlines",
) -> Tuple[Figure, animation.FuncAnimation]:
    """Create streamline animation.

    Args:
        animation_frames: AnimationFrames containing velocity data.
        figsize: Figure size.
        cmap: Colormap name.
        density: Streamline density.
        interval: Animation interval in milliseconds.
        title_prefix: Prefix for animation title.

    Returns:
        Tuple of (figure, animation) objects.
    """
    if not animation_frames.frames:
        raise ValueError("No frames to animate")

    fig, ax = plt.subplots(figsize=figsize)

    first_frame = animation_frames.frames[0]
    X, Y = first_frame.X, first_frame.Y

    # Get 1D coordinate arrays for streamplot
    x = X[0, :] if len(X.shape) == 2 else X
    y = Y[:, 0] if len(Y.shape) == 2 else Y

    def animate(frame_idx: int) -> List:
        ax.clear()
        frame = animation_frames.frames[frame_idx]

        u = frame.fields.get("u")
        v = frame.fields.get("v")
        if u is None or v is None:
            return []

        # Ensure correct shape for streamplot
        if u.shape == (len(x), len(y)):
            u = u.T
            v = v.T

        speed = np.sqrt(u**2 + v**2)

        ax.streamplot(
            x,
            y,
            u,
            v,
            color=speed,
            cmap=cmap,
            density=density,
            linewidth=0.5,
        )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"{title_prefix} - Step: {frame.time_index}")
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())

        return []

    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=len(animation_frames),
        interval=interval,
        repeat=True,
    )

    return fig, anim


def create_vector_animation(
    animation_frames: AnimationFrames,
    figsize: Tuple[int, int] = (12, 6),
    cmap: str = "viridis",
    subsample: int = 5,
    scale: float = 15,
    interval: int = 200,
    title_prefix: str = "CFD Vector Field",
) -> Tuple[Figure, animation.FuncAnimation]:
    """Create vector field animation.

    Args:
        animation_frames: AnimationFrames containing velocity data.
        figsize: Figure size.
        cmap: Colormap name.
        subsample: Subsampling step for vectors.
        scale: Arrow scale factor.
        interval: Animation interval in milliseconds.
        title_prefix: Prefix for animation title.

    Returns:
        Tuple of (figure, animation) objects.
    """
    if not animation_frames.frames:
        raise ValueError("No frames to animate")

    fig, ax = plt.subplots(figsize=figsize)

    first_frame = animation_frames.frames[0]
    X, Y = first_frame.X, first_frame.Y

    def animate(frame_idx: int) -> List:
        ax.clear()
        frame = animation_frames.frames[frame_idx]

        u = frame.fields.get("u")
        v = frame.fields.get("v")
        if u is None or v is None:
            return []

        X_sub = frame.X[::subsample, ::subsample]
        Y_sub = frame.Y[::subsample, ::subsample]
        u_sub = u[::subsample, ::subsample]
        v_sub = v[::subsample, ::subsample]
        vel_sub = np.sqrt(u_sub**2 + v_sub**2)

        ax.quiver(
            X_sub,
            Y_sub,
            u_sub,
            v_sub,
            vel_sub,
            cmap=cmap,
            scale=scale,
            width=0.003,
        )

        ax.set_xlim(X.min(), X.max())
        ax.set_ylim(Y.min(), Y.max())
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"{title_prefix} - Step: {frame.time_index}")

        return []

    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=len(animation_frames),
        interval=interval,
        repeat=True,
    )

    return fig, anim


def create_multi_panel_animation(
    animation_frames: AnimationFrames,
    figsize: Tuple[int, int] = (18, 10),
    interval: int = 500,
    title: str = "CFD Flow Analysis Dashboard",
) -> Tuple[Figure, animation.FuncAnimation]:
    """Create multi-panel animation showing multiple views.

    Creates a 2x3 grid with:
    - Velocity magnitude, Pressure, Vorticity (top row)
    - Vector field, Streamlines, Combined analysis (bottom row)

    Args:
        animation_frames: AnimationFrames containing all field data.
        figsize: Figure size.
        interval: Animation interval in milliseconds.
        title: Figure title.

    Returns:
        Tuple of (figure, animation) objects.
    """
    if not animation_frames.frames:
        raise ValueError("No frames to animate")

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight="bold")
    axes = axes.flatten()

    # Get color ranges
    vel_range = animation_frames.get_field_range("velocity_mag")
    p_range = animation_frames.get_field_range("p")
    vort_range = animation_frames.get_field_range("vorticity")

    velocity_cmap = create_velocity_colormap()

    first_frame = animation_frames.frames[0]
    X, Y = first_frame.X, first_frame.Y

    def animate(frame_idx: int) -> List:
        for ax in axes:
            ax.clear()

        frame = animation_frames.frames[frame_idx]
        u = frame.fields.get("u", np.zeros_like(X))
        v = frame.fields.get("v", np.zeros_like(X))
        p = frame.fields.get("p", np.zeros_like(X))
        vel_mag = frame.fields.get("velocity_mag", np.sqrt(u**2 + v**2))
        vorticity = frame.fields.get("vorticity", np.zeros_like(X))

        # Velocity magnitude
        axes[0].imshow(
            vel_mag,
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            origin="lower",
            aspect="auto",
            vmin=vel_range[0],
            vmax=vel_range[1],
            cmap=velocity_cmap,
        )
        axes[0].set_title("Velocity Magnitude", fontweight="bold")
        axes[0].set_xlabel("X")
        axes[0].set_ylabel("Y")

        # Pressure
        axes[1].imshow(
            p,
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            origin="lower",
            aspect="auto",
            vmin=p_range[0],
            vmax=p_range[1],
            cmap="RdBu_r",
        )
        contours = axes[1].contour(X, Y, p, levels=8, colors="black", linewidths=0.5)
        axes[1].clabel(contours, inline=True, fontsize=8)
        axes[1].set_title("Pressure Field", fontweight="bold")
        axes[1].set_xlabel("X")
        axes[1].set_ylabel("Y")

        # Vorticity
        axes[2].imshow(
            vorticity,
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            origin="lower",
            aspect="auto",
            vmin=vort_range[0],
            vmax=vort_range[1],
            cmap="seismic",
        )
        axes[2].set_title("Vorticity", fontweight="bold")
        axes[2].set_xlabel("X")
        axes[2].set_ylabel("Y")

        # Vector field
        skip = 5
        X_sub = X[::skip, ::skip]
        Y_sub = Y[::skip, ::skip]
        u_sub = u[::skip, ::skip]
        v_sub = v[::skip, ::skip]
        vel_sub = vel_mag[::skip, ::skip]

        axes[3].quiver(
            X_sub,
            Y_sub,
            u_sub,
            v_sub,
            vel_sub,
            cmap=velocity_cmap,
            scale=15,
            width=0.003,
        )
        axes[3].set_xlim(X.min(), X.max())
        axes[3].set_ylim(Y.min(), Y.max())
        axes[3].set_title("Vector Field", fontweight="bold")
        axes[3].set_xlabel("X")
        axes[3].set_ylabel("Y")

        # Streamlines
        axes[4].streamplot(
            X, Y, u, v, color=vel_mag, cmap=velocity_cmap, density=2, linewidth=1.5
        )
        axes[4].set_xlim(X.min(), X.max())
        axes[4].set_ylim(Y.min(), Y.max())
        axes[4].set_title("Streamlines", fontweight="bold")
        axes[4].set_xlabel("X")
        axes[4].set_ylabel("Y")

        # Combined analysis
        axes[5].imshow(
            vel_mag,
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            origin="lower",
            aspect="auto",
            vmin=vel_range[0],
            vmax=vel_range[1],
            cmap=velocity_cmap,
            alpha=0.8,
        )
        axes[5].contour(X, Y, p, levels=6, colors="white", linewidths=1.5)
        vort_threshold = np.percentile(np.abs(vorticity), 85)
        high_vort_mask = np.abs(vorticity) > vort_threshold
        axes[5].contour(X, Y, high_vort_mask, levels=[0.5], colors="red", linewidths=2)
        axes[5].set_xlim(X.min(), X.max())
        axes[5].set_ylim(Y.min(), Y.max())
        axes[5].set_title("Combined Analysis", fontweight="bold")
        axes[5].set_xlabel("X")
        axes[5].set_ylabel("Y")

        fig.suptitle(
            f"{title} - Step: {frame.time_index}",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()

        return []

    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=len(animation_frames),
        interval=interval,
        repeat=True,
    )

    return fig, anim


def create_particle_trace_animation(
    animation_frames: AnimationFrames,
    particle_traces: ParticleTraces,
    figsize: Tuple[int, int] = (15, 8),
    interval: int = 100,
    title_prefix: str = "Particle Traces",
) -> Tuple[Figure, animation.FuncAnimation]:
    """Create Lagrangian particle trace animation.

    Args:
        animation_frames: AnimationFrames containing velocity data.
        particle_traces: ParticleTraces with precomputed positions.
        figsize: Figure size.
        interval: Animation interval in milliseconds.
        title_prefix: Prefix for animation title.

    Returns:
        Tuple of (figure, animation) objects.
    """
    if not animation_frames.frames:
        raise ValueError("No frames to animate")

    fig, ax = plt.subplots(figsize=figsize)

    first_frame = animation_frames.frames[0]
    X, Y = first_frame.X, first_frame.Y

    def animate(frame_idx: int) -> List:
        ax.clear()

        # Use frame index modulo number of frames for cycling
        frame = animation_frames.frames[frame_idx % len(animation_frames)]

        # Background velocity magnitude
        vel_mag = frame.fields.get("velocity_mag")
        if vel_mag is not None:
            ax.imshow(
                vel_mag,
                extent=[X.min(), X.max(), Y.min(), Y.max()],
                origin="lower",
                aspect="auto",
                cmap="Blues",
                alpha=0.6,
            )

        # Get particle positions for this frame
        if frame_idx < len(particle_traces.positions_x):
            px = particle_traces.positions_x[frame_idx]
            py = particle_traces.positions_y[frame_idx]

            # Draw particle traces
            max_hist = min(frame_idx + 1, particle_traces.max_history)
            for i in range(particle_traces.n_particles):
                if max_hist > 1:
                    start_idx = max(0, frame_idx - max_hist + 1)
                    x_hist = [
                        particle_traces.positions_x[j][i]
                        for j in range(start_idx, frame_idx + 1)
                    ]
                    y_hist = [
                        particle_traces.positions_y[j][i]
                        for j in range(start_idx, frame_idx + 1)
                    ]

                    if len(x_hist) > 1:
                        alphas = np.linspace(0.2, 1.0, len(x_hist))
                        for j in range(len(x_hist) - 1):
                            ax.plot(
                                [x_hist[j], x_hist[j + 1]],
                                [y_hist[j], y_hist[j + 1]],
                                "r-",
                                alpha=alphas[j],
                                linewidth=2,
                            )

            # Current particle positions
            ax.scatter(
                px,
                py,
                c="red",
                s=30,
                alpha=0.8,
                edgecolors="white",
                linewidth=1,
            )

        # Streamlines
        u = frame.fields.get("u")
        v = frame.fields.get("v")
        if u is not None and v is not None:
            ax.streamplot(
                frame.X, frame.Y, u, v, color="gray", density=1, linewidth=0.5
            )

        ax.set_xlim(X.min(), X.max())
        ax.set_ylim(Y.min(), Y.max())
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"{title_prefix} - Step: {frame.time_index}")

        return []

    # Animate through particle trace history
    n_frames = len(particle_traces.positions_x)
    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=n_frames,
        interval=interval,
        repeat=True,
    )

    return fig, anim


def create_vorticity_analysis_animation(
    animation_frames: AnimationFrames,
    figsize: Tuple[int, int] = (15, 6),
    interval: int = 400,
    title_prefix: str = "Vorticity Analysis",
) -> Tuple[Figure, animation.FuncAnimation]:
    """Create side-by-side vorticity field and streamlines animation.

    Args:
        animation_frames: AnimationFrames containing velocity data.
        figsize: Figure size.
        interval: Animation interval in milliseconds.
        title_prefix: Prefix for animation title.

    Returns:
        Tuple of (figure, animation) objects.
    """
    if not animation_frames.frames:
        raise ValueError("No frames to animate")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    vort_range = animation_frames.get_field_range("vorticity")
    first_frame = animation_frames.frames[0]
    X, Y = first_frame.X, first_frame.Y

    def animate(frame_idx: int) -> List:
        ax1.clear()
        ax2.clear()

        frame = animation_frames.frames[frame_idx]
        u = frame.fields.get("u")
        v = frame.fields.get("v")
        vorticity = frame.fields.get("vorticity")
        vel_mag = frame.fields.get("velocity_mag")

        if vorticity is None or u is None or v is None:
            return []

        # Vorticity field
        ax1.imshow(
            vorticity,
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            origin="lower",
            aspect="auto",
            vmin=vort_range[0],
            vmax=vort_range[1],
            cmap="RdBu",
        )
        ax1.set_title("Vorticity Field", fontweight="bold")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")

        # Streamlines colored by vorticity
        if vel_mag is not None:
            speed = vel_mag
        else:
            speed = np.sqrt(u**2 + v**2)

        lw = 5 * speed / speed.max() if speed.max() > 0 else 1

        ax2.streamplot(
            X,
            Y,
            u,
            v,
            color=vorticity,
            linewidth=lw,
            cmap="RdBu",
            density=2,
        )
        ax2.set_title("Streamlines Colored by Vorticity", fontweight="bold")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_xlim(X.min(), X.max())
        ax2.set_ylim(Y.min(), Y.max())

        fig.suptitle(
            f"{title_prefix} - Step: {frame.time_index}",
            fontsize=14,
            fontweight="bold",
        )

        return []

    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=len(animation_frames),
        interval=interval,
        repeat=True,
    )

    return fig, anim


def create_3d_surface_animation(
    animation_frames: AnimationFrames,
    field_name: str = "velocity_mag",
    figsize: Tuple[int, int] = (12, 8),
    interval: int = 200,
    rotate_camera: bool = True,
    title_prefix: str = "3D Surface",
) -> Tuple[Figure, animation.FuncAnimation]:
    """Create 3D rotating surface animation.

    Args:
        animation_frames: AnimationFrames containing field data.
        field_name: Field to plot as surface height.
        figsize: Figure size.
        interval: Animation interval in milliseconds.
        rotate_camera: Whether to rotate camera during animation.
        title_prefix: Prefix for animation title.

    Returns:
        Tuple of (figure, animation) objects.
    """
    if not animation_frames.frames:
        raise ValueError("No frames to animate")

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    vmin, vmax = animation_frames.get_field_range(field_name)
    first_frame = animation_frames.frames[0]
    X, Y = first_frame.X, first_frame.Y

    def animate(frame_idx: int) -> List:
        ax.clear()

        frame = animation_frames.frames[frame_idx]
        field = frame.fields.get(field_name)
        if field is None:
            return []

        ax.plot_surface(
            X,
            Y,
            field,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            alpha=0.8,
            linewidth=0,
            antialiased=True,
        )

        # Contour projection
        ax.contour(
            X,
            Y,
            field,
            zdir="z",
            offset=vmin - 0.1 * (vmax - vmin),
            cmap="viridis",
            alpha=0.5,
        )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel(field_name.replace("_", " ").title())
        ax.set_title(
            f"{title_prefix} - {field_name.replace('_', ' ').title()} - Step: {frame.time_index}"
        )
        ax.set_xlim(X.min(), X.max())
        ax.set_ylim(Y.min(), Y.max())
        ax.set_zlim(vmin, vmax)

        if rotate_camera:
            ax.view_init(elev=30, azim=frame_idx * 2)

        return []

    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=len(animation_frames),
        interval=interval,
        repeat=True,
    )

    return fig, anim
