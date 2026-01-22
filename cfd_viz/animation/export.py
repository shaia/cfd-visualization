"""Frame Export Functions for CFD Animations.

Functions for exporting animation frames to image files and saving animations.
"""

from pathlib import Path
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.figure import Figure

from .frames import AnimationFrames, FrameData
from .renderers import create_velocity_colormap


def export_frame_to_image(
    frame: FrameData,
    output_path: Union[str, Path],
    figsize: Tuple[int, int] = (16, 12),
    dpi: int = 150,
    include_vectors: bool = True,
    include_streamlines: bool = True,
    include_pressure: bool = True,
) -> None:
    """Export a single frame to an image file.

    Creates a 2x2 panel visualization with:
    - Velocity magnitude contours
    - Velocity vectors (optional)
    - Streamlines (optional)
    - Pressure field or combined view

    Args:
        frame: FrameData to export.
        output_path: Path to save the image.
        figsize: Figure size.
        dpi: Image resolution.
        include_vectors: Whether to include vector subplot.
        include_streamlines: Whether to include streamlines subplot.
        include_pressure: Whether to include pressure subplot.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    X, Y = frame.X, frame.Y
    u = frame.fields.get("u")
    v = frame.fields.get("v")
    p = frame.fields.get("p")
    velocity_mag = frame.fields.get("velocity_mag")

    if velocity_mag is None and u is not None and v is not None:
        velocity_mag = np.sqrt(u**2 + v**2)

    # Panel 1: Velocity magnitude contours
    if velocity_mag is not None:
        im1 = axes[0].contourf(X, Y, velocity_mag, levels=20, cmap="viridis")
        axes[0].set_title("Velocity Magnitude")
        axes[0].axis("equal")
        plt.colorbar(im1, ax=axes[0])
    else:
        axes[0].text(0.5, 0.5, "No velocity data", ha="center", va="center")
        axes[0].set_title("Velocity Magnitude")

    # Panel 2: Velocity vectors
    if include_vectors and u is not None and v is not None:
        subsample = max(1, min(X.shape[0], X.shape[1]) // 20)
        X_sub = X[::subsample, ::subsample]
        Y_sub = Y[::subsample, ::subsample]
        u_sub = u[::subsample, ::subsample]
        v_sub = v[::subsample, ::subsample]

        axes[1].quiver(
            X_sub,
            Y_sub,
            u_sub,
            v_sub,
            np.sqrt(u_sub**2 + v_sub**2),
            cmap="plasma",
            scale_units="xy",
            angles="xy",
        )
        axes[1].set_title("Velocity Vectors")
        axes[1].axis("equal")
    else:
        axes[1].axis("off")

    # Panel 3: Streamlines
    if include_streamlines and u is not None and v is not None:
        try:
            if velocity_mag is not None:
                axes[2].streamplot(
                    X, Y, u, v, color=velocity_mag, cmap="viridis", density=2
                )
            else:
                axes[2].streamplot(X, Y, u, v, density=2)
        except ValueError:
            # Streamplot can fail with certain grid configurations
            if velocity_mag is not None:
                axes[2].contourf(X, Y, velocity_mag, levels=20, cmap="viridis")
        axes[2].set_title("Flow Streamlines")
        axes[2].axis("equal")
    else:
        axes[2].axis("off")

    # Panel 4: Pressure or combined view
    if include_pressure and p is not None:
        im4 = axes[3].contourf(X, Y, p, levels=20, cmap="RdBu_r")
        axes[3].set_title("Pressure Field")
        axes[3].axis("equal")
        plt.colorbar(im4, ax=axes[3])
    elif velocity_mag is not None and u is not None and v is not None:
        im4 = axes[3].contourf(X, Y, velocity_mag, levels=20, cmap="viridis", alpha=0.7)
        axes[3].streamplot(X, Y, u, v, color="white", density=1, linewidth=0.8)
        axes[3].set_title("Combined: Magnitude + Streamlines")
        plt.colorbar(im4, ax=axes[3])
    else:
        axes[3].axis("off")

    fig.suptitle(f"CFD Flow Visualization - Step {frame.time_index}", fontsize=16)
    plt.tight_layout()

    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def export_animation_frames(
    animation_frames: AnimationFrames,
    output_dir: Union[str, Path],
    prefix: str = "frame",
    figsize: Tuple[int, int] = (16, 12),
    dpi: int = 150,
) -> List[str]:
    """Export all frames from AnimationFrames to image files.

    Args:
        animation_frames: AnimationFrames to export.
        output_dir: Directory to save images.
        prefix: Filename prefix for exported images.
        figsize: Figure size.
        dpi: Image resolution.

    Returns:
        List of exported file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exported_files = []
    for i, frame in enumerate(animation_frames.frames):
        filename = output_dir / f"{prefix}_{frame.time_index:04d}.png"
        export_frame_to_image(frame, filename, figsize=figsize, dpi=dpi)
        exported_files.append(str(filename))

        if (i + 1) % 10 == 0:
            print(f"Exported {i + 1}/{len(animation_frames)} frames")

    return exported_files


def save_animation(
    anim: animation.FuncAnimation,
    output_path: Union[str, Path],
    writer: str = "pillow",
    fps: int = 5,
    dpi: int = 100,
) -> None:
    """Save a matplotlib animation to file.

    Args:
        anim: The FuncAnimation object to save.
        output_path: Path to save the animation (e.g., "output.gif" or "output.mp4").
        writer: Animation writer ("pillow" for GIF, "ffmpeg" for MP4).
        fps: Frames per second.
        dpi: Resolution.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    anim.save(str(output_path), writer=writer, fps=fps, dpi=dpi)


def create_comprehensive_frame_figure(
    frame: FrameData,
    figsize: Tuple[int, int] = (18, 10),
) -> Figure:
    """Create a comprehensive 2x3 figure for a single frame.

    Args:
        frame: FrameData to visualize.
        figsize: Figure size.

    Returns:
        Matplotlib Figure object.
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    X, Y = frame.X, frame.Y
    u = frame.fields.get("u", np.zeros_like(X))
    v = frame.fields.get("v", np.zeros_like(X))
    p = frame.fields.get("p", np.zeros_like(X))
    velocity_mag = frame.fields.get("velocity_mag", np.sqrt(u**2 + v**2))
    vorticity = frame.fields.get("vorticity", np.zeros_like(X))

    velocity_cmap = create_velocity_colormap()

    # Velocity magnitude
    im0 = axes[0].contourf(X, Y, velocity_mag, levels=20, cmap=velocity_cmap)
    axes[0].set_title("Velocity Magnitude", fontweight="bold")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")
    axes[0].set_aspect("equal")
    plt.colorbar(im0, ax=axes[0])

    # Pressure
    im1 = axes[1].contourf(X, Y, p, levels=20, cmap="RdBu_r")
    contours = axes[1].contour(X, Y, p, levels=8, colors="black", linewidths=0.5)
    axes[1].clabel(contours, inline=True, fontsize=8)
    axes[1].set_title("Pressure Field", fontweight="bold")
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Y")
    axes[1].set_aspect("equal")
    plt.colorbar(im1, ax=axes[1])

    # Vorticity
    im2 = axes[2].contourf(X, Y, vorticity, levels=20, cmap="seismic")
    axes[2].set_title("Vorticity", fontweight="bold")
    axes[2].set_xlabel("X")
    axes[2].set_ylabel("Y")
    axes[2].set_aspect("equal")
    plt.colorbar(im2, ax=axes[2])

    # Vector field
    skip = max(1, min(X.shape[0], X.shape[1]) // 15)
    X_sub = X[::skip, ::skip]
    Y_sub = Y[::skip, ::skip]
    u_sub = u[::skip, ::skip]
    v_sub = v[::skip, ::skip]
    vel_sub = velocity_mag[::skip, ::skip]

    axes[3].quiver(
        X_sub, Y_sub, u_sub, v_sub, vel_sub, cmap=velocity_cmap, scale=15, width=0.003
    )
    axes[3].set_xlim(X.min(), X.max())
    axes[3].set_ylim(Y.min(), Y.max())
    axes[3].set_title("Vector Field", fontweight="bold")
    axes[3].set_xlabel("X")
    axes[3].set_ylabel("Y")
    axes[3].set_aspect("equal")

    # Streamlines
    axes[4].streamplot(
        X, Y, u, v, color=velocity_mag, cmap=velocity_cmap, density=2, linewidth=1.5
    )
    axes[4].set_xlim(X.min(), X.max())
    axes[4].set_ylim(Y.min(), Y.max())
    axes[4].set_title("Streamlines", fontweight="bold")
    axes[4].set_xlabel("X")
    axes[4].set_ylabel("Y")
    axes[4].set_aspect("equal")

    # Combined analysis
    axes[5].contourf(X, Y, velocity_mag, levels=20, cmap=velocity_cmap, alpha=0.8)
    axes[5].contour(X, Y, p, levels=6, colors="white", linewidths=1.5)
    if np.any(vorticity != 0):
        vort_threshold = np.percentile(np.abs(vorticity), 85)
        high_vort_mask = np.abs(vorticity) > vort_threshold
        axes[5].contour(X, Y, high_vort_mask, levels=[0.5], colors="red", linewidths=2)
    axes[5].set_xlim(X.min(), X.max())
    axes[5].set_ylim(Y.min(), Y.max())
    axes[5].set_title("Combined Analysis", fontweight="bold")
    axes[5].set_xlabel("X")
    axes[5].set_ylabel("Y")
    axes[5].set_aspect("equal")

    fig.suptitle(
        f"CFD Flow Analysis - Step {frame.time_index}", fontsize=16, fontweight="bold"
    )
    plt.tight_layout()

    return fig
