#!/usr/bin/env python3
"""Animation Creation Example.

This example demonstrates animation capabilities:

1. Creating animation frame data from simulation results
2. Creating field animations (velocity, vorticity)
3. Creating streamline animations
4. Creating vector field animations
5. Creating multi-panel animations
6. Saving animations to different formats

Usage:
    python examples/animation_creation.py

Requirements:
    - cfd_python package (pip install -e ../cfd-python)
    - cfd_viz package
    - ffmpeg (for MP4 export, optional)
"""

import sys

import matplotlib.pyplot as plt
import numpy as np

# Check for cfd_python
try:
    import cfd_python
except ImportError:
    print("Error: cfd_python package not installed.")
    print("Install with: pip install -e ../cfd-python")
    sys.exit(1)

from cfd_viz.animation import (
    create_animation_frames,
    create_cfd_colormap,
    create_field_animation,
    create_frame_data,
    create_multi_panel_animation,
    create_streamline_animation,
    create_vector_animation,
    export_frame_to_image,
    save_animation,
)


def generate_time_series_data():
    """Generate a time series of flow data by running simulation snapshots."""
    print("Generating time series data...")

    nx, ny = 50, 50
    xmin, xmax = 0.0, 1.0
    ymin, ymax = 0.0, 1.0

    # Create coordinate arrays
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y)
    dx = (xmax - xmin) / (nx - 1)
    dy = (ymax - ymin) / (ny - 1)

    frames = []

    # Generate frames by running simulation with increasing steps
    # This simulates time evolution
    step_counts = [50, 100, 150, 200, 250, 300, 350, 400]

    for i, steps in enumerate(step_counts):
        print(f"  Generating frame {i + 1}/{len(step_counts)} (steps={steps})...")

        result = cfd_python.run_simulation_with_params(
            nx=nx,
            ny=ny,
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            steps=steps,
        )

        # Extract velocity fields
        u = np.array(result["u"]).reshape((ny, nx))
        v = np.array(result["v"]).reshape((ny, nx))

        # Get pressure if available
        p = None
        if result.get("p"):
            p = np.array(result["p"]).reshape((ny, nx))
        else:
            # Create pressure approximation
            p = np.zeros((ny, nx))

        frames.append((X, Y, u, v, p))

    print(f"Generated {len(frames)} frames")
    return frames, dx, dy


def create_frame_data_demo(frames):
    """Demonstrate creating FrameData objects."""
    print("\n1. Creating Frame Data")
    print("-" * 40)

    # Create individual FrameData
    X, Y, u, v, p = frames[0]
    frame_data = create_frame_data(
        X=X,
        Y=Y,
        u=u,
        v=v,
        p=p,
        time=0.0,
    )

    print("Created FrameData:")
    print(f"  Grid size: {frame_data.nx} x {frame_data.ny}")
    print(f"  Velocity magnitude range: {frame_data.velocity_mag.min():.4f} - "
          f"{frame_data.velocity_mag.max():.4f}")

    return frame_data


def create_animation_frames_demo(frames, dx, dy):
    """Demonstrate creating AnimationFrames collection."""
    print("\n2. Creating Animation Frames Collection")
    print("-" * 40)

    # Create time array
    times = np.linspace(0, 1, len(frames))

    # Create AnimationFrames from list of tuples
    animation_frames = create_animation_frames(
        frames_data=frames,
        times=times,
        dx=dx,
        dy=dy,
    )

    print("Created AnimationFrames:")
    print(f"  Number of frames: {animation_frames.num_frames}")
    print(f"  Time range: {animation_frames.time_start:.2f} - "
          f"{animation_frames.time_end:.2f}")

    return animation_frames


def create_field_animation_demo(animation_frames):
    """Demonstrate field animation creation."""
    print("\n3. Creating Field Animation (Velocity Magnitude)")
    print("-" * 40)

    fig, anim = create_field_animation(
        animation_frames,
        field="velocity_mag",
        cmap="viridis",
        title="Velocity Magnitude Evolution",
        interval=200,  # 200ms between frames
    )

    print("Created velocity magnitude animation")
    return fig, anim


def create_vorticity_animation_demo(animation_frames):
    """Demonstrate vorticity animation creation."""
    print("\n4. Creating Vorticity Animation")
    print("-" * 40)

    fig, anim = create_field_animation(
        animation_frames,
        field="vorticity",
        cmap="RdBu_r",
        title="Vorticity Evolution",
        interval=200,
    )

    print("Created vorticity animation")
    return fig, anim


def create_streamline_animation_demo(animation_frames):
    """Demonstrate streamline animation."""
    print("\n5. Creating Streamline Animation")
    print("-" * 40)

    fig, anim = create_streamline_animation(
        animation_frames,
        density=1.5,
        color="velocity_mag",
        cmap="viridis",
        title="Streamline Evolution",
        interval=200,
    )

    print("Created streamline animation")
    return fig, anim


def create_vector_animation_demo(animation_frames):
    """Demonstrate vector field animation."""
    print("\n6. Creating Vector Field Animation")
    print("-" * 40)

    fig, anim = create_vector_animation(
        animation_frames,
        skip=3,  # Show every 3rd vector for clarity
        scale=20,
        title="Velocity Vector Evolution",
        interval=200,
    )

    print("Created vector field animation")
    return fig, anim


def create_multi_panel_demo(animation_frames):
    """Demonstrate multi-panel animation."""
    print("\n7. Creating Multi-Panel Animation")
    print("-" * 40)

    # Create a 2x2 multi-panel animation
    panel_configs = [
        {"field": "velocity_mag", "cmap": "viridis", "title": "Velocity"},
        {"field": "vorticity", "cmap": "RdBu_r", "title": "Vorticity"},
        {"field": "u", "cmap": "coolwarm", "title": "U-velocity"},
        {"field": "v", "cmap": "coolwarm", "title": "V-velocity"},
    ]

    fig, anim = create_multi_panel_animation(
        animation_frames,
        panel_configs=panel_configs,
        nrows=2,
        ncols=2,
        figsize=(12, 10),
        interval=200,
    )

    print("Created multi-panel animation (2x2 layout)")
    return fig, anim


def export_single_frame_demo(animation_frames):
    """Demonstrate exporting a single frame."""
    print("\n8. Exporting Single Frame")
    print("-" * 40)

    # Export first frame
    frame = animation_frames.frames[0]
    output_file = "animation_frame_0.png"

    export_frame_to_image(
        frame,
        output_file,
        field="velocity_mag",
        cmap="viridis",
        dpi=150,
    )

    print(f"Exported frame to: {output_file}")


def save_animations_demo(anim_velocity, anim_multi):
    """Demonstrate saving animations to files."""
    print("\n9. Saving Animations")
    print("-" * 40)

    # Save velocity animation as GIF
    gif_file = "velocity_animation.gif"
    print(f"Saving {gif_file}...")
    save_animation(anim_velocity, gif_file, fps=5)
    print(f"  Saved: {gif_file}")

    # Save multi-panel as GIF
    multi_gif = "multi_panel_animation.gif"
    print(f"Saving {multi_gif}...")
    save_animation(anim_multi, multi_gif, fps=5)
    print(f"  Saved: {multi_gif}")

    print("\nNote: For MP4 export, ffmpeg must be installed:")
    print("  save_animation(anim, 'output.mp4', fps=10)")


def colormap_demo():
    """Demonstrate custom CFD colormaps."""
    print("\n10. Custom CFD Colormaps")
    print("-" * 40)

    cfd_cmap = create_cfd_colormap()
    print("Created CFD colormap (blue-cyan-green-yellow-red)")

    # Show colormap preview
    fig, ax = plt.subplots(figsize=(8, 1))
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(gradient, aspect="auto", cmap=cfd_cmap)
    ax.set_axis_off()
    ax.set_title("CFD Colormap")
    plt.savefig("cfd_colormap.png", dpi=100, bbox_inches="tight")
    plt.close()
    print("Saved colormap preview: cfd_colormap.png")


def main():
    """Run animation creation example."""
    print("Animation Creation Example")
    print("=" * 40)
    print()

    # Generate time series data
    frames, dx, dy = generate_time_series_data()

    # Create frame data demonstrations
    frame_data = create_frame_data_demo(frames)
    animation_frames = create_animation_frames_demo(frames, dx, dy)

    # Create various animation types
    fig_vel, anim_vel = create_field_animation_demo(animation_frames)
    fig_vort, anim_vort = create_vorticity_animation_demo(animation_frames)
    fig_stream, anim_stream = create_streamline_animation_demo(animation_frames)
    fig_vec, anim_vec = create_vector_animation_demo(animation_frames)
    fig_multi, anim_multi = create_multi_panel_demo(animation_frames)

    # Export and save
    export_single_frame_demo(animation_frames)
    save_animations_demo(anim_vel, anim_multi)
    colormap_demo()

    print("\n" + "=" * 40)
    print("Animation creation complete!")
    print()
    print("Key functions demonstrated:")
    print("  - create_frame_data(): Create single frame from arrays")
    print("  - create_animation_frames(): Create frame collection")
    print("  - create_field_animation(): Animate scalar fields")
    print("  - create_streamline_animation(): Animate streamlines")
    print("  - create_vector_animation(): Animate velocity vectors")
    print("  - create_multi_panel_animation(): Multi-panel dashboard")
    print("  - save_animation(): Export to GIF/MP4")
    print("  - export_frame_to_image(): Export single frame")
    print()
    print("Files created:")
    print("  - velocity_animation.gif")
    print("  - multi_panel_animation.gif")
    print("  - animation_frame_0.png")
    print("  - cfd_colormap.png")

    # Show the multi-panel animation
    plt.show()


if __name__ == "__main__":
    main()
