"""Animation subpackage for CFD visualization.

This package provides pure functions for creating animations from CFD data.
Functions accept numpy arrays and dataclasses, returning matplotlib
Animation objects that can be saved or displayed.

Submodules:
    frames: Dataclasses and functions for frame data generation.
    renderers: Functions for creating matplotlib animations.
    export: Functions for exporting frames and saving animations.

Example:
    >>> from cfd_viz.animation import (
    ...     create_frame_data,
    ...     create_animation_frames,
    ...     create_field_animation,
    ...     save_animation,
    ... )
    >>>
    >>> # Create frame data from velocity/pressure arrays
    >>> frames_list = [(X, Y, u1, v1, p1), (X, Y, u2, v2, p2), ...]
    >>> animation_frames = create_animation_frames(frames_list)
    >>>
    >>> # Create and save animation
    >>> fig, anim = create_field_animation(animation_frames, "velocity_mag")
    >>> save_animation(anim, "velocity.gif", fps=5)
"""

from .export import (
    create_comprehensive_frame_figure,
    export_animation_frames,
    export_frame_to_image,
    save_animation,
)
from .frames import (
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
from .renderers import (
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

__all__ = [
    # Frame dataclasses
    "FrameData",
    "AnimationFrames",
    "ParticleTraces",
    # Frame functions
    "create_frame_data",
    "create_animation_frames",
    "subsample_frame",
    "compute_particle_positions",
    "initialize_particles",
    "advect_particles_through_frames",
    # Colormaps
    "create_cfd_colormap",
    "create_velocity_colormap",
    # Frame renderers
    "render_contour_frame",
    "render_vector_frame",
    "render_streamline_frame",
    # Animation creators
    "create_field_animation",
    "create_streamline_animation",
    "create_vector_animation",
    "create_multi_panel_animation",
    "create_particle_trace_animation",
    "create_vorticity_analysis_animation",
    "create_3d_surface_animation",
    # Export functions
    "export_frame_to_image",
    "export_animation_frames",
    "save_animation",
    "create_comprehensive_frame_figure",
]
