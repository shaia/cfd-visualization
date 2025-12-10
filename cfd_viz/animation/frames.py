"""Frame Generation for CFD Animations.

Pure functions for generating individual frame data from flow fields.
These functions accept numpy arrays and return data structures suitable
for animation rendering.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class FrameData:
    """Data for a single animation frame.

    Attributes:
        X: 2D array of x-coordinates (meshgrid).
        Y: 2D array of y-coordinates (meshgrid).
        fields: Dictionary of field name -> 2D array.
        time_index: Frame index or iteration number.
        timestamp: Optional simulation time.
    """

    X: NDArray
    Y: NDArray
    fields: Dict[str, NDArray]
    time_index: int
    timestamp: Optional[float] = None


@dataclass
class AnimationFrames:
    """Collection of frames for animation.

    Attributes:
        frames: List of FrameData objects.
        field_ranges: Dict mapping field names to (min, max) tuples for
            consistent colormapping across all frames.
    """

    frames: List[FrameData]
    field_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    def __len__(self) -> int:
        """Return number of frames."""
        return len(self.frames)

    def __getitem__(self, index: int) -> FrameData:
        """Get frame by index."""
        return self.frames[index]

    def get_field_range(self, field_name: str) -> Tuple[float, float]:
        """Get min/max range for a field across all frames.

        Args:
            field_name: Name of the field.

        Returns:
            Tuple of (min_value, max_value).
        """
        if field_name in self.field_ranges:
            return self.field_ranges[field_name]

        values = []
        for frame in self.frames:
            if field_name in frame.fields:
                values.append(frame.fields[field_name])

        if not values:
            return (0.0, 1.0)

        vmin = float(min(np.min(v) for v in values))
        vmax = float(max(np.max(v) for v in values))
        self.field_ranges[field_name] = (vmin, vmax)
        return (vmin, vmax)


def create_frame_data(
    X: NDArray,
    Y: NDArray,
    u: NDArray,
    v: NDArray,
    p: Optional[NDArray] = None,
    time_index: int = 0,
    timestamp: Optional[float] = None,
    compute_derived: bool = True,
) -> FrameData:
    """Create frame data from velocity and pressure fields.

    Args:
        X: 2D array of x-coordinates (meshgrid).
        Y: 2D array of y-coordinates (meshgrid).
        u: 2D array of x-velocity component.
        v: 2D array of y-velocity component.
        p: Optional 2D array of pressure.
        time_index: Frame index or iteration number.
        timestamp: Optional simulation time.
        compute_derived: Whether to compute derived fields (velocity_mag, vorticity).

    Returns:
        FrameData with all field data.
    """
    fields: Dict[str, NDArray] = {
        "u": u,
        "v": v,
    }

    if p is not None:
        fields["p"] = p

    if compute_derived:
        # Compute velocity magnitude
        fields["velocity_mag"] = np.sqrt(u**2 + v**2)

        # Compute vorticity (dv/dx - du/dy)
        # Use grid spacing from coordinates
        dx = X[0, 1] - X[0, 0] if X.shape[1] > 1 else 1.0
        dy = Y[1, 0] - Y[0, 0] if Y.shape[0] > 1 else 1.0
        dvdx = np.gradient(v, dx, axis=1)
        dudy = np.gradient(u, dy, axis=0)
        fields["vorticity"] = dvdx - dudy

    return FrameData(
        X=X,
        Y=Y,
        fields=fields,
        time_index=time_index,
        timestamp=timestamp,
    )


def create_animation_frames(
    frames_list: List[Tuple[NDArray, NDArray, NDArray, NDArray, Optional[NDArray]]],
    time_indices: Optional[List[int]] = None,
    timestamps: Optional[List[float]] = None,
    compute_derived: bool = True,
) -> AnimationFrames:
    """Create animation frames from a list of field data.

    Args:
        frames_list: List of (X, Y, u, v, p) tuples for each frame.
            Each tuple contains coordinate meshgrids and velocity/pressure arrays.
        time_indices: Optional list of time indices for each frame.
        timestamps: Optional list of simulation times for each frame.
        compute_derived: Whether to compute derived fields.

    Returns:
        AnimationFrames containing all frame data.
    """
    frames = []
    for i, frame_tuple in enumerate(frames_list):
        X, Y, u, v, p = frame_tuple

        time_idx = time_indices[i] if time_indices else i
        time_val = timestamps[i] if timestamps else None

        frame_data = create_frame_data(
            X=X,
            Y=Y,
            u=u,
            v=v,
            p=p,
            time_index=time_idx,
            timestamp=time_val,
            compute_derived=compute_derived,
        )
        frames.append(frame_data)

    return AnimationFrames(frames=frames)


def subsample_frame(
    frame: FrameData,
    step: int = 2,
    fields_to_subsample: Optional[List[str]] = None,
) -> FrameData:
    """Subsample frame data for vector field visualization.

    Args:
        frame: Original frame data.
        step: Subsampling step size.
        fields_to_subsample: List of field names to subsample (default: all).

    Returns:
        New FrameData with subsampled data.
    """
    if fields_to_subsample is None:
        fields_to_subsample = list(frame.fields.keys())

    subsampled_fields = {}
    for name, data in frame.fields.items():
        if name in fields_to_subsample:
            subsampled_fields[name] = data[::step, ::step]
        else:
            subsampled_fields[name] = data

    return FrameData(
        X=frame.X[::step, ::step],
        Y=frame.Y[::step, ::step],
        fields=subsampled_fields,
        time_index=frame.time_index,
        timestamp=frame.timestamp,
    )


def compute_particle_positions(
    X: NDArray,
    Y: NDArray,
    u: NDArray,
    v: NDArray,
    particles_x: NDArray,
    particles_y: NDArray,
    dt: float = 0.01,
) -> Tuple[NDArray, NDArray]:
    """Advect particles through velocity field using simple Euler integration.

    Args:
        X: 2D array of x-coordinates (meshgrid).
        Y: 2D array of y-coordinates (meshgrid).
        u: 2D array of x-velocity.
        v: 2D array of y-velocity.
        particles_x: Current x positions of particles.
        particles_y: Current y positions of particles.
        dt: Time step for advection.

    Returns:
        Tuple of (new_x, new_y) particle positions.
    """
    new_x = particles_x.copy()
    new_y = particles_y.copy()

    x_min, x_max = X.min(), X.max()
    y_min, y_max = Y.min(), Y.max()
    nx, ny = u.shape[1], u.shape[0]

    for i in range(len(particles_x)):
        px, py = particles_x[i], particles_y[i]

        # Check if particle is in domain
        if x_min <= px < x_max and y_min <= py < y_max:
            # Compute grid indices
            xi = int((px - x_min) / (x_max - x_min) * (nx - 1))
            yi = int((py - y_min) / (y_max - y_min) * (ny - 1))
            xi = max(0, min(xi, nx - 1))
            yi = max(0, min(yi, ny - 1))

            # Simple nearest-neighbor interpolation
            u_interp = u[yi, xi]
            v_interp = v[yi, xi]

            # Euler step
            new_x[i] = px + u_interp * dt
            new_y[i] = py + v_interp * dt

    return new_x, new_y


@dataclass
class ParticleTraces:
    """Particle trace data for Lagrangian visualization.

    Attributes:
        positions_x: List of x-position arrays, one per timestep.
        positions_y: List of y-position arrays, one per timestep.
        n_particles: Number of particles being tracked.
        max_history: Maximum history length per particle.
    """

    positions_x: List[NDArray]
    positions_y: List[NDArray]
    n_particles: int
    max_history: int = 20

    def get_current_positions(self) -> Tuple[NDArray, NDArray]:
        """Get most recent particle positions."""
        if not self.positions_x:
            return np.array([]), np.array([])
        return self.positions_x[-1], self.positions_y[-1]

    def get_particle_history(
        self, particle_idx: int
    ) -> Tuple[List[float], List[float]]:
        """Get position history for a single particle.

        Args:
            particle_idx: Index of particle.

        Returns:
            Tuple of (x_history, y_history) lists.
        """
        x_hist = [pos[particle_idx] for pos in self.positions_x[-self.max_history :]]
        y_hist = [pos[particle_idx] for pos in self.positions_y[-self.max_history :]]
        return x_hist, y_hist


def initialize_particles(
    X: NDArray,
    Y: NDArray,
    n_particles: int = 50,
    inlet_fraction: float = 0.1,
    seed: Optional[int] = None,
) -> Tuple[NDArray, NDArray]:
    """Initialize particle positions for Lagrangian tracking.

    Args:
        X: 2D array of x-coordinates (meshgrid).
        Y: 2D array of y-coordinates (meshgrid).
        n_particles: Number of particles to create.
        inlet_fraction: Fraction of domain width for inlet region.
        seed: Optional random seed for reproducibility.

    Returns:
        Tuple of (x_positions, y_positions) arrays.
    """
    if seed is not None:
        np.random.seed(seed)

    x_min, x_max = X.min(), X.max()
    y_min, y_max = Y.min(), Y.max()

    # Initialize particles near inlet (left side)
    inlet_width = inlet_fraction * (x_max - x_min)
    particles_x = np.random.uniform(x_min, x_min + inlet_width, n_particles)
    particles_y = np.random.uniform(y_min, y_max, n_particles)

    return particles_x, particles_y


def advect_particles_through_frames(
    animation_frames: AnimationFrames,
    n_particles: int = 50,
    dt: float = 0.01,
    steps_per_frame: int = 10,
    max_history: int = 20,
    seed: Optional[int] = None,
) -> ParticleTraces:
    """Advect particles through animation frames.

    Args:
        animation_frames: AnimationFrames containing velocity data.
        n_particles: Number of particles to track.
        dt: Time step per advection step.
        steps_per_frame: Number of advection steps per frame.
        max_history: Maximum particle history to keep.
        seed: Optional random seed.

    Returns:
        ParticleTraces with position history.
    """
    if not animation_frames.frames:
        return ParticleTraces([], [], n_particles, max_history)

    first_frame = animation_frames.frames[0]
    X, Y = first_frame.X, first_frame.Y
    particles_x, particles_y = initialize_particles(X, Y, n_particles, seed=seed)

    all_x = [particles_x.copy()]
    all_y = [particles_y.copy()]

    for frame in animation_frames.frames:
        u = frame.fields.get("u")
        v = frame.fields.get("v")
        if u is None or v is None:
            continue

        # Take multiple advection steps per frame
        for _ in range(steps_per_frame):
            particles_x, particles_y = compute_particle_positions(
                frame.X, frame.Y, u, v, particles_x, particles_y, dt
            )

            # Reset particles that leave domain
            x_min, x_max = X.min(), X.max()
            y_min, y_max = Y.min(), Y.max()
            out_of_bounds = (
                (particles_x >= x_max)
                | (particles_x < x_min)
                | (particles_y >= y_max)
                | (particles_y < y_min)
            )
            particles_x[out_of_bounds] = x_min + 0.01 * (x_max - x_min)
            particles_y[out_of_bounds] = np.random.uniform(
                y_min, y_max, np.sum(out_of_bounds)
            )

        all_x.append(particles_x.copy())
        all_y.append(particles_y.copy())

    return ParticleTraces(all_x, all_y, n_particles, max_history)
