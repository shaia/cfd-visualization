"""
CFD Field Computations
======================

Pure functions for computing flow field quantities from CFD simulation data.
All functions take numpy arrays as input and return numpy arrays or dataclasses.

Modules:
    velocity: Velocity magnitude, components, fluctuations
    vorticity: Vorticity, Q-criterion, circulation, vortex detection
    gradients: Velocity gradients, strain rate tensors
    derived: Energy, pressure coefficients, flow statistics

Example:
    >>> from cfd_viz.fields import velocity, vorticity
    >>> from cfd_viz.io import read_vtk_file
    >>>
    >>> data = read_vtk_file("flow.vtk")
    >>> speed = velocity.magnitude(data.u, data.v)
    >>> omega = vorticity.vorticity(data.u, data.v, data.dx, data.dy)
    >>> Q = vorticity.q_criterion(data.u, data.v, data.dx, data.dy)
"""

# Velocity functions
# Derived field functions and classes
from .derived import (
    FlowStatistics,
    calculate_flow_statistics,
    dissipation_rate,
    dynamic_pressure,
    helicity_density,
    kinetic_energy,
    mach_number,
    pressure_coefficient,
    reynolds_number_local,
    stream_function,
    total_kinetic_energy,
    total_pressure,
)

# Gradient functions and classes
from .gradients import (
    StrainRateTensor,
    VelocityGradients,
    divergence,
    normal_strain_rates,
    rotation_rate,
    shear_strain_rate,
    strain_rate_components,
    strain_rate_magnitude,
    strain_rate_tensor,
    velocity_gradients,
    wall_shear_stress,
)
from .velocity import (
    angle,
    angle_degrees,
    components_from_magnitude_angle,
    fluctuations,
    magnitude,
    normalize,
    speed,
    turbulent_intensity,
)

# Vorticity functions
from .vorticity import (
    circulation,
    detect_vortex_cores,
    enstrophy,
    lambda2_criterion,
    q_criterion,
    vorticity,
    vorticity_from_gradients,
)

__all__ = [
    # velocity
    "magnitude",
    "speed",
    "normalize",
    "angle",
    "angle_degrees",
    "components_from_magnitude_angle",
    "fluctuations",
    "turbulent_intensity",
    # vorticity
    "vorticity",
    "vorticity_from_gradients",
    "q_criterion",
    "lambda2_criterion",
    "enstrophy",
    "circulation",
    "detect_vortex_cores",
    # gradients
    "VelocityGradients",
    "StrainRateTensor",
    "velocity_gradients",
    "strain_rate_tensor",
    "strain_rate_components",
    "divergence",
    "shear_strain_rate",
    "normal_strain_rates",
    "strain_rate_magnitude",
    "rotation_rate",
    "wall_shear_stress",
    # derived
    "FlowStatistics",
    "kinetic_energy",
    "total_kinetic_energy",
    "dynamic_pressure",
    "pressure_coefficient",
    "total_pressure",
    "mach_number",
    "reynolds_number_local",
    "stream_function",
    "helicity_density",
    "dissipation_rate",
    "calculate_flow_statistics",
]
