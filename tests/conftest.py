"""Pytest configuration and shared fixtures for cfd_viz tests."""

import numpy as np
import pytest


@pytest.fixture
def uniform_grid():
    """Create a uniform 2D grid for testing."""
    nx, ny = 50, 50
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    return {"x": x, "y": y, "X": X, "Y": Y, "dx": dx, "dy": dy, "nx": nx, "ny": ny}


@pytest.fixture
def uniform_flow(uniform_grid):
    """Create uniform flow field (constant velocity)."""
    shape = (uniform_grid["ny"], uniform_grid["nx"])
    u = np.ones(shape) * 1.0  # u = 1 m/s
    v = np.zeros(shape)  # v = 0
    p = np.zeros(shape)  # p = 0
    return {"u": u, "v": v, "p": p, **uniform_grid}


@pytest.fixture
def shear_flow(uniform_grid):
    """Create simple shear flow (u varies linearly with y)."""
    Y = uniform_grid["Y"]
    u = Y.copy()  # u = y (linear shear)
    v = np.zeros_like(Y)
    p = np.zeros_like(Y)
    return {"u": u, "v": v, "p": p, **uniform_grid}


@pytest.fixture
def vortex_flow(uniform_grid):
    """Create a solid body rotation (vortex) flow field."""
    X, Y = uniform_grid["X"], uniform_grid["Y"]
    # Center the coordinates
    xc, yc = 0.5, 0.5
    Xc, Yc = X - xc, Y - yc
    # Solid body rotation: u = -omega*y, v = omega*x
    omega = 2 * np.pi  # Angular velocity
    u = -omega * Yc
    v = omega * Xc
    p = np.zeros_like(X)
    return {"u": u, "v": v, "p": p, "omega": omega, **uniform_grid}


@pytest.fixture
def stagnation_flow(uniform_grid):
    """Create stagnation point flow."""
    X, Y = uniform_grid["X"], uniform_grid["Y"]
    # Center the coordinates
    xc, yc = 0.5, 0.5
    Xc, Yc = X - xc, Y - yc
    # Stagnation flow: u = k*x, v = -k*y
    k = 1.0
    u = k * Xc
    v = -k * Yc
    p = np.zeros_like(X)
    return {"u": u, "v": v, "p": p, "k": k, **uniform_grid}


@pytest.fixture
def channel_flow(uniform_grid):
    """Create parabolic channel flow (Poiseuille flow)."""
    Y = uniform_grid["Y"]
    y = uniform_grid["y"]
    # Parabolic profile: u = U_max * (1 - (2y/H - 1)^2)
    H = y[-1] - y[0]
    U_max = 1.0
    y_norm = 2 * (Y - y[0]) / H - 1  # Normalize to [-1, 1]
    u = U_max * (1 - y_norm**2)
    v = np.zeros_like(Y)
    # Pressure decreases linearly in x
    X = uniform_grid["X"]
    dp_dx = -0.1
    p = dp_dx * X
    return {"u": u, "v": v, "p": p, "U_max": U_max, "H": H, **uniform_grid}


@pytest.fixture
def random_flow(uniform_grid):
    """Create random flow field for general testing."""
    np.random.seed(42)  # Reproducible
    shape = (uniform_grid["ny"], uniform_grid["nx"])
    u = np.random.randn(*shape)
    v = np.random.randn(*shape)
    p = np.random.randn(*shape)
    return {"u": u, "v": v, "p": p, **uniform_grid}
