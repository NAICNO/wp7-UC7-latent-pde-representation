"""
Shared fixtures for the latent-representation-of-pde-solutions test suite.

All fixtures are pure numpy/scipy -- no TensorFlow dependency.
"""

import sys
import os

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Make src/ importable without installing the package
# ---------------------------------------------------------------------------
SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(SRC_DIR))


# ---------------------------------------------------------------------------
# Grid fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def grid_n8():
    """Return the full make_grid output for n=8 (small, fast)."""
    from cd2d_streamfunc import make_grid
    return make_grid(8)


@pytest.fixture
def grid_n16():
    """Return the full make_grid output for n=16."""
    from cd2d_streamfunc import make_grid
    return make_grid(16)


@pytest.fixture
def grid_n4():
    """Return the full make_grid output for n=4 (minimal)."""
    from cd2d_streamfunc import make_grid
    return make_grid(4)


# ---------------------------------------------------------------------------
# Streamfunction fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def idx_nsf2():
    """Monomial multi-indices for n_sf=2."""
    from cd2d_streamfunc import monomial_multiindices
    return monomial_multiindices(2)


@pytest.fixture
def coeffs_nsf2_10samples():
    """Coefficient array for n_sf=2, 10 samples, normal mode, fixed seed."""
    from cd2d_streamfunc import sample_coeffs
    C, idx = sample_coeffs(n_sf=2, n_sol=10, mode="normal", seed=42)
    return C, idx


@pytest.fixture
def simple_linear_coeffs(grid_n8):
    """
    A streamfunction psi = x  (only a_{1,0}=1, all others 0).
    v_x = d psi/dy = 0,  v_y = -d psi/dx = -1.
    """
    from cd2d_streamfunc import monomial_multiindices
    idx = monomial_multiindices(2)
    # find (1,0) position
    pos_10 = idx.index((1, 0))
    n_coeff = len(idx)
    coeffs = np.zeros(n_coeff, dtype=np.float64)
    coeffs[pos_10] = 1.0
    return coeffs, idx


@pytest.fixture
def simple_y_coeffs(grid_n8):
    """
    A streamfunction psi = y  (only a_{0,1}=1, all others 0).
    v_x = d psi/dy = 1,  v_y = -d psi/dx = 0.
    """
    from cd2d_streamfunc import monomial_multiindices
    idx = monomial_multiindices(2)
    pos_01 = idx.index((0, 1))
    n_coeff = len(idx)
    coeffs = np.zeros(n_coeff, dtype=np.float64)
    coeffs[pos_01] = 1.0
    return coeffs, idx


# ---------------------------------------------------------------------------
# FV / solver fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def zero_velocity_fields(grid_n8):
    """Zero velocity on all face grids (pure diffusion problem)."""
    _, _, _, _, Xef, Yef, Xnf, Ynf, _ = grid_n8
    vx_f = np.zeros_like(Xef)
    vy_f = np.zeros_like(Xnf)
    return vx_f, vy_f


@pytest.fixture
def assembled_pure_diffusion(zero_velocity_fields):
    """FV matrix for pure diffusion (zero convection) on n=8 grid."""
    from cd2d_streamfunc import assemble_fv_system, constant_diffusion
    vx_f, vy_f = zero_velocity_fields
    n = 8
    dfun = constant_diffusion()
    A, h = assemble_fv_system(n, vx_f, vy_f, dfun)
    return A, h, n


# ---------------------------------------------------------------------------
# Splitting fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_index_array():
    """A simple sequential index array of length 100."""
    return np.arange(100, dtype=int)


@pytest.fixture
def default_rng():
    """A seeded numpy RNG for reproducible split tests."""
    return np.random.default_rng(seed=0)


# ---------------------------------------------------------------------------
# Error metric fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def perfect_predictions():
    """y_true == y_pred → REE should be 0."""
    rng = np.random.default_rng(7)
    y = rng.standard_normal((5, 16, 16)).astype(np.float32)
    return y, y.copy()


@pytest.fixture
def batch_1d_arrays():
    """Simple 1-D per-sample arrays for ree_rel_sq."""
    y_true = np.array([[1.0, 0.0], [0.0, 1.0], [3.0, 4.0]], dtype=np.float32)
    y_pred = np.array([[1.0, 0.0], [0.0, 2.0], [0.0, 0.0]], dtype=np.float32)
    return y_true, y_pred
