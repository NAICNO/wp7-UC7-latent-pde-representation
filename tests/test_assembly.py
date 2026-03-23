"""
Tests for:
  - constant_diffusion()
  - assemble_fv_system(n, vx_f, vy_f, dfun)
  - apply_dirichlet(A, s_grid, g_grid)
  - solve_level(n, coeffs, idx, vel_scale)

Focus: pure numpy/scipy, no TensorFlow.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from cd2d_streamfunc import (
    make_grid,
    monomial_multiindices,
    sample_coeffs,
    assemble_fv_system,
    apply_dirichlet,
    constant_diffusion,
    gaussian_all_boundaries,
    solve_level,
)


# ============================================================
# constant_diffusion
# ============================================================

class TestConstantDiffusion:
    """Unit tests for the constant_diffusion() factory."""

    def test_returns_callable(self):
        d = constant_diffusion()
        assert callable(d)

    def test_returns_ones(self):
        d = constant_diffusion()
        X = np.linspace(-1, 1, 8).reshape(8, 1)
        Y = np.linspace(-1, 1, 8).reshape(1, 8)
        Xg, Yg = np.meshgrid(X, Y)
        result = d(Xg, Yg)
        np.testing.assert_allclose(result, np.ones_like(Xg))

    def test_output_dtype_float64(self):
        d = constant_diffusion()
        X = np.zeros((4, 4))
        Y = np.zeros((4, 4))
        assert d(X, Y).dtype == np.float64


# ============================================================
# Helpers
# ============================================================

def _face_velocities(n, vx_val=0.0, vy_val=0.0):
    """Constant face velocities for testing."""
    _, _, _, _, Xef, _, Xnf, _, _ = make_grid(n)
    vx_f = np.full_like(Xef, vx_val, dtype=np.float64)
    vy_f = np.full_like(Xnf, vy_val, dtype=np.float64)
    return vx_f, vy_f


# ============================================================
# assemble_fv_system
# ============================================================

class TestAssembleFvSystem:
    """Unit tests for assemble_fv_system."""

    def test_returns_tuple_of_two(self):
        n = 8
        vx_f, vy_f = _face_velocities(n)
        result = assemble_fv_system(n, vx_f, vy_f, constant_diffusion())
        assert isinstance(result, tuple) and len(result) == 2

    def test_matrix_shape(self):
        n = 8
        vx_f, vy_f = _face_velocities(n)
        A, _ = assemble_fv_system(n, vx_f, vy_f, constant_diffusion())
        assert A.shape == (n * n, n * n)

    def test_h_value(self):
        n = 8
        vx_f, vy_f = _face_velocities(n)
        _, h = assemble_fv_system(n, vx_f, vy_f, constant_diffusion())
        assert float(h) == pytest.approx(2.0 / (n - 1))

    def test_matrix_is_csr(self):
        n = 8
        vx_f, vy_f = _face_velocities(n)
        A, _ = assemble_fv_system(n, vx_f, vy_f, constant_diffusion())
        assert sp.issparse(A)
        assert A.format == "csr"

    def test_pure_diffusion_interior_rows_sum_to_zero(self):
        """
        For pure diffusion on a uniform grid, each *interior* row sums to 0
        (by conservation of flux: diffusion matrix is consistent).
        Boundary rows are not in the interior but may be zero (not yet overwritten).
        """
        n = 8
        vx_f, vy_f = _face_velocities(n, 0.0, 0.0)
        A, _ = assemble_fv_system(n, vx_f, vy_f, constant_diffusion())

        A_dense = A.toarray()
        I = np.arange(1, n - 1)
        J = np.arange(1, n - 1)
        II, JJ = np.meshgrid(I, J, indexing="ij")
        interior_flat = (II * n + JJ).ravel()
        row_sums = A_dense[interior_flat, :].sum(axis=1)
        np.testing.assert_allclose(row_sums, 0.0, atol=1e-12)

    def test_pure_diffusion_symmetric(self):
        """
        Pure diffusion (constant D, zero v) should produce a symmetric operator
        on the interior degrees of freedom.  Boundary rows are zero before
        Dirichlet application, so symmetry is only guaranteed for interior rows
        and columns simultaneously.
        """
        n = 8
        vx_f, vy_f = _face_velocities(n, 0.0, 0.0)
        A, _ = assemble_fv_system(n, vx_f, vy_f, constant_diffusion())
        # Extract interior index set
        I = np.arange(1, n - 1)
        J = np.arange(1, n - 1)
        II, JJ = np.meshgrid(I, J, indexing="ij")
        int_idx = (II * n + JJ).ravel()
        A_dense = A.toarray()
        A_int = A_dense[np.ix_(int_idx, int_idx)]
        np.testing.assert_allclose(A_int, A_int.T, atol=1e-12)

    def test_sparsity_reasonable(self):
        """Interior cells have at most 5 neighbours; boundary cells fewer."""
        n = 16
        vx_f, vy_f = _face_velocities(n)
        A, _ = assemble_fv_system(n, vx_f, vy_f, constant_diffusion())
        nnz_per_row = np.diff(A.indptr)
        assert np.all(nnz_per_row <= 5)

    def test_positive_diagonal_pure_diffusion(self):
        """Diagonal of a pure-diffusion interior operator should be positive."""
        n = 8
        vx_f, vy_f = _face_velocities(n)
        A, _ = assemble_fv_system(n, vx_f, vy_f, constant_diffusion())
        diag = A.diagonal()
        # Interior cells
        I = np.arange(1, n - 1)
        J = np.arange(1, n - 1)
        II, JJ = np.meshgrid(I, J, indexing="ij")
        interior_flat = (II * n + JJ).ravel()
        assert np.all(diag[interior_flat] > 0.0)

    def test_nonzero_velocity_changes_matrix(self):
        """Adding convection changes the assembled matrix."""
        n = 8
        vx0, vy0 = _face_velocities(n, 0.0, 0.0)
        vx1, vy1 = _face_velocities(n, 10.0, 0.0)
        A0, _ = assemble_fv_system(n, vx0, vy0, constant_diffusion())
        A1, _ = assemble_fv_system(n, vx1, vy1, constant_diffusion())
        assert not np.allclose(A0.toarray(), A1.toarray())

    @pytest.mark.parametrize("n", [4, 8, 16])
    def test_various_sizes(self, n):
        vx_f, vy_f = _face_velocities(n)
        A, h = assemble_fv_system(n, vx_f, vy_f, constant_diffusion())
        assert A.shape == (n * n, n * n)


# ============================================================
# apply_dirichlet
# ============================================================

class TestApplyDirichlet:
    """Unit tests for apply_dirichlet."""

    def _assemble_and_dirichlet(self, n, g_grid=None):
        vx_f, vy_f = _face_velocities(n)
        A, _ = assemble_fv_system(n, vx_f, vy_f, constant_diffusion())
        s = np.zeros((n, n), dtype=np.float64)
        return apply_dirichlet(A, s, g_grid)

    def test_returns_tuple_of_two(self):
        result = self._assemble_and_dirichlet(8)
        assert isinstance(result, tuple) and len(result) == 2

    def test_boundary_rows_identity(self):
        """After Dirichlet, each boundary row should be an identity row."""
        n = 8
        A_dir, _ = self._assemble_and_dirichlet(n)
        A_dense = A_dir.toarray()
        # Build boundary mask
        bmask = np.zeros((n, n), dtype=bool)
        bmask[0, :] = True; bmask[-1, :] = True
        bmask[:, 0] = True; bmask[:, -1] = True
        brows = np.flatnonzero(bmask.ravel())
        for r in brows:
            row = A_dense[r, :]
            assert row[r] == pytest.approx(1.0)
            assert np.sum(np.abs(row)) == pytest.approx(1.0)

    def test_rhs_at_boundary_equals_g(self):
        """b[boundary] must equal the prescribed g values."""
        n = 8
        g = gaussian_all_boundaries(n)
        A_dir, b = self._assemble_and_dirichlet(n, g_grid=g)
        bmask = np.zeros((n, n), dtype=bool)
        bmask[0, :] = True; bmask[-1, :] = True
        bmask[:, 0] = True; bmask[:, -1] = True
        np.testing.assert_allclose(b[bmask.ravel()], g.ravel()[bmask.ravel()])

    def test_rhs_interior_zero_when_no_source(self):
        """Interior RHS should be zero when no source term is given."""
        n = 8
        A_dir, b = self._assemble_and_dirichlet(n)
        bmask = np.zeros((n, n), dtype=bool)
        bmask[0, :] = True; bmask[-1, :] = True
        bmask[:, 0] = True; bmask[:, -1] = True
        interior = ~bmask.ravel()
        np.testing.assert_allclose(b[interior], 0.0, atol=1e-15)

    def test_zero_g_default(self):
        """Default g=None → g=0 everywhere, so b=0 at boundary too."""
        n = 8
        _, b = self._assemble_and_dirichlet(n, g_grid=None)
        np.testing.assert_allclose(b, 0.0, atol=1e-15)

    def test_output_is_csr(self):
        n = 8
        A_dir, _ = self._assemble_and_dirichlet(n)
        assert A_dir.format == "csr"


# ============================================================
# solve_level
# ============================================================

class TestSolveLevel:
    """Integration tests for the full FV solve pipeline."""

    def _simple_coeffs(self, n_sf=2):
        idx = monomial_multiindices(n_sf)
        coeffs = np.zeros(len(idx), dtype=np.float64)
        # psi = y → uniform convection, easy to solve
        if (0, 1) in idx:
            coeffs[idx.index((0, 1))] = 0.1
        return coeffs, idx

    def test_returns_2d_array(self):
        n = 8
        coeffs, idx = self._simple_coeffs()
        u = solve_level(n, coeffs, idx, vel_scale=1.0, solver="scipy")
        assert isinstance(u, np.ndarray)
        assert u.ndim == 2

    def test_output_shape(self):
        n = 8
        coeffs, idx = self._simple_coeffs()
        u = solve_level(n, coeffs, idx, vel_scale=1.0, solver="scipy")
        assert u.shape == (n, n)

    def test_pure_diffusion_symmetric_solution(self):
        """
        With psi=0 (no convection) and symmetric BCs, the solution should be
        approximately symmetric (left-right and top-bottom).
        """
        n = 16
        idx = monomial_multiindices(2)
        coeffs = np.zeros(len(idx), dtype=np.float64)  # zero streamfunction
        u = solve_level(n, coeffs, idx, vel_scale=0.0, solver="scipy")
        # Left-right symmetry (x → -x)
        np.testing.assert_allclose(u, u[::-1, :], atol=1e-10)
        # Top-bottom symmetry (y → -y)
        np.testing.assert_allclose(u, u[:, ::-1], atol=1e-10)

    def test_solution_finite(self):
        """Solution values must all be finite."""
        n = 8
        coeffs, idx = self._simple_coeffs()
        u = solve_level(n, coeffs, idx, vel_scale=1.0, solver="scipy")
        assert np.all(np.isfinite(u))

    def test_solution_nonnegative_with_positive_bc(self):
        """
        Gaussian BCs are non-negative and source is zero.  By the maximum
        principle the solution should lie in [0, max_BC] (within solver
        tolerance for the discrete system).
        """
        n = 16
        idx = monomial_multiindices(2)
        # mild convection
        coeffs = np.zeros(len(idx), dtype=np.float64)
        if (0, 1) in idx:
            coeffs[idx.index((0, 1))] = 0.05
        u = solve_level(n, coeffs, idx, vel_scale=1.0, solver="scipy")
        assert np.all(u >= -1e-6)

    def test_higher_resolution_runs(self):
        """Solver must succeed for n=32 without crashing."""
        n = 32
        coeffs, idx = self._simple_coeffs(n_sf=3)
        u = solve_level(n, coeffs, idx, vel_scale=1.0, solver="scipy")
        assert u.shape == (n, n)
        assert np.all(np.isfinite(u))

    def test_jacobi_scale_option(self):
        """jacobi_scale=True must also return a valid finite array."""
        n = 8
        coeffs, idx = self._simple_coeffs()
        u = solve_level(n, coeffs, idx, vel_scale=1.0, solver="scipy", jacobi_scale=True)
        assert u.shape == (n, n)
        assert np.all(np.isfinite(u))

    def test_different_vel_scales_differ(self):
        """Different velocity scalings must produce different solutions."""
        n = 16
        idx = monomial_multiindices(2)
        rng = np.random.default_rng(5)
        coeffs = rng.standard_normal(len(idx))
        u_low = solve_level(n, coeffs, idx, vel_scale=1.0, solver="scipy")
        u_high = solve_level(n, coeffs, idx, vel_scale=1e5, solver="scipy")
        # High Peclet number will tilt the solution
        assert not np.allclose(u_low, u_high, atol=1e-6)

    def test_debug_mode_runs(self, capsys):
        """debug=True path must execute without error and print diagnostics."""
        n = 8
        idx = monomial_multiindices(2)
        coeffs = np.zeros(len(idx), dtype=np.float64)
        if (0, 1) in idx:
            coeffs[idx.index((0, 1))] = 0.1
        u = solve_level(
            n, coeffs, idx, vel_scale=1.0, solver="scipy",
            debug=True, dbg_tag="unit-test"
        )
        assert u.shape == (n, n)
        captured = capsys.readouterr()
        assert "unit-test" in captured.out
