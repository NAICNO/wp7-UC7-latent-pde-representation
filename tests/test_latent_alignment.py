"""
Tests for pure numpy/scipy functions in analyze_latent_alignment.py:
  - ree(A, B, eps)
  - center(A)
  - procrustes_R(Ac, Bc)
  - pairwise_matrix(arrs, metric_fn)

No TensorFlow dependency.  All operations are pure numpy.
"""

import importlib.util
import os
import sys

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Load the module directly (no TF dependency in this file)
# ---------------------------------------------------------------------------

_SRC = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src")
)

def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_SRC, filename)
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_ala = _load_module("analyze_latent_alignment", "analyze_latent_alignment.py")

ree = _ala.ree
center = _ala.center
procrustes_R = _ala.procrustes_R
pairwise_matrix = _ala.pairwise_matrix


# ============================================================
# ree(A, B, eps)
# ============================================================

class TestReeMatrixMetric:
    """Tests for the Frobenius-norm-based REE metric in analyze_latent_alignment."""

    def test_identical_arrays_zero(self):
        """REE of A vs itself is 0 (numerator ||A-A||^2 = 0)."""
        A = np.eye(5, dtype=np.float32)
        assert ree(A, A) == pytest.approx(0.0, abs=1e-12)

    def test_nonnegative(self):
        rng = np.random.default_rng(0)
        A = rng.standard_normal((10, 8)).astype(np.float32)
        B = rng.standard_normal((10, 8)).astype(np.float32)
        assert ree(A, B) >= 0.0

    def test_symmetry(self):
        """ree(A, B) == ree(B, A) because the denominator is symmetric."""
        rng = np.random.default_rng(1)
        A = rng.standard_normal((6, 4)).astype(np.float32)
        B = rng.standard_normal((6, 4)).astype(np.float32)
        assert ree(A, B) == pytest.approx(ree(B, A), rel=1e-6)

    def test_zero_denominator_uses_eps(self):
        """When both A and B are zero, eps prevents division-by-zero."""
        A = np.zeros((3, 3), dtype=np.float32)
        B = np.ones((3, 3), dtype=np.float32)
        result = ree(A, B, eps=1e-12)
        assert np.isfinite(result)

    def test_known_value_1d(self):
        """
        A = [1, 0],  B = [0, 1]
        num = 2,  den = ||A|| * ||B|| + eps = 1*1 = 1
        REE = 2.0
        """
        A = np.array([[1.0, 0.0]], dtype=np.float32)
        B = np.array([[0.0, 1.0]], dtype=np.float32)
        result = ree(A, B, eps=0.0)
        assert result == pytest.approx(2.0, rel=1e-6)

    def test_scale_effect(self):
        """Scaling both A and B by the same factor alpha cancels in the ratio."""
        rng = np.random.default_rng(3)
        A = rng.standard_normal((8, 4)).astype(np.float32)
        B = rng.standard_normal((8, 4)).astype(np.float32)
        alpha = 5.0
        r1 = ree(A, B, eps=0.0)
        r2 = ree(alpha * A, alpha * B, eps=0.0)
        assert r1 == pytest.approx(r2, rel=1e-5)

    def test_large_arrays(self):
        """No errors or overflows with large N=1000, D=64 arrays."""
        rng = np.random.default_rng(7)
        A = rng.standard_normal((1000, 64)).astype(np.float32)
        B = rng.standard_normal((1000, 64)).astype(np.float32)
        result = ree(A, B)
        assert np.isfinite(result)
        assert result >= 0.0

    @pytest.mark.parametrize("shape", [(5, 3), (20, 16), (100, 8)])
    def test_finite_for_various_shapes(self, shape):
        rng = np.random.default_rng(42)
        A = rng.standard_normal(shape).astype(np.float32)
        B = rng.standard_normal(shape).astype(np.float32)
        result = ree(A, B)
        assert np.isfinite(result)


# ============================================================
# center(A)
# ============================================================

class TestCenter:
    """Tests for center(A) — subtracts column means."""

    def test_centered_column_mean_is_zero(self):
        rng = np.random.default_rng(0)
        A = rng.standard_normal((50, 8)).astype(np.float32)
        Ac = center(A)
        np.testing.assert_allclose(Ac.mean(axis=0), 0.0, atol=1e-6)

    def test_preserves_shape(self):
        A = np.ones((10, 5), dtype=np.float64)
        Ac = center(A)
        assert Ac.shape == A.shape

    def test_constant_matrix_becomes_zero(self):
        A = np.full((6, 4), 7.5, dtype=np.float64)
        Ac = center(A)
        np.testing.assert_allclose(Ac, 0.0, atol=1e-12)

    def test_does_not_mutate_input(self):
        A = np.arange(12, dtype=np.float64).reshape(4, 3)
        A_copy = A.copy()
        _ = center(A)
        np.testing.assert_array_equal(A, A_copy)

    def test_identity_center(self):
        """center(A) + mean(A, axis=0) should equal A."""
        rng = np.random.default_rng(5)
        A = rng.standard_normal((20, 6)).astype(np.float64)
        Ac = center(A)
        np.testing.assert_allclose(Ac + A.mean(axis=0), A, atol=1e-12)

    def test_already_centered_unchanged(self):
        """An already zero-mean matrix should be unchanged."""
        A = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=np.float64)
        Ac = center(A)
        np.testing.assert_allclose(Ac, A, atol=1e-12)

    def test_single_row_becomes_zero(self):
        A = np.array([[3.0, 4.0, 5.0]])
        Ac = center(A)
        np.testing.assert_allclose(Ac, np.zeros_like(A), atol=1e-12)

    @pytest.mark.parametrize("n,d", [(3, 2), (100, 32), (1, 8)])
    def test_parametrized_shapes(self, n, d):
        rng = np.random.default_rng(n)
        A = rng.standard_normal((n, d))
        Ac = center(A)
        assert Ac.shape == (n, d)
        np.testing.assert_allclose(Ac.mean(axis=0), 0.0, atol=1e-6)


# ============================================================
# procrustes_R(Ac, Bc)
# ============================================================

class TestProcrustesR:
    """Tests for procrustes_R(Ac, Bc) — optimal orthogonal alignment."""

    def _is_orthogonal(self, R, atol=1e-6):
        n = R.shape[0]
        np.testing.assert_allclose(R @ R.T, np.eye(n), atol=atol)
        np.testing.assert_allclose(R.T @ R, np.eye(n), atol=atol)

    def test_result_is_square(self):
        rng = np.random.default_rng(0)
        A = rng.standard_normal((20, 4))
        B = rng.standard_normal((20, 4))
        R = procrustes_R(A, B)
        assert R.shape == (4, 4)

    def test_result_is_orthogonal(self):
        rng = np.random.default_rng(1)
        A = rng.standard_normal((30, 6))
        B = rng.standard_normal((30, 6))
        R = procrustes_R(A, B)
        self._is_orthogonal(R)

    def test_determinant_positive(self):
        """Proper rotation: det(R) should be +1, not -1."""
        rng = np.random.default_rng(2)
        A = rng.standard_normal((25, 5))
        B = rng.standard_normal((25, 5))
        R = procrustes_R(A, B)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-6)

    def test_identity_on_identical_inputs(self):
        """When Ac == Bc the optimal rotation is the identity."""
        rng = np.random.default_rng(3)
        A = rng.standard_normal((15, 4))
        R = procrustes_R(A, A)
        np.testing.assert_allclose(R, np.eye(4), atol=1e-6)

    def test_minimizes_frobenius_distance(self):
        """After applying R, ||Ac R - Bc||_F should be minimal over all rotations."""
        rng = np.random.default_rng(4)
        A = rng.standard_normal((20, 3))
        # B is A plus noise after a known rotation
        theta = np.pi / 6
        Q_known = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0,              0,             1],
        ])
        B = A @ Q_known + 0.01 * rng.standard_normal((20, 3))
        R = procrustes_R(A, B)
        # ||A R - B|| should be small (near the noise level)
        residual = np.linalg.norm(A @ R - B, "fro")
        assert residual < 1.0  # loose bound: definitely less than uninformed

    @pytest.mark.parametrize("d", [2, 4, 8, 16])
    def test_various_dimensions(self, d):
        rng = np.random.default_rng(d)
        A = rng.standard_normal((50, d))
        B = rng.standard_normal((50, d))
        R = procrustes_R(A, B)
        assert R.shape == (d, d)
        self._is_orthogonal(R)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-5)


# ============================================================
# pairwise_matrix(arrs, metric_fn)
# ============================================================

class TestPairwiseMatrix:
    """Tests for pairwise_matrix(arrs, metric_fn)."""

    def test_output_shape(self):
        arrs = [np.ones((10, 4)) * i for i in range(5)]
        M = pairwise_matrix(arrs, ree)
        assert M.shape == (5, 5)

    def test_diagonal_zero_with_ree(self):
        """REE of identical arrays is 0, so diagonal should be 0."""
        rng = np.random.default_rng(0)
        arrs = [rng.standard_normal((8, 4)).astype(np.float32) for _ in range(4)]
        M = pairwise_matrix(arrs, ree)
        np.testing.assert_allclose(np.diag(M), 0.0, atol=1e-12)

    def test_symmetric_matrix(self):
        """Result should be symmetric: M[i,j] == M[j,i]."""
        rng = np.random.default_rng(1)
        arrs = [rng.standard_normal((12, 5)).astype(np.float32) for _ in range(4)]
        M = pairwise_matrix(arrs, ree)
        np.testing.assert_allclose(M, M.T, atol=1e-6)

    def test_nonnegative_entries(self):
        rng = np.random.default_rng(2)
        arrs = [rng.standard_normal((10, 3)).astype(np.float32) for _ in range(3)]
        M = pairwise_matrix(arrs, ree)
        assert np.all(M >= 0.0)

    def test_two_arrays(self):
        """Minimum case: 2 arrays."""
        A = np.ones((5, 3), dtype=np.float32)
        B = np.zeros((5, 3), dtype=np.float32)
        M = pairwise_matrix([A, B], ree)
        assert M.shape == (2, 2)
        assert M[0, 0] == pytest.approx(0.0, abs=1e-12)
        assert M[1, 1] == pytest.approx(0.0, abs=1e-12)
        # off-diagonal entries should be equal
        assert M[0, 1] == pytest.approx(M[1, 0], rel=1e-6)

    def test_output_dtype_float32(self):
        arrs = [np.ones((4, 2), dtype=np.float32) * i for i in range(3)]
        M = pairwise_matrix(arrs, ree)
        assert M.dtype == np.float32

    def test_custom_metric_fn(self):
        """Works with any callable metric, not just ree."""
        def always_one(A, B):
            return 1.0

        arrs = [np.eye(3) for _ in range(4)]
        M = pairwise_matrix(arrs, always_one)
        # Diagonal stays 0 (loop only fills i < j)
        np.testing.assert_allclose(np.diag(M), 0.0)
        # All off-diagonal should be 1.0
        off = M[~np.eye(4, dtype=bool)]
        np.testing.assert_allclose(off, 1.0)

    @pytest.mark.parametrize("M_size", [2, 3, 5])
    def test_various_sizes(self, M_size):
        rng = np.random.default_rng(M_size)
        arrs = [rng.standard_normal((20, 6)).astype(np.float32) for _ in range(M_size)]
        M = pairwise_matrix(arrs, ree)
        assert M.shape == (M_size, M_size)
