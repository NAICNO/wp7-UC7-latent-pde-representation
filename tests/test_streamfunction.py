"""
Tests for streamfunction helpers:
  - monomial_multiindices(n_sf)
  - sample_coeffs(n_sf, n_sol, mode, seed)
  - eval_velocity_from_streamfunc_coeffs(...)
"""

import numpy as np
import pytest

from cd2d_streamfunc import (
    make_grid,
    monomial_multiindices,
    sample_coeffs,
    eval_velocity_from_streamfunc_coeffs,
)


# ============================================================
# monomial_multiindices
# ============================================================

class TestMonomialMultiindices:
    """Unit tests for monomial_multiindices(n_sf)."""

    def test_excludes_00(self):
        for n_sf in range(1, 6):
            idx = monomial_multiindices(n_sf)
            assert (0, 0) not in idx, f"(0,0) must be excluded for n_sf={n_sf}"

    def test_length_formula(self):
        """Length must equal 0.5*(n_sf+1)*(n_sf+2) - 1."""
        for n_sf in range(1, 7):
            expected = (n_sf + 1) * (n_sf + 2) // 2 - 1
            assert len(monomial_multiindices(n_sf)) == expected

    def test_nsf1_entries(self):
        """For n_sf=1: only (0,1) and (1,0)."""
        idx = monomial_multiindices(1)
        assert set(idx) == {(0, 1), (1, 0)}

    def test_nsf2_entries(self):
        """For n_sf=2: (0,1),(0,2),(1,0),(1,1),(2,0)."""
        idx = monomial_multiindices(2)
        assert set(idx) == {(0, 1), (0, 2), (1, 0), (1, 1), (2, 0)}

    def test_degree_constraint(self):
        """All entries (i,j) must satisfy i+j <= n_sf."""
        for n_sf in range(1, 6):
            for i, j in monomial_multiindices(n_sf):
                assert i + j <= n_sf

    def test_all_nonneg_indices(self):
        for n_sf in range(1, 5):
            for i, j in monomial_multiindices(n_sf):
                assert i >= 0 and j >= 0

    def test_returns_list_of_tuples(self):
        idx = monomial_multiindices(3)
        assert isinstance(idx, list)
        assert all(isinstance(t, tuple) and len(t) == 2 for t in idx)

    def test_nsf0_empty(self):
        """For n_sf=0 the only monomial is (0,0), which is excluded → empty list."""
        idx = monomial_multiindices(0)
        assert idx == []

    def test_no_duplicates(self):
        for n_sf in range(1, 6):
            idx = monomial_multiindices(n_sf)
            assert len(idx) == len(set(idx))

    @pytest.mark.parametrize("n_sf", [3, 4, 5])
    def test_parametrized_length(self, n_sf):
        expected = (n_sf + 1) * (n_sf + 2) // 2 - 1
        assert len(monomial_multiindices(n_sf)) == expected


# ============================================================
# sample_coeffs
# ============================================================

class TestSampleCoeffs:
    """Unit tests for sample_coeffs(n_sf, n_sol, mode, seed)."""

    def test_returns_tuple_of_two(self):
        result = sample_coeffs(2, 5)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_coeff_shape(self):
        n_sf, n_sol = 3, 20
        C, idx = sample_coeffs(n_sf, n_sol)
        expected_n_coeff = len(monomial_multiindices(n_sf))
        assert C.shape == (n_sol, expected_n_coeff)

    def test_idx_length_matches_coeffs(self):
        C, idx = sample_coeffs(4, 10)
        assert C.shape[1] == len(idx)

    def test_dtype_float64(self):
        C, _ = sample_coeffs(2, 5)
        assert C.dtype == np.float64

    def test_reproducibility_with_seed(self):
        C1, _ = sample_coeffs(2, 10, seed=99)
        C2, _ = sample_coeffs(2, 10, seed=99)
        np.testing.assert_array_equal(C1, C2)

    def test_different_seeds_differ(self):
        C1, _ = sample_coeffs(2, 10, seed=1)
        C2, _ = sample_coeffs(2, 10, seed=2)
        assert not np.allclose(C1, C2)

    def test_normal_mode_mean_approx_zero(self):
        """Normal(0,1) samples: mean should be near 0 for large n_sol."""
        C, _ = sample_coeffs(2, 10000, mode="normal", seed=0)
        assert abs(float(np.mean(C))) < 0.05

    def test_normal_mode_std_approx_one(self):
        C, _ = sample_coeffs(2, 10000, mode="normal", seed=0)
        assert abs(float(np.std(C)) - 1.0) < 0.05

    def test_uniform_mode_bounds(self):
        """Uniform(-1,1): all values must lie in [-1, 1]."""
        C, _ = sample_coeffs(3, 500, mode="uniform", seed=7)
        assert np.all(C >= -1.0)
        assert np.all(C <= 1.0)

    def test_uniform_mode_mean_near_zero(self):
        C, _ = sample_coeffs(3, 10000, mode="uniform", seed=0)
        assert abs(float(np.mean(C))) < 0.05

    def test_n_sol_1(self):
        C, idx = sample_coeffs(2, 1)
        assert C.shape[0] == 1

    def test_different_n_sf_gives_different_n_coeff(self):
        C2, _ = sample_coeffs(2, 5)
        C4, _ = sample_coeffs(4, 5)
        assert C2.shape[1] < C4.shape[1]

    def test_idx_matches_monomial_multiindices(self):
        _, idx = sample_coeffs(3, 5)
        expected = monomial_multiindices(3)
        assert idx == expected


# ============================================================
# eval_velocity_from_streamfunc_coeffs
# ============================================================

class TestEvalVelocityFromStreamfuncCoeffs:
    """Tests for velocity field evaluation from streamfunction coefficients."""

    def _make_grid_parts(self, n=8):
        _, _, Xc, Yc, Xef, Yef, Xnf, Ynf, _ = make_grid(n)
        return Xc, Yc, Xef, Yef, Xnf, Ynf

    def test_return_structure(self):
        n = 8
        idx = monomial_multiindices(2)
        coeffs = np.ones(len(idx), dtype=np.float64)
        Xc, Yc, Xef, Yef, Xnf, Ynf = self._make_grid_parts(n)
        result = eval_velocity_from_streamfunc_coeffs(
            coeffs, idx, Xc, Yc, Xef, Yef, Xnf, Ynf
        )
        (vx_c, vy_c), (vx_f, vy_f), scaled_coeffs = result
        assert vx_c.shape == (n, n)
        assert vy_c.shape == (n, n)

    def test_xface_velocity_shape(self):
        n = 8
        idx = monomial_multiindices(2)
        coeffs = np.ones(len(idx), dtype=np.float64)
        Xc, Yc, Xef, Yef, Xnf, Ynf = self._make_grid_parts(n)
        _, (vx_f, vy_f), _ = eval_velocity_from_streamfunc_coeffs(
            coeffs, idx, Xc, Yc, Xef, Yef, Xnf, Ynf
        )
        assert vx_f.shape == (n - 1, n)
        assert vy_f.shape == (n, n - 1)

    def test_psi_equals_y_gives_vx_one(self):
        """
        psi(x,y) = y  (coeff at (0,1) = 1, all others 0)
        => vx = d psi/dy = 1,  vy = -d psi/dx = 0
        """
        n = 8
        idx = monomial_multiindices(2)
        pos_01 = idx.index((0, 1))
        coeffs = np.zeros(len(idx), dtype=np.float64)
        coeffs[pos_01] = 1.0
        Xc, Yc, Xef, Yef, Xnf, Ynf = self._make_grid_parts(n)
        (vx_c, vy_c), _, _ = eval_velocity_from_streamfunc_coeffs(
            coeffs, idx, Xc, Yc, Xef, Yef, Xnf, Ynf
        )
        np.testing.assert_allclose(vx_c, np.ones((n, n)), atol=1e-12)
        np.testing.assert_allclose(vy_c, np.zeros((n, n)), atol=1e-12)

    def test_psi_equals_x_gives_vy_minus_one(self):
        """
        psi(x,y) = x  (coeff at (1,0) = 1, all others 0)
        => vx = d psi/dy = 0,  vy = -d psi/dx = -1
        """
        n = 8
        idx = monomial_multiindices(2)
        pos_10 = idx.index((1, 0))
        coeffs = np.zeros(len(idx), dtype=np.float64)
        coeffs[pos_10] = 1.0
        Xc, Yc, Xef, Yef, Xnf, Ynf = self._make_grid_parts(n)
        (vx_c, vy_c), _, _ = eval_velocity_from_streamfunc_coeffs(
            coeffs, idx, Xc, Yc, Xef, Yef, Xnf, Ynf
        )
        np.testing.assert_allclose(vx_c, np.zeros((n, n)), atol=1e-12)
        np.testing.assert_allclose(vy_c, -np.ones((n, n)), atol=1e-12)

    def test_target_rms_scaling(self):
        """When target_rms is given, RMS speed on centers should match."""
        n = 16
        idx = monomial_multiindices(3)
        rng = np.random.default_rng(0)
        coeffs = rng.standard_normal(len(idx))
        Xc, Yc, Xef, Yef, Xnf, Ynf = self._make_grid_parts(n)
        target = 5.0
        (vx_c, vy_c), _, _ = eval_velocity_from_streamfunc_coeffs(
            coeffs, idx, Xc, Yc, Xef, Yef, Xnf, Ynf, target_rms=target
        )
        rms = float(np.sqrt(np.mean(vx_c**2 + vy_c**2)))
        assert rms == pytest.approx(target, rel=1e-6)

    def test_no_scaling_when_target_none(self):
        """Without target_rms, scale factor is 1 → scaled_coeffs == coeffs."""
        n = 8
        idx = monomial_multiindices(2)
        coeffs = np.array([2.0] + [0.0] * (len(idx) - 1), dtype=np.float64)
        Xc, Yc, Xef, Yef, Xnf, Ynf = self._make_grid_parts(n)
        _, _, scaled_coeffs = eval_velocity_from_streamfunc_coeffs(
            coeffs, idx, Xc, Yc, Xef, Yef, Xnf, Ynf, target_rms=None
        )
        np.testing.assert_allclose(scaled_coeffs, coeffs)

    def test_scaled_coeffs_proportional_to_original(self):
        """scaled_coeffs = scale * original_coeffs (uniform scaling)."""
        n = 16
        idx = monomial_multiindices(2)
        rng = np.random.default_rng(1)
        coeffs = rng.standard_normal(len(idx))
        Xc, Yc, Xef, Yef, Xnf, Ynf = self._make_grid_parts(n)
        target = 3.0
        _, _, sc = eval_velocity_from_streamfunc_coeffs(
            coeffs, idx, Xc, Yc, Xef, Yef, Xnf, Ynf, target_rms=target
        )
        # All ratios sc / coeffs should be identical (where coeffs != 0)
        nonzero = np.abs(coeffs) > 1e-14
        ratios = sc[nonzero] / coeffs[nonzero]
        assert np.allclose(ratios, ratios[0], rtol=1e-10)

    def test_zero_coeffs_gives_zero_velocity(self):
        """All-zero streamfunction → zero velocity everywhere."""
        n = 8
        idx = monomial_multiindices(2)
        coeffs = np.zeros(len(idx), dtype=np.float64)
        Xc, Yc, Xef, Yef, Xnf, Ynf = self._make_grid_parts(n)
        (vx_c, vy_c), (vx_f, vy_f), _ = eval_velocity_from_streamfunc_coeffs(
            coeffs, idx, Xc, Yc, Xef, Yef, Xnf, Ynf
        )
        assert np.all(vx_c == 0.0)
        assert np.all(vy_c == 0.0)
        assert np.all(vx_f == 0.0)
        assert np.all(vy_f == 0.0)

    def test_linearity_in_coefficients(self):
        """Velocity is linear in coefficients: v(a+b) = v(a) + v(b)."""
        n = 8
        idx = monomial_multiindices(2)
        rng = np.random.default_rng(42)
        a = rng.standard_normal(len(idx))
        b = rng.standard_normal(len(idx))
        Xc, Yc, Xef, Yef, Xnf, Ynf = self._make_grid_parts(n)

        (vxa, vya), _, _ = eval_velocity_from_streamfunc_coeffs(
            a, idx, Xc, Yc, Xef, Yef, Xnf, Ynf
        )
        (vxb, vyb), _, _ = eval_velocity_from_streamfunc_coeffs(
            b, idx, Xc, Yc, Xef, Yef, Xnf, Ynf
        )
        (vxab, vyab), _, _ = eval_velocity_from_streamfunc_coeffs(
            a + b, idx, Xc, Yc, Xef, Yef, Xnf, Ynf
        )
        np.testing.assert_allclose(vxab, vxa + vxb, atol=1e-12)
        np.testing.assert_allclose(vyab, vya + vyb, atol=1e-12)

    def test_higher_degree_streamfunction(self):
        """n_sf=4 should run without error and return correct shapes."""
        n = 16
        idx = monomial_multiindices(4)
        rng = np.random.default_rng(3)
        coeffs = rng.standard_normal(len(idx))
        Xc, Yc, Xef, Yef, Xnf, Ynf = self._make_grid_parts(n)
        (vx_c, vy_c), (vx_f, vy_f), sc = eval_velocity_from_streamfunc_coeffs(
            coeffs, idx, Xc, Yc, Xef, Yef, Xnf, Ynf
        )
        assert vx_c.shape == (n, n)
        assert vy_c.shape == (n, n)
        assert vx_f.shape == (n - 1, n)
        assert vy_f.shape == (n, n - 1)
        assert sc.shape == coeffs.shape

    def test_quadratic_streamfunction_vx(self):
        """
        psi = x*y  → coeff (1,1)=1
        vx = d psi/dy = x,  vy = -d psi/dx = -y
        """
        n = 16
        idx = monomial_multiindices(2)
        pos_11 = idx.index((1, 1))
        coeffs = np.zeros(len(idx), dtype=np.float64)
        coeffs[pos_11] = 1.0
        Xc, Yc, Xef, Yef, Xnf, Ynf = self._make_grid_parts(n)
        (vx_c, vy_c), _, _ = eval_velocity_from_streamfunc_coeffs(
            coeffs, idx, Xc, Yc, Xef, Yef, Xnf, Ynf
        )
        np.testing.assert_allclose(vx_c, Xc, atol=1e-12)
        np.testing.assert_allclose(vy_c, -Yc, atol=1e-12)
