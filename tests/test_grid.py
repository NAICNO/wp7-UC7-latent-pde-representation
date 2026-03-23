"""
Tests for make_grid() and gaussian_all_boundaries() in cd2d_streamfunc.py.

RED phase: describe expected behaviour, then implement to pass.
"""

import numpy as np
import pytest

from cd2d_streamfunc import make_grid, gaussian_all_boundaries


# ============================================================
# make_grid
# ============================================================

class TestMakeGrid:
    """Unit tests for make_grid(n)."""

    def test_returns_nine_tuple(self):
        result = make_grid(8)
        assert len(result) == 9

    def test_x_y_shapes(self):
        n = 10
        x, y, *_ = make_grid(n)
        assert x.shape == (n,)
        assert y.shape == (n,)

    def test_x_y_range(self):
        """Grid must span [-1, 1]."""
        x, y, *_ = make_grid(16)
        assert float(x[0]) == pytest.approx(-1.0)
        assert float(x[-1]) == pytest.approx(1.0)
        assert float(y[0]) == pytest.approx(-1.0)
        assert float(y[-1]) == pytest.approx(1.0)

    def test_cell_centers_shape(self):
        n = 8
        _, _, Xc, Yc, *_ = make_grid(n)
        assert Xc.shape == (n, n)
        assert Yc.shape == (n, n)

    def test_xface_shape(self):
        """x-faces (east/west) have shape (n-1, n)."""
        n = 8
        result = make_grid(n)
        Xef, Yef = result[4], result[5]
        assert Xef.shape == (n - 1, n)
        assert Yef.shape == (n - 1, n)

    def test_yface_shape(self):
        """y-faces (north/south) have shape (n, n-1)."""
        n = 8
        result = make_grid(n)
        Xnf, Ynf = result[6], result[7]
        assert Xnf.shape == (n, n - 1)
        assert Ynf.shape == (n, n - 1)

    def test_h_is_uniform_spacing(self):
        n = 8
        x, *_, h = make_grid(n)
        expected_h = x[1] - x[0]
        assert float(h) == pytest.approx(float(expected_h))

    def test_h_value_n4(self):
        """For n=4 on [-1,1]: h = 2/3."""
        _, *_, h = make_grid(4)
        assert float(h) == pytest.approx(2.0 / 3.0)

    def test_centers_are_indexing_ij(self):
        """Xc[i, j] should vary along first axis (i) not second."""
        n = 5
        _, _, Xc, Yc, *_ = make_grid(n)
        # All entries in a row have the same x value
        assert np.allclose(Xc[:, 0], Xc[:, 1])
        # All entries in a column have the same y value
        assert np.allclose(Yc[0, :], Yc[1, :])

    def test_face_coords_are_midpoints(self):
        """x-face midpoints in x should lie between adjacent cell centers."""
        n = 8
        x, y, Xc, Yc, Xef, Yef, Xnf, Ynf, h = make_grid(n)
        x_e_expected = 0.5 * (x[:-1] + x[1:])
        assert np.allclose(Xef[:, 0], x_e_expected)

    def test_n1_edge_case(self):
        """n=1 is degenerate (only one grid point, cannot compute h from two points).
        The function raises an IndexError for n=1; document this known limitation."""
        with pytest.raises(IndexError):
            make_grid(1)

    def test_large_n_consistent_h(self):
        """For large n the cell-centered spacing equals 2/(n-1)."""
        n = 128
        x, *_, h = make_grid(n)
        expected = 2.0 / (n - 1)
        assert float(h) == pytest.approx(expected, rel=1e-10)

    @pytest.mark.parametrize("n", [4, 8, 16, 32])
    def test_dtype_float64(self, n):
        x, y, Xc, Yc, Xef, Yef, Xnf, Ynf, h = make_grid(n)
        for arr in [Xc, Yc, Xef, Yef, Xnf, Ynf]:
            assert arr.dtype == np.float64, f"Expected float64, got {arr.dtype}"


# ============================================================
# gaussian_all_boundaries
# ============================================================

class TestGaussianAllBoundaries:
    """Unit tests for gaussian_all_boundaries(n, sigma)."""

    def test_returns_2d_array(self):
        g = gaussian_all_boundaries(8)
        assert g.ndim == 2

    def test_shape_n_n(self):
        n = 12
        g = gaussian_all_boundaries(n)
        assert g.shape == (n, n)

    def test_dtype_float64(self):
        g = gaussian_all_boundaries(8)
        assert g.dtype == np.float64

    def test_all_boundaries_nonzero_at_center(self):
        """Peak of Gaussian is at y=0 (center), so middle of left/right boundary must be 1."""
        n = 101  # odd so exact center index exists
        g = gaussian_all_boundaries(n, sigma=0.25)
        mid = n // 2
        assert float(g[0, mid]) == pytest.approx(1.0, abs=1e-6)
        assert float(g[-1, mid]) == pytest.approx(1.0, abs=1e-6)
        assert float(g[mid, 0]) == pytest.approx(1.0, abs=1e-6)
        assert float(g[mid, -1]) == pytest.approx(1.0, abs=1e-6)

    def test_interior_is_zero(self):
        """Boundary condition is only applied on the edges."""
        n = 16
        g = gaussian_all_boundaries(n)
        interior = g[1:-1, 1:-1]
        assert np.all(interior == 0.0)

    def test_values_nonnegative(self):
        """All Gaussian values must be >= 0."""
        g = gaussian_all_boundaries(16)
        assert np.all(g >= 0.0)

    def test_values_at_most_one(self):
        """Gaussian peak is 1; no value should exceed 1."""
        g = gaussian_all_boundaries(16)
        assert np.all(g <= 1.0 + 1e-12)

    def test_left_right_symmetry(self):
        """Left (x=-1) and right (x=+1) boundaries should be identical (same Gaussian in y)."""
        n = 20
        g = gaussian_all_boundaries(n)
        np.testing.assert_allclose(g[0, :], g[-1, :])

    def test_bottom_top_symmetry(self):
        """Bottom (y=-1) and top (y=+1) boundaries should be identical."""
        n = 20
        g = gaussian_all_boundaries(n)
        np.testing.assert_allclose(g[:, 0], g[:, -1])

    def test_narrow_sigma_decays_fast(self):
        """A very narrow sigma means values far from center are nearly zero."""
        n = 101
        g = gaussian_all_boundaries(n, sigma=0.01)
        # Index 1 away from center: y = -1 + 2*(mid±1)/(n-1)
        assert float(g[0, 0]) < 0.01  # far corner should be tiny

    def test_wide_sigma_is_flatter(self):
        """A wide sigma gives values closer to 1 far from center."""
        n = 101
        g_narrow = gaussian_all_boundaries(n, sigma=0.01)
        g_wide = gaussian_all_boundaries(n, sigma=10.0)
        assert float(g_wide[0, 0]) > float(g_narrow[0, 0])

    def test_corner_is_max_of_contributions(self):
        """Corner values take max of horizontal and vertical Gaussians."""
        n = 101
        g = gaussian_all_boundaries(n, sigma=0.25)
        # Corners are at (x,y) = (±1, ±1); both contributions are exp(-0.5*(1/0.25)^2)
        # = exp(-8) ≈ 3.35e-4; since both are equal, max equals either one
        corner_expected = float(np.exp(-0.5 * (1.0 / 0.25) ** 2))
        assert float(g[0, 0]) == pytest.approx(corner_expected, rel=1e-6)

    @pytest.mark.parametrize("n", [4, 8, 16, 64])
    def test_various_sizes(self, n):
        g = gaussian_all_boundaries(n)
        assert g.shape == (n, n)
        assert np.all(g >= 0.0)
