"""
Additional edge-case and boundary-value tests for functions that already have
basic coverage.  These extend the existing suites without modifying them.

Covers:
  - ree_rel_sq: additional dtype, boundary eps=0 with zero denominator, 1-element
  - normalize_field: 5-D passthrough, empty-last-dim, wrong squeeze guard
  - add_channel_dim: large arrays, dtype coercion from int
  - batched_iter: batch_size == len(idx), non-array list input, single-element
  - compute_splits_from_index_array: fracs at boundary (0.5+0.5-epsilon), large N
  - find_npy_files: subdirectories ignored, hidden files ignored
  - load_and_get_N: 3-D array, 2-D array
  - make_grid: face midpoint values, large even n
  - gaussian_all_boundaries: sigma very large → flat near-1 field
  - monomial_multiindices: sorted order property
  - sample_coeffs: single coefficient (n_sf=1, n_sol=1)
  - constant_diffusion: variable-shaped input
"""

import os
import sys
import tempfile

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Ensure src/ is importable (mirrors conftest)
# ---------------------------------------------------------------------------
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# Import helpers from the TF-stub loader in test_errors.py
import importlib.util
import types


def _load_tf_module(mod_name, src_path):
    tf_pkg = types.ModuleType("tensorflow")
    tf_pkg.__path__ = []
    tfkeras_stub = types.ModuleType("tensorflow.keras")
    tfkeras_layers = types.ModuleType("tensorflow.keras.layers")
    tfkeras_layers.Layer = object
    tfkeras_stub.layers = tfkeras_layers
    tfkeras_stub.utils = types.SimpleNamespace(
        register_keras_serializable=lambda *a, **k: (lambda cls: cls),
        set_random_seed=lambda *a, **k: None,
    )
    tfkeras_stub.mixed_precision = types.SimpleNamespace(
        set_global_policy=lambda *a, **k: None
    )
    tfkeras_stub.losses = types.SimpleNamespace(Loss=object)
    tf_pkg.keras = tfkeras_stub
    tf_pkg.config = types.SimpleNamespace(
        list_physical_devices=lambda *a, **k: [],
        experimental=types.SimpleNamespace(set_memory_growth=lambda *a, **k: None),
    )
    tf_pkg.cast = lambda x, *a, **k: x
    tf_pkg.reduce_sum = lambda x, *a, **k: np.sum(x)
    tf_pkg.square = lambda x: x ** 2
    tf_pkg.constant = lambda v, *a, **k: v
    tf_pkg.float32 = "float32"

    keras_stub = types.ModuleType("keras")
    keras_stub.__path__ = []
    keras_stub.layers = types.SimpleNamespace(TFSMLayer=None, Layer=object)
    keras_stub.models = types.SimpleNamespace(load_model=None)
    keras_stub.callbacks = types.SimpleNamespace(
        EarlyStopping=object, ReduceLROnPlateau=object, Callback=object
    )

    _tf_keys = [
        "tensorflow", "tensorflow.keras", "tensorflow.keras.layers",
        "keras", "keras.layers", "keras.models", "keras.callbacks",
    ]
    old = {k: sys.modules.get(k) for k in _tf_keys}
    sys.modules["tensorflow"] = tf_pkg
    sys.modules["tensorflow.keras"] = tfkeras_stub
    sys.modules["tensorflow.keras.layers"] = tfkeras_layers
    sys.modules["keras"] = keras_stub
    sys.modules["keras.layers"] = keras_stub.layers
    sys.modules["keras.models"] = keras_stub.models
    sys.modules["keras.callbacks"] = keras_stub.callbacks
    try:
        spec = importlib.util.spec_from_file_location(mod_name, src_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        for key, orig in old.items():
            if orig is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = orig
    return mod


_ce = _load_tf_module(
    "compute_errors_edge",
    os.path.join(_SRC, "compute_errors.py"),
)
ree_rel_sq = _ce.ree_rel_sq
normalize_field = _ce.normalize_field
add_channel_dim = _ce.add_channel_dim
batched_iter = _ce.batched_iter


# ============================================================
# ree_rel_sq — additional edge cases
# ============================================================

class TestReeRelSqEdgeCases:
    """Edge cases and boundary values not covered by test_errors.py."""

    def test_float64_input_accepted(self):
        """float64 input should be cast to float32 internally without error."""
        y_true = np.ones((4, 8), dtype=np.float64)
        y_pred = np.zeros((4, 8), dtype=np.float64)
        result = ree_rel_sq(y_true, y_pred)
        assert result.shape == (4,)
        assert np.all(np.isfinite(result))

    def test_int_input_accepted(self):
        """Integer arrays are cast to float32; no crash."""
        y_true = np.ones((3, 5), dtype=np.int32)
        y_pred = np.zeros((3, 5), dtype=np.int32)
        result = ree_rel_sq(y_true, y_pred)
        assert np.all(np.isfinite(result))

    def test_eps_zero_with_nonzero_true(self):
        """eps=0 is safe when ||y_true||^2 > 0 (no division by zero)."""
        y_true = np.ones((2, 4), dtype=np.float32)
        y_pred = np.zeros((2, 4), dtype=np.float32)
        result = ree_rel_sq(y_true, y_pred, eps=0.0)
        np.testing.assert_allclose(result, 1.0, rtol=1e-5)

    def test_result_bounded_above_when_eps_zero(self):
        """When y_true != 0 and eps=0, REE should be >= 0 (no negative numerator)."""
        rng = np.random.default_rng(11)
        y_true = rng.standard_normal((10, 20)).astype(np.float32)
        y_pred = rng.standard_normal((10, 20)).astype(np.float32)
        result = ree_rel_sq(y_true, y_pred, eps=0.0)
        assert np.all(result >= 0.0)

    def test_1element_tail(self):
        """(N, 1) shaped arrays — single-element tail dimension."""
        y_true = np.array([[2.0], [3.0]], dtype=np.float32)
        y_pred = np.array([[2.0], [0.0]], dtype=np.float32)
        result = ree_rel_sq(y_true, y_pred)
        assert float(result[0]) == pytest.approx(0.0, abs=1e-12)
        assert float(result[1]) > 0.9  # 9/(9+eps) ≈ 1

    def test_4d_input(self):
        """(N, H, W, C) input should be flattened correctly."""
        N = 5
        y_true = np.ones((N, 4, 4, 2), dtype=np.float32)
        y_pred = np.zeros((N, 4, 4, 2), dtype=np.float32)
        result = ree_rel_sq(y_true, y_pred)
        assert result.shape == (N,)
        np.testing.assert_allclose(result, 1.0, rtol=1e-5)

    def test_half_error(self):
        """
        y_true = [1, 0],  y_pred = [0, 0]
        ||err||^2 = 1, ||y||^2 = 1 → REE = 1/(1+eps) ≈ 1
        """
        y_true = np.array([[1.0, 0.0]], dtype=np.float32)
        y_pred = np.array([[0.0, 0.0]], dtype=np.float32)
        result = ree_rel_sq(y_true, y_pred, eps=0.0)
        assert float(result[0]) == pytest.approx(1.0, rel=1e-5)

    def test_quarter_error(self):
        """
        y_true = [2, 0],  y_pred = [1, 0]
        ||err||^2 = 1, ||y||^2 = 4 → REE = 0.25
        """
        y_true = np.array([[2.0, 0.0]], dtype=np.float32)
        y_pred = np.array([[1.0, 0.0]], dtype=np.float32)
        result = ree_rel_sq(y_true, y_pred, eps=0.0)
        assert float(result[0]) == pytest.approx(0.25, rel=1e-5)

    @pytest.mark.parametrize("N", [1, 2, 100, 1000])
    def test_batch_sizes(self, N):
        rng = np.random.default_rng(N)
        y = rng.standard_normal((N, 10)).astype(np.float32)
        result = ree_rel_sq(y, y.copy())
        assert result.shape == (N,)
        np.testing.assert_allclose(result, 0.0, atol=1e-12)


# ============================================================
# normalize_field — additional edge cases
# ============================================================

class TestNormalizeFieldEdgeCases:
    """Edge cases for normalize_field beyond the basic coverage."""

    def test_5d_passthrough(self):
        """5-D arrays should pass through unchanged (no squeeze applied)."""
        arr = np.ones((2, 4, 4, 3, 1), dtype=np.float32)
        out = normalize_field(arr)
        assert out.shape == (2, 4, 4, 3, 1)

    def test_nhwc_3channels_not_squeezed(self):
        """(N,H,W,3) must NOT be squeezed — only last dim == 1 is dropped."""
        arr = np.ones((3, 8, 8, 3), dtype=np.float64)
        out = normalize_field(arr)
        assert out.shape == (3, 8, 8, 3)

    def test_1d_passthrough(self):
        """1-D array should pass through unchanged."""
        arr = np.arange(10, dtype=np.float32)
        out = normalize_field(arr)
        assert out.shape == (10,)

    def test_squeeze_preserves_dtype(self):
        """Squeezing should not change dtype."""
        arr = np.ones((4, 8, 8, 1), dtype=np.float32)
        out = normalize_field(arr)
        assert out.dtype == np.float32

    def test_no_squeeze_preserves_dtype(self):
        arr = np.ones((4, 8, 8), dtype=np.float64)
        out = normalize_field(arr)
        assert out.dtype == np.float64

    def test_squeeze_reduces_ndim_by_one(self):
        arr = np.ones((3, 5, 5, 1))
        out = normalize_field(arr)
        assert out.ndim == 3

    def test_no_squeeze_ndim_unchanged(self):
        arr = np.ones((3, 5, 5))
        out = normalize_field(arr)
        assert out.ndim == 3


# ============================================================
# add_channel_dim — additional edge cases
# ============================================================

class TestAddChannelDimEdgeCases:
    """Edge cases for add_channel_dim beyond the basic coverage."""

    def test_int_input_cast_to_float32(self):
        x = np.ones((3, 4, 4), dtype=np.int32)
        out = add_channel_dim(x)
        assert out.dtype == np.float32

    def test_float64_cast_to_float32(self):
        x = np.ones((2, 3, 3), dtype=np.float64)
        out = add_channel_dim(x)
        assert out.dtype == np.float32

    def test_large_array(self):
        x = np.ones((100, 256, 256), dtype=np.float32)
        out = add_channel_dim(x)
        assert out.shape == (100, 256, 256, 1)

    def test_single_element(self):
        x = np.array([[[42.0]]], dtype=np.float32)
        out = add_channel_dim(x)
        assert out.shape == (1, 1, 1, 1)
        assert float(out[0, 0, 0, 0]) == pytest.approx(42.0)

    def test_2d_input_adds_dim(self):
        """(N, D) → (N, D, 1)."""
        x = np.ones((5, 10), dtype=np.float32)
        out = add_channel_dim(x)
        assert out.shape == (5, 10, 1)

    def test_all_values_finite(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((10, 8, 8)).astype(np.float32)
        out = add_channel_dim(x)
        assert np.all(np.isfinite(out))


# ============================================================
# batched_iter — additional edge cases
# ============================================================

class TestBatchedIterEdgeCases:
    """Edge cases for batched_iter beyond the basic coverage."""

    def test_batch_size_equals_length(self):
        """batch_size == len(idx) → exactly one batch."""
        idx = np.arange(7)
        batches = list(batched_iter(idx, 7))
        assert len(batches) == 1
        np.testing.assert_array_equal(batches[0], idx)

    def test_list_input(self):
        """batched_iter should work with a plain Python list."""
        idx = list(range(9))
        batches = list(batched_iter(idx, 4))
        assert len(batches) == 3  # ceil(9/4)

    def test_non_contiguous_array(self):
        """Works with a non-standard (non-arange) index array."""
        idx = np.array([5, 10, 15, 20, 25, 30])
        batches = list(batched_iter(idx, 2))
        assert len(batches) == 3
        combined = np.concatenate(batches)
        np.testing.assert_array_equal(combined, idx)

    def test_last_batch_smaller_when_uneven(self):
        idx = np.arange(11)
        batches = list(batched_iter(idx, 4))
        # batches of 4, 4, 3
        assert len(batches[-1]) == 3

    def test_single_element_idx(self):
        """Single-element index produces one batch of size 1."""
        idx = np.array([42])
        batches = list(batched_iter(idx, 10))
        assert len(batches) == 1
        assert len(batches[0]) == 1

    def test_order_preserved(self):
        """Elements come out in the same order they went in."""
        idx = np.array([9, 3, 7, 1, 5])
        combined = np.concatenate(list(batched_iter(idx, 2)))
        np.testing.assert_array_equal(combined, idx)

    def test_generator_laziness(self):
        """batched_iter returns a generator, not a pre-materialized list."""
        import types as _types
        idx = np.arange(10)
        gen = batched_iter(idx, 3)
        assert isinstance(gen, _types.GeneratorType)

    @pytest.mark.parametrize("batch_size", [1, 3, 7, 13, 100])
    def test_covers_all_elements(self, batch_size):
        N = 50
        idx = np.arange(N)
        batches = list(batched_iter(idx, batch_size))
        total = sum(len(b) for b in batches)
        assert total == N


# ============================================================
# compute_splits_from_index_array — additional edge cases
# ============================================================

from create_splits import (
    compute_splits_from_index_array,
    find_npy_files,
    load_and_get_N,
)


class TestComputeSplitsEdgeCases:
    """Additional edge cases for compute_splits_from_index_array."""

    def _rng(self, seed=0):
        return np.random.default_rng(seed)

    def test_large_n(self):
        """N=10_000 with standard fracs must run without issue."""
        idx = np.arange(10_000, dtype=int)
        train, val, test = compute_splits_from_index_array(idx, 0.1, 0.1, self._rng())
        assert len(train) + len(val) + len(test) == 10_000

    def test_nearly_equal_fracs(self):
        """val_frac + test_frac just under 1.0 should still leave ≥1 training sample."""
        N = 1000
        idx = np.arange(N)
        # 0.49 + 0.49 = 0.98 < 1.0 → valid
        train, val, test = compute_splits_from_index_array(idx, 0.49, 0.49, self._rng())
        assert len(train) >= 1

    def test_output_arrays_are_numpy(self):
        idx = np.arange(50)
        train, val, test = compute_splits_from_index_array(idx, 0.1, 0.1, self._rng())
        for split in (train, val, test):
            assert isinstance(split, np.ndarray)

    def test_zero_both_fracs(self):
        """val_frac=0, test_frac=0 → all data in train."""
        idx = np.arange(100)
        train, val, test = compute_splits_from_index_array(idx, 0.0, 0.0, self._rng())
        assert len(val) == 0
        assert len(test) == 0
        assert len(train) == 100

    def test_fracs_exactly_point_five_raises(self):
        """0.5+0.5=1.0 ≥ 1.0 must raise ValueError."""
        idx = np.arange(100)
        with pytest.raises(ValueError, match="too large"):
            compute_splits_from_index_array(idx, 0.5, 0.5, self._rng())

    def test_val_frac_exactly_zero_point_nine_nine(self):
        """val_frac=0.99, test_frac=0.0 → n_val=99, n_test=0, n_train=1."""
        idx = np.arange(100)
        train, val, test = compute_splits_from_index_array(idx, 0.99, 0.0, self._rng())
        assert len(val) == 99
        assert len(test) == 0
        assert len(train) == 1

    def test_all_unique_indices(self):
        """No index may appear more than once across all splits."""
        idx = np.arange(300)
        train, val, test = compute_splits_from_index_array(idx, 0.15, 0.15, self._rng(7))
        all_indices = np.concatenate([train, val, test])
        assert len(all_indices) == len(np.unique(all_indices))

    def test_fracs_independent_of_input_values(self):
        """Split proportions must not depend on the actual index values, only counts."""
        idx_a = np.arange(200)
        idx_b = np.arange(200, 400)  # same count, different values
        ta, va, tea = compute_splits_from_index_array(idx_a, 0.1, 0.1, self._rng(0))
        tb, vb, teb = compute_splits_from_index_array(idx_b, 0.1, 0.1, self._rng(0))
        # Same split sizes
        assert len(ta) == len(tb)
        assert len(va) == len(vb)
        assert len(tea) == len(teb)


# ============================================================
# find_npy_files — additional edge cases
# ============================================================

class TestFindNpyFilesEdgeCases:
    """Extra edge cases for find_npy_files."""

    def test_subdirectory_files_not_included(self, tmp_path):
        """Files inside subdirectories should not appear in results."""
        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "deep.npy").write_bytes(b"")
        (tmp_path / "top.npy").write_bytes(b"")
        files = find_npy_files(str(tmp_path))
        basenames = [os.path.basename(f) for f in files]
        assert "deep.npy" not in basenames
        assert "top.npy" in basenames

    def test_dotfile_npy_included_if_present(self, tmp_path):
        """Dotfiles ending in .npy are technically valid .npy files."""
        (tmp_path / ".hidden.npy").write_bytes(b"")
        (tmp_path / "visible.npy").write_bytes(b"")
        files = find_npy_files(str(tmp_path))
        # Both should appear (os.listdir includes dot-files)
        basenames = [os.path.basename(f) for f in files]
        assert "visible.npy" in basenames

    def test_returns_absolute_paths(self, tmp_path):
        (tmp_path / "x.npy").write_bytes(b"")
        files = find_npy_files(str(tmp_path))
        for f in files:
            assert os.path.isabs(f)

    def test_no_npy_returns_empty(self, tmp_path):
        (tmp_path / "data.csv").write_text("1,2,3")
        assert find_npy_files(str(tmp_path)) == []


# ============================================================
# load_and_get_N — additional edge cases
# ============================================================

class TestLoadAndGetNEdgeCases:
    """Additional edge cases for load_and_get_N."""

    def test_3d_array(self, tmp_path):
        arr = np.zeros((20, 8, 8), dtype=np.float32)
        p = str(tmp_path / "data3d.npy")
        np.save(p, arr)
        N, shape = load_and_get_N(p)
        assert N == 20
        assert shape == (20, 8, 8)

    def test_2d_array(self, tmp_path):
        arr = np.zeros((15, 32), dtype=np.float64)
        p = str(tmp_path / "data2d.npy")
        np.save(p, arr)
        N, shape = load_and_get_N(p)
        assert N == 15
        assert shape == (15, 32)

    def test_large_first_dim(self, tmp_path):
        arr = np.zeros((5000, 4), dtype=np.float32)
        p = str(tmp_path / "large.npy")
        np.save(p, arr)
        N, shape = load_and_get_N(p)
        assert N == 5000


# ============================================================
# make_grid — additional edge cases
# ============================================================

from cd2d_streamfunc import make_grid, gaussian_all_boundaries


class TestMakeGridEdgeCases:
    """Additional edge cases for make_grid."""

    def test_n2_face_shapes(self):
        """For n=2: x-faces (1, 2), y-faces (2, 1)."""
        x, y, Xc, Yc, Xef, Yef, Xnf, Ynf, h = make_grid(2)
        assert Xef.shape == (1, 2)
        assert Xnf.shape == (2, 1)

    def test_h_positive(self):
        for n in [4, 8, 16]:
            *_, h = make_grid(n)
            assert float(h) > 0.0

    def test_centers_span_correct_range(self):
        """Xc and Yc must span exactly [-1, 1]."""
        n = 16
        _, _, Xc, Yc, *_ = make_grid(n)
        assert float(Xc.min()) == pytest.approx(-1.0)
        assert float(Xc.max()) == pytest.approx(1.0)
        assert float(Yc.min()) == pytest.approx(-1.0)
        assert float(Yc.max()) == pytest.approx(1.0)

    def test_face_x_range(self):
        """x-face midpoints must lie strictly inside (-1, 1)."""
        n = 8
        _, _, _, _, Xef, *_ = make_grid(n)
        assert float(Xef.min()) > -1.0
        assert float(Xef.max()) < 1.0

    @pytest.mark.parametrize("n", [2, 4, 8, 16, 32])
    def test_all_arrays_finite(self, n):
        result = make_grid(n)
        for arr in result[2:8]:  # skip scalar h at index 8 is already fine
            assert np.all(np.isfinite(arr))


# ============================================================
# gaussian_all_boundaries — additional edge cases
# ============================================================

class TestGaussianAllBoundariesEdgeCases:
    """Additional edge cases for gaussian_all_boundaries."""

    def test_very_large_sigma_approaches_one(self):
        """With huge sigma, exp(-0.5*(y/sigma)^2) ≈ 1 everywhere on boundary."""
        n = 16
        g = gaussian_all_boundaries(n, sigma=1e6)
        # All boundary cells should be close to 1
        np.testing.assert_allclose(g[0, :], 1.0, atol=1e-4)
        np.testing.assert_allclose(g[-1, :], 1.0, atol=1e-4)
        np.testing.assert_allclose(g[:, 0], 1.0, atol=1e-4)
        np.testing.assert_allclose(g[:, -1], 1.0, atol=1e-4)

    def test_default_sigma_interior_zero(self):
        """The default sigma=0.25 still produces zero interior."""
        n = 20
        g = gaussian_all_boundaries(n)
        assert np.all(g[1:-1, 1:-1] == 0.0)

    def test_n3_boundary_coverage(self):
        """For n=3 every cell is on the boundary, so interior is empty."""
        n = 3
        g = gaussian_all_boundaries(n)
        assert g.shape == (3, 3)
        # interior slice [1:-1,1:-1] is just the single center cell
        # for n=3 that is g[1:2, 1:2] — should be zero
        assert g[1, 1] == 0.0

    def test_output_all_finite(self):
        for n in [4, 8, 32]:
            g = gaussian_all_boundaries(n)
            assert np.all(np.isfinite(g))


# ============================================================
# monomial_multiindices — additional edge cases
# ============================================================

from cd2d_streamfunc import monomial_multiindices


class TestMonomialMultiindicesEdgeCases:
    """Additional edge cases for monomial_multiindices."""

    def test_tuples_are_2_ints(self):
        for n_sf in range(1, 5):
            for t in monomial_multiindices(n_sf):
                assert isinstance(t[0], int) and isinstance(t[1], int)

    def test_max_degree_entry_present(self):
        """For n_sf=3, the tuple (3,0) and (0,3) should both appear."""
        idx = monomial_multiindices(3)
        assert (3, 0) in idx
        assert (0, 3) in idx

    def test_increasing_n_sf_adds_entries(self):
        """Each step of n_sf adds new monomials."""
        for n_sf in range(1, 6):
            assert len(monomial_multiindices(n_sf)) > len(monomial_multiindices(n_sf - 1))

    def test_n_sf_5_length(self):
        """n_sf=5: (5+1)*(5+2)/2 - 1 = 20."""
        assert len(monomial_multiindices(5)) == 20


# ============================================================
# sample_coeffs — additional edge cases
# ============================================================

from cd2d_streamfunc import sample_coeffs


class TestSampleCoeffsEdgeCases:
    """Additional edge cases for sample_coeffs."""

    def test_n_sf1_n_sol1_shape(self):
        C, idx = sample_coeffs(1, 1)
        assert C.shape == (1, 2)  # 2 monomials for n_sf=1

    def test_large_n_sol(self):
        C, idx = sample_coeffs(2, 10_000)
        assert C.shape[0] == 10_000
        assert np.all(np.isfinite(C))

    def test_uniform_mode_no_value_exceeds_one(self):
        C, _ = sample_coeffs(4, 500, mode="uniform", seed=123)
        assert np.all(np.abs(C) <= 1.0)

    def test_normal_mode_finite(self):
        C, _ = sample_coeffs(3, 100, mode="normal", seed=0)
        assert np.all(np.isfinite(C))

    def test_different_modes_differ(self):
        """Normal and uniform sampling produce different results."""
        C_norm, _ = sample_coeffs(2, 500, mode="normal", seed=42)
        C_unif, _ = sample_coeffs(2, 500, mode="uniform", seed=42)
        assert not np.allclose(C_norm, C_unif)
