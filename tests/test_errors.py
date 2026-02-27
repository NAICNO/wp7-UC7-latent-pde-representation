"""
Tests for pure numpy utility functions sourced from compute_errors.py /
evaluate_decoder_end_to_end.py:

  - ree_rel_sq(y_true, y_pred, eps)   — from compute_errors
  - normalize_field(arr)              — from compute_errors
  - add_channel_dim(x)                — from compute_errors
  - batched_iter(idx, batch_size)     — from compute_errors

These functions have no TensorFlow dependency and can be tested in isolation.
"""

import importlib
import sys
import types

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Import the pure-numpy helpers without triggering the top-level TF import.
# We load the module source manually and execute only the needed functions.
# ---------------------------------------------------------------------------

def _load_tf_module(mod_name, src_path):
    """
    Load a module that imports tensorflow at the top level by replacing
    tensorflow with a lightweight stub before importing.
    """
    # Create a minimal TF stub so the import doesn't crash
    tf_stub = types.ModuleType("tensorflow")
    tf_stub.config = types.SimpleNamespace(
        list_physical_devices=lambda *a, **k: [],
        experimental=types.SimpleNamespace(set_memory_growth=lambda *a, **k: None),
    )
    tf_stub.keras = types.SimpleNamespace(
        mixed_precision=types.SimpleNamespace(set_global_policy=lambda *a, **k: None),
        utils=types.SimpleNamespace(set_random_seed=lambda *a, **k: None),
    )

    keras_stub = types.ModuleType("keras")
    keras_stub.layers = types.SimpleNamespace(TFSMLayer=None)
    keras_stub.models = types.SimpleNamespace(load_model=None)

    old_tf = sys.modules.get("tensorflow")
    old_keras = sys.modules.get("keras")
    sys.modules["tensorflow"] = tf_stub
    sys.modules["keras"] = keras_stub
    sys.modules["keras.layers"] = keras_stub.layers
    sys.modules["keras.models"] = keras_stub.models

    try:
        spec = importlib.util.spec_from_file_location(mod_name, src_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        # Restore original state
        for key, orig in [("tensorflow", old_tf), ("keras", old_keras)]:
            if orig is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = orig

    return mod


import importlib.util
import os

_SRC = os.path.join(os.path.dirname(__file__), "..", "src")

_compute_errors_mod = _load_tf_module(
    "compute_errors",
    os.path.abspath(os.path.join(_SRC, "compute_errors.py")),
)

ree_rel_sq = _compute_errors_mod.ree_rel_sq
normalize_field = _compute_errors_mod.normalize_field
add_channel_dim = _compute_errors_mod.add_channel_dim
batched_iter = _compute_errors_mod.batched_iter


# ============================================================
# ree_rel_sq
# ============================================================

class TestReeRelSq:
    """Tests for ree_rel_sq(y_true, y_pred, eps)."""

    def test_perfect_prediction_zero_error(self):
        y = np.ones((4, 10), dtype=np.float32)
        result = ree_rel_sq(y, y.copy())
        np.testing.assert_allclose(result, 0.0, atol=1e-12)

    def test_returns_vector_of_length_n(self):
        N = 7
        y_true = np.ones((N, 5), dtype=np.float32)
        y_pred = np.zeros((N, 5), dtype=np.float32)
        result = ree_rel_sq(y_true, y_pred)
        assert result.shape == (N,)

    def test_known_value_1d(self):
        """
        y_true = [3, 4]  → ||y||^2 = 25
        y_pred = [0, 0]  → ||e||^2 = 25
        REE = 25 / (25 + eps) ≈ 1
        """
        y_true = np.array([[3.0, 4.0]], dtype=np.float32)
        y_pred = np.array([[0.0, 0.0]], dtype=np.float32)
        result = ree_rel_sq(y_true, y_pred, eps=0.0)
        assert float(result[0]) == pytest.approx(1.0, rel=1e-5)

    def test_known_value_partial_error(self):
        """
        y_true = [0, 1]  → ||y||^2 = 1
        y_pred = [0, 2]  → ||e||^2 = 1
        REE = 1 / (1 + eps) ≈ 1
        """
        y_true = np.array([[0.0, 1.0]], dtype=np.float32)
        y_pred = np.array([[0.0, 2.0]], dtype=np.float32)
        result = ree_rel_sq(y_true, y_pred, eps=0.0)
        assert float(result[0]) == pytest.approx(1.0, rel=1e-5)

    def test_independent_per_sample(self):
        """Each sample's error is computed independently."""
        y_true = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        y_pred = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
        result = ree_rel_sq(y_true, y_pred, eps=0.0)
        assert float(result[0]) == pytest.approx(0.0, abs=1e-12)
        assert float(result[1]) == pytest.approx(1.0, rel=1e-5)

    def test_3d_input_flattened(self):
        """(N, H, W) input should be handled by flattening tail dims."""
        y_true = np.ones((3, 4, 4), dtype=np.float32)
        y_pred = np.zeros((3, 4, 4), dtype=np.float32)
        result = ree_rel_sq(y_true, y_pred, eps=0.0)
        assert result.shape == (3,)
        np.testing.assert_allclose(result, 1.0, rtol=1e-5)

    def test_eps_prevents_division_by_zero(self):
        """When y_true = 0, eps prevents NaN / inf."""
        y_true = np.zeros((2, 5), dtype=np.float32)
        y_pred = np.ones((2, 5), dtype=np.float32)
        result = ree_rel_sq(y_true, y_pred, eps=1e-12)
        assert np.all(np.isfinite(result))

    def test_result_nonnegative(self):
        rng = np.random.default_rng(0)
        y_true = rng.standard_normal((10, 20)).astype(np.float32)
        y_pred = rng.standard_normal((10, 20)).astype(np.float32)
        result = ree_rel_sq(y_true, y_pred)
        assert np.all(result >= 0.0)

    def test_scaling_invariance(self):
        """
        REE = ||alpha * e||^2 / ||alpha * y||^2 = ||e||^2 / ||y||^2 (scale cancels)
        But REE is computed on y_true and (y_pred - y_true), so scaling BOTH by
        the same factor alpha should give the same result.
        """
        rng = np.random.default_rng(1)
        y_true = rng.standard_normal((5, 8)).astype(np.float32)
        y_pred = rng.standard_normal((5, 8)).astype(np.float32)
        alpha = 3.0
        r1 = ree_rel_sq(y_true, y_pred, eps=0.0)
        r2 = ree_rel_sq(alpha * y_true, alpha * y_pred, eps=0.0)
        np.testing.assert_allclose(r1, r2, rtol=1e-5)

    def test_single_sample(self):
        y_true = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        y_pred = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        result = ree_rel_sq(y_true, y_pred)
        assert result.shape == (1,)
        assert float(result[0]) == pytest.approx(0.0, abs=1e-12)

    def test_large_batch(self):
        """Performance check: 10 000 samples should run quickly."""
        rng = np.random.default_rng(2)
        N = 10_000
        y_true = rng.standard_normal((N, 16)).astype(np.float32)
        y_pred = rng.standard_normal((N, 16)).astype(np.float32)
        result = ree_rel_sq(y_true, y_pred)
        assert result.shape == (N,)
        assert np.all(np.isfinite(result))


# ============================================================
# normalize_field
# ============================================================

class TestNormalizeField:
    """Tests for normalize_field(arr)."""

    def test_nhw_unchanged(self):
        """(N,H,W) arrays should pass through unchanged."""
        arr = np.ones((5, 8, 8), dtype=np.float64)
        out = normalize_field(arr)
        assert out.shape == (5, 8, 8)

    def test_nhw1_squeezed_to_nhw(self):
        """(N,H,W,1) → (N,H,W)."""
        arr = np.ones((5, 8, 8, 1), dtype=np.float64)
        out = normalize_field(arr)
        assert out.shape == (5, 8, 8)

    def test_nhwc_with_c_gt_1_unchanged(self):
        """(N,H,W,C) with C>1 should NOT be squeezed."""
        arr = np.ones((5, 8, 8, 3), dtype=np.float64)
        out = normalize_field(arr)
        assert out.shape == (5, 8, 8, 3)

    def test_2d_unchanged(self):
        """2-D arrays (e.g. coefficients (N,C)) pass through untouched."""
        arr = np.ones((10, 20), dtype=np.float32)
        out = normalize_field(arr)
        assert out.shape == (10, 20)

    def test_values_preserved_after_squeeze(self):
        rng = np.random.default_rng(3)
        arr4d = rng.standard_normal((4, 6, 6, 1))
        out = normalize_field(arr4d)
        np.testing.assert_allclose(out, arr4d[..., 0])

    def test_values_preserved_no_squeeze(self):
        arr = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        out = normalize_field(arr)
        np.testing.assert_array_equal(out, arr)


# ============================================================
# add_channel_dim
# ============================================================

class TestAddChannelDim:
    """Tests for add_channel_dim(x)."""

    def test_nhw_becomes_nhw1(self):
        x = np.ones((5, 8, 8), dtype=np.float64)
        out = add_channel_dim(x)
        assert out.shape == (5, 8, 8, 1)

    def test_output_dtype_float32(self):
        x = np.ones((3, 4, 4), dtype=np.float64)
        out = add_channel_dim(x)
        assert out.dtype == np.float32

    def test_values_preserved(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((4, 5, 5)).astype(np.float32)
        out = add_channel_dim(x)
        np.testing.assert_allclose(out[..., 0], x)

    def test_round_trip(self):
        """add_channel_dim then [... 0] should recover the original (within float32)."""
        x = np.arange(30, dtype=np.float32).reshape(2, 3, 5)
        out = add_channel_dim(x)
        np.testing.assert_allclose(out[..., 0], x)

    def test_1d_per_sample(self):
        """(N, D) → (N, D, 1) — just adds last dim."""
        x = np.ones((5, 10), dtype=np.float32)
        out = add_channel_dim(x)
        assert out.shape == (5, 10, 1)


# ============================================================
# batched_iter
# ============================================================

class TestBatchedIter:
    """Tests for batched_iter(idx, batch_size)."""

    def test_yields_batches(self):
        idx = np.arange(10)
        batches = list(batched_iter(idx, 3))
        assert len(batches) == 4  # ceil(10/3)

    def test_each_batch_size(self):
        idx = np.arange(10)
        batches = list(batched_iter(idx, 4))
        assert len(batches[0]) == 4
        assert len(batches[1]) == 4
        assert len(batches[2]) == 2  # remainder

    def test_covers_all_indices(self):
        idx = np.arange(25)
        batches = list(batched_iter(idx, 7))
        combined = np.concatenate(batches)
        np.testing.assert_array_equal(combined, idx)

    def test_batch_size_larger_than_idx(self):
        """Single batch returned when batch_size > len(idx)."""
        idx = np.arange(5)
        batches = list(batched_iter(idx, 100))
        assert len(batches) == 1
        np.testing.assert_array_equal(batches[0], idx)

    def test_empty_index(self):
        idx = np.array([], dtype=int)
        batches = list(batched_iter(idx, 10))
        assert batches == []

    def test_exact_multiple(self):
        idx = np.arange(12)
        batches = list(batched_iter(idx, 4))
        assert len(batches) == 3
        assert all(len(b) == 4 for b in batches)

    def test_batch_size_one(self):
        idx = np.arange(5)
        batches = list(batched_iter(idx, 1))
        assert len(batches) == 5

    def test_preserves_values(self):
        idx = np.array([10, 20, 30, 40, 50])
        batches = list(batched_iter(idx, 2))
        combined = np.concatenate(batches)
        np.testing.assert_array_equal(combined, idx)

    def test_large_index(self):
        N = 10_000
        idx = np.arange(N)
        batches = list(batched_iter(idx, 256))
        total = sum(len(b) for b in batches)
        assert total == N
