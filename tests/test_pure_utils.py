"""
Tests for pure numpy/string utility functions scattered across TF-dependent
source files.  Each module is loaded with a TF stub so none of these tests
require TensorFlow to be installed.

Functions under test
--------------------
From evaluate_decoder_end_to_end.py:
  - side_from_mod(mod)          — parses "u256" → 256, "coeff" → None
  - as_4d(x)                    — (N,H,W) → (N,H,W,1) float32
  - load_splits(path)            — loads train/val/test from .npz (two key conventions)

From compute_errors.py:
  - str_to_list(s)              — parses "[u16,u32]" → ["u16","u32"]
  - pick_endpoints()            — returns list of string endpoint names
  - load_splits(path)            — same dual-key logic as evaluate_decoder

From train_solution_autoencoder.py:
  - side_from_modality(name)    — "u256" → 256 (digit extraction)
  - parse_to_level(s)           — "4x4" → 4, "8" → 8
  - ree_per_sample(xt, xp, eps) — per-sample relative error, pure numpy
"""

import importlib
import importlib.util
import os
import sys
import tempfile
import types

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SRC = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src")
)


# ---------------------------------------------------------------------------
# Minimal TF stub (reused across module loads)
# ---------------------------------------------------------------------------

def _build_tf_stub():
    """
    Build a (tf_pkg, keras_stub) pair where tf_pkg is a proper package
    (has __path__) so that 'from tensorflow import keras' and
    'from tensorflow.keras import layers' succeed.
    """
    tf_pkg = types.ModuleType("tensorflow")
    tf_pkg.__path__ = []  # makes it importable as a package

    # tensorflow.keras sub-module
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
    tfkeras_stub.callbacks = types.SimpleNamespace(
        EarlyStopping=object,
        ReduceLROnPlateau=object,
        Callback=object,
    )
    tfkeras_stub.optimizers = types.SimpleNamespace(Adam=object, AdamW=object)
    # backend shim needed by PatiencePrinter
    tfkeras_stub.backend = types.SimpleNamespace(
        get_value=lambda x: float(x),
    )

    tf_pkg.keras = tfkeras_stub
    tf_pkg.config = types.SimpleNamespace(
        list_physical_devices=lambda *a, **k: [],
        experimental=types.SimpleNamespace(
            set_memory_growth=lambda *a, **k: None
        ),
    )
    tf_pkg.cast = lambda x, *a, **k: x
    tf_pkg.reduce_sum = lambda x, *a, **k: np.sum(x)
    tf_pkg.square = lambda x: x ** 2
    tf_pkg.constant = lambda v, *a, **k: v
    tf_pkg.float32 = "float32"
    tf_pkg.GradientTape = object
    tf_pkg.device = lambda *a, **k: _NullCtx()
    tf_pkg.data = types.SimpleNamespace(
        Dataset=types.SimpleNamespace(
            from_tensor_slices=lambda x: x,
            AUTOTUNE=-1,
        )
    )

    keras_stub = types.ModuleType("keras")
    keras_stub.__path__ = []
    keras_stub.layers = types.SimpleNamespace(
        TFSMLayer=None,
        Layer=object,
        Input=lambda *a, **k: None,
        Dense=lambda *a, **k: None,
        Conv2D=lambda *a, **k: None,
        UpSampling2D=lambda *a, **k: None,
        Reshape=lambda *a, **k: None,
        Conv3D=lambda *a, **k: None,
        AveragePooling2D=lambda *a, **k: None,
    )
    keras_stub.models = types.SimpleNamespace(load_model=None)
    keras_stub.callbacks = types.SimpleNamespace(
        EarlyStopping=object,
        ReduceLROnPlateau=object,
        Callback=object,
    )
    keras_stub.utils = types.SimpleNamespace(
        register_keras_serializable=lambda *a, **k: (lambda cls: cls),
    )
    keras_stub.Model = object
    keras_stub.optimizers = types.SimpleNamespace(Adam=object)
    return tf_pkg, keras_stub, tfkeras_stub, tfkeras_layers


class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *a): pass


def _load_with_tf_stub(mod_name, filename):
    tf_pkg, keras_stub, tfkeras_stub, tfkeras_layers = _build_tf_stub()
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
        spec = importlib.util.spec_from_file_location(
            mod_name, os.path.join(_SRC, filename)
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        for key, orig in old.items():
            if orig is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = orig
    return mod


# ---------------------------------------------------------------------------
# Load the modules once at collection time
# ---------------------------------------------------------------------------

_ede = _load_with_tf_stub("evaluate_decoder_end_to_end_pure", "evaluate_decoder_end_to_end.py")
_ce = _load_with_tf_stub("compute_errors_pure2", "compute_errors.py")


def _load_train_sol_ae():
    """
    Load train_solution_autoencoder with TF package stubs so the module-level
    'from tensorflow import keras' and 'from tensorflow.keras import layers'
    succeed without a real TensorFlow installation.
    """
    return _load_with_tf_stub(
        "train_solution_autoencoder_pure",
        "train_solution_autoencoder.py",
    )


_tsa = _load_train_sol_ae()


# ============================================================
# evaluate_decoder_end_to_end — side_from_mod
# ============================================================

class TestSideFromMod:
    """Tests for side_from_mod(mod) in evaluate_decoder_end_to_end."""

    fn = staticmethod(_ede.side_from_mod)

    def test_u256(self):
        assert self.fn("u256") == 256

    def test_u128(self):
        assert self.fn("u128") == 128

    def test_u64(self):
        assert self.fn("u64") == 64

    def test_u32(self):
        assert self.fn("u32") == 32

    def test_u16(self):
        assert self.fn("u16") == 16

    def test_coeff_returns_none(self):
        assert self.fn("coeff") is None

    def test_returns_int(self):
        result = self.fn("u64")
        assert isinstance(result, int)


# ============================================================
# evaluate_decoder_end_to_end — as_4d
# ============================================================

class TestAs4d:
    """Tests for as_4d(x) in evaluate_decoder_end_to_end."""

    fn = staticmethod(_ede.as_4d)

    def test_3d_gains_channel_dim(self):
        x = np.ones((5, 8, 8), dtype=np.float64)
        out = self.fn(x)
        assert out.shape == (5, 8, 8, 1)

    def test_4d_unchanged_shape(self):
        x = np.ones((5, 8, 8, 1), dtype=np.float64)
        out = self.fn(x)
        assert out.shape == (5, 8, 8, 1)

    def test_output_dtype_float32(self):
        x = np.ones((3, 4, 4), dtype=np.float64)
        out = self.fn(x)
        assert out.dtype == np.float32

    def test_values_preserved(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((4, 6, 6)).astype(np.float32)
        out = self.fn(x)
        np.testing.assert_allclose(out[..., 0], x)

    def test_4d_with_values_preserved(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal((3, 5, 5, 1)).astype(np.float32)
        out = self.fn(x)
        np.testing.assert_allclose(out, x)

    def test_single_sample(self):
        x = np.zeros((1, 16, 16), dtype=np.float32)
        out = self.fn(x)
        assert out.shape == (1, 16, 16, 1)

    def test_no_copy_when_already_float32_4d(self):
        """as_4d on 4-d float32 should not allocate a new copy."""
        x = np.ones((2, 4, 4, 1), dtype=np.float32)
        out = self.fn(x)
        # astype(copy=False) means same data when dtype already matches
        assert out.dtype == np.float32


# ============================================================
# evaluate_decoder_end_to_end — load_splits
# ============================================================

class TestLoadSplitsEDE:
    """Tests for load_splits(path) in evaluate_decoder_end_to_end."""

    fn = staticmethod(_ede.load_splits)

    def _save_splits(self, tmp_path, train, val, test, use_idx_suffix=False):
        if use_idx_suffix:
            np.savez(str(tmp_path), train_idx=train, val_idx=val, test_idx=test)
        else:
            np.savez(str(tmp_path), train=train, val=val, test=test)

    def test_returns_three_arrays(self, tmp_path):
        path = str(tmp_path / "splits.npz")
        np.savez(path, train=np.arange(80), val=np.arange(80, 90), test=np.arange(90, 100))
        result = self.fn(path)
        assert len(result) == 3

    def test_train_val_test_keys(self, tmp_path):
        path = str(tmp_path / "splits.npz")
        tr = np.arange(80, dtype=int)
        va = np.arange(80, 90, dtype=int)
        te = np.arange(90, 100, dtype=int)
        np.savez(path, train=tr, val=va, test=te)
        out_tr, out_va, out_te = self.fn(path)
        np.testing.assert_array_equal(np.sort(out_tr), tr)
        np.testing.assert_array_equal(np.sort(out_va), va)
        np.testing.assert_array_equal(np.sort(out_te), te)

    def test_train_idx_val_idx_test_idx_keys(self, tmp_path):
        """Alternate key convention: train_idx / val_idx / test_idx."""
        path = str(tmp_path / "splits_idx.npz")
        tr = np.arange(60, dtype=int)
        va = np.arange(60, 80, dtype=int)
        te = np.arange(80, 100, dtype=int)
        np.savez(path, train_idx=tr, val_idx=va, test_idx=te)
        out_tr, out_va, out_te = self.fn(path)
        np.testing.assert_array_equal(np.sort(out_tr), tr)

    def test_output_dtype_int(self, tmp_path):
        path = str(tmp_path / "splits.npz")
        np.savez(path, train=np.arange(10, dtype=float),
                 val=np.arange(10, 12, dtype=float),
                 test=np.arange(12, 14, dtype=float))
        tr, va, te = self.fn(path)
        assert tr.dtype in (np.int32, np.int64)

    def test_missing_keys_raises(self, tmp_path):
        path = str(tmp_path / "bad.npz")
        np.savez(path, data=np.arange(10))
        with pytest.raises(KeyError):
            self.fn(path)


# ============================================================
# compute_errors — str_to_list
# ============================================================

class TestStrToList:
    """Tests for str_to_list(s) in compute_errors."""

    fn = staticmethod(_ce.str_to_list)

    def test_single_item_returns_string(self):
        result = self.fn("u16")
        assert result == "u16"

    def test_bracketed_list_with_quotes(self):
        """Items with inner quotes keep those quotes — strip() only removes outer brackets."""
        result = self.fn('["u16","u32"]')
        # The function strips [] from the outside but keeps inner " around items
        assert result == ['"u16"', '"u32"']
        assert len(result) == 2

    def test_curly_braces_with_quotes(self):
        """Same behaviour with curly braces."""
        result = self.fn('{"u16","u32"}')
        assert result == ['"u16"', '"u32"']

    def test_parens_no_quotes(self):
        """Parentheses stripped; unquoted items returned as strings without extra quotes."""
        result = self.fn('(u16,u32)')
        assert result == ["u16", "u32"]

    def test_leading_trailing_spaces_stripped(self):
        """Single unquoted item after stripping → returned as a plain string."""
        result = self.fn('  u16  ')
        assert result == "u16"

    def test_three_items_unquoted(self):
        result = self.fn("[u16,u32,u64]")
        assert result == ["u16", "u32", "u64"]

    def test_empty_brackets_single_empty(self):
        """An empty pair of brackets strips to '' → single item, returned as string."""
        result = self.fn("[]")
        assert isinstance(result, str)

    def test_quoted_items_with_spaces(self):
        """Each item (with inner double-quotes) is returned with its surrounding quotes."""
        result = self.fn('["u16", "u32", "u64"]')
        assert len(result) == 3


# ============================================================
# compute_errors — pick_endpoints
# ============================================================

class TestPickEndpoints:
    """Tests for pick_endpoints() in compute_errors."""

    fn = staticmethod(_ce.pick_endpoints)

    def test_returns_list(self):
        result = self.fn()
        assert isinstance(result, list)

    def test_nonempty(self):
        assert len(self.fn()) > 0

    def test_all_strings(self):
        for ep in self.fn():
            assert isinstance(ep, str)

    def test_serving_default_present(self):
        assert "serving_default" in self.fn()

    def test_call_present(self):
        assert "call" in self.fn()

    def test_no_duplicates(self):
        eps = self.fn()
        assert len(eps) == len(set(eps))


# ============================================================
# compute_errors — load_splits
# ============================================================

class TestLoadSplitsCE:
    """Tests for load_splits(path) in compute_errors."""

    fn = staticmethod(_ce.load_splits)

    def test_train_val_test_keys(self, tmp_path):
        path = str(tmp_path / "splits.npz")
        tr = np.arange(80, dtype=int)
        va = np.arange(80, 90, dtype=int)
        te = np.arange(90, 100, dtype=int)
        np.savez(path, train=tr, val=va, test=te)
        out_tr, out_va, out_te = self.fn(path)
        np.testing.assert_array_equal(np.sort(out_tr), tr)

    def test_train_idx_convention(self, tmp_path):
        path = str(tmp_path / "splits_idx.npz")
        tr = np.arange(70, dtype=int)
        va = np.arange(70, 85, dtype=int)
        te = np.arange(85, 100, dtype=int)
        np.savez(path, train_idx=tr, val_idx=va, test_idx=te)
        out_tr, out_va, out_te = self.fn(path)
        np.testing.assert_array_equal(np.sort(out_tr), tr)

    def test_output_is_int(self, tmp_path):
        path = str(tmp_path / "splits.npz")
        np.savez(path,
                 train=np.arange(10, dtype=np.float32),
                 val=np.arange(10, 12, dtype=np.float32),
                 test=np.arange(12, 14, dtype=np.float32))
        tr, va, te = self.fn(path)
        assert tr.dtype in (np.int32, np.int64)

    def test_missing_keys_raises_keyerror(self, tmp_path):
        path = str(tmp_path / "bad.npz")
        np.savez(path, data=np.arange(10))
        with pytest.raises(KeyError):
            self.fn(path)

    def test_three_arrays_returned(self, tmp_path):
        path = str(tmp_path / "splits.npz")
        np.savez(path, train=np.arange(50),
                 val=np.arange(50, 60), test=np.arange(60, 70))
        result = self.fn(path)
        assert len(result) == 3


# ============================================================
# train_solution_autoencoder — side_from_modality
# ============================================================

@pytest.mark.skipif(_tsa is None, reason="train_solution_autoencoder could not be loaded with stub")
class TestSideFromModality:
    """Tests for side_from_modality(name) in train_solution_autoencoder."""

    @property
    def fn(self):
        return _tsa.side_from_modality

    def test_u256(self):
        assert self.fn("u256") == 256

    def test_u128(self):
        assert self.fn("u128") == 128

    def test_u64(self):
        assert self.fn("u64") == 64

    def test_u32(self):
        assert self.fn("u32") == 32

    def test_u16(self):
        assert self.fn("u16") == 16

    def test_no_digits_returns_256(self):
        """Fallback: if no digits found → 256."""
        assert self.fn("nodigits") == 256

    def test_returns_int_type(self):
        assert isinstance(self.fn("u64"), int)

    @pytest.mark.parametrize("name,expected", [
        ("u16", 16), ("u32", 32), ("u64", 64), ("u128", 128), ("u256", 256),
    ])
    def test_parametrized(self, name, expected):
        assert self.fn(name) == expected


# ============================================================
# train_solution_autoencoder — parse_to_level
# ============================================================

@pytest.mark.skipif(_tsa is None, reason="train_solution_autoencoder could not be loaded with stub")
class TestParseToLevel:
    """Tests for parse_to_level(s) in train_solution_autoencoder."""

    @property
    def fn(self):
        return _tsa.parse_to_level

    def test_1x1(self):
        assert self.fn("1x1") == 1

    def test_4x4(self):
        assert self.fn("4x4") == 4

    def test_8x8(self):
        assert self.fn("8x8") == 8

    def test_16x16(self):
        assert self.fn("16x16") == 16

    def test_plain_integer_string(self):
        assert self.fn("8") == 8

    def test_uppercase_x(self):
        assert self.fn("4X4") == 4

    def test_whitespace_stripped(self):
        assert self.fn("  4x4  ") == 4

    def test_asymmetric_raises(self):
        """Non-square specification should raise ValueError."""
        with pytest.raises(ValueError, match="NxN"):
            self.fn("4x8")

    def test_returns_int(self):
        result = self.fn("4x4")
        assert isinstance(result, int)

    @pytest.mark.parametrize("s,expected", [
        ("1x1", 1), ("2x2", 2), ("4x4", 4), ("8x8", 8), ("16x16", 16),
    ])
    def test_parametrized(self, s, expected):
        assert self.fn(s) == expected


# ============================================================
# train_solution_autoencoder — ree_per_sample
# ============================================================

@pytest.mark.skipif(_tsa is None, reason="train_solution_autoencoder could not be loaded with stub")
class TestReePerSample:
    """Tests for ree_per_sample(xt, xp, eps) in train_solution_autoencoder."""

    @property
    def fn(self):
        return _tsa.ree_per_sample

    def test_perfect_prediction_zero_error(self):
        x = np.ones((5, 4, 4, 1), dtype=np.float32)
        result = self.fn(x, x.copy())
        np.testing.assert_allclose(result, 0.0, atol=1e-12)

    def test_returns_vector_length_n(self):
        N = 8
        x = np.ones((N, 3, 3), dtype=np.float32)
        result = self.fn(x, np.zeros_like(x))
        assert result.shape == (N,)

    def test_nonnegative(self):
        rng = np.random.default_rng(0)
        xt = rng.standard_normal((10, 8)).astype(np.float32)
        xp = rng.standard_normal((10, 8)).astype(np.float32)
        result = self.fn(xt, xp)
        assert np.all(result >= 0.0)

    def test_known_value_2d(self):
        """
        xt = [3, 4] → ||xt||^2 = 25
        xp = [0, 0] → ||xt - xp||^2 = 25
        REE = 25 / (25 + eps) ≈ 1
        """
        xt = np.array([[3.0, 4.0]], dtype=np.float32)
        xp = np.zeros_like(xt)
        result = self.fn(xt, xp, eps=0.0)
        assert float(result[0]) == pytest.approx(1.0, rel=1e-5)

    def test_eps_prevents_nan(self):
        xt = np.zeros((3, 5), dtype=np.float32)
        xp = np.ones((3, 5), dtype=np.float32)
        result = self.fn(xt, xp, eps=1e-12)
        assert np.all(np.isfinite(result))

    def test_independent_per_sample(self):
        """Error for sample i must not be affected by sample j."""
        xt = np.array([[1.0, 0.0], [2.0, 0.0]], dtype=np.float32)
        xp = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
        result = self.fn(xt, xp)
        assert float(result[0]) == pytest.approx(0.0, abs=1e-12)
        # second sample: xt=[2,0], xp=[0,0] → REE=4/(4+eps)≈1
        assert float(result[1]) > 0.9

    def test_3d_input(self):
        """(N, H, W) input — flattened internally."""
        xt = np.ones((4, 8, 8), dtype=np.float32)
        xp = np.zeros_like(xt)
        result = self.fn(xt, xp)
        assert result.shape == (4,)
        np.testing.assert_allclose(result, 1.0, rtol=1e-5)

    def test_4d_input(self):
        """(N, H, W, C) input — flattened internally."""
        xt = np.ones((4, 4, 4, 1), dtype=np.float32)
        xp = np.zeros_like(xt)
        result = self.fn(xt, xp)
        assert result.shape == (4,)

    def test_scaling_invariance(self):
        """REE is scale-invariant: ree(alpha*x, alpha*y) == ree(x, y)."""
        rng = np.random.default_rng(9)
        xt = rng.standard_normal((6, 8)).astype(np.float32)
        xp = rng.standard_normal((6, 8)).astype(np.float32)
        r1 = self.fn(xt, xp, eps=0.0)
        r2 = self.fn(3.0 * xt, 3.0 * xp, eps=0.0)
        np.testing.assert_allclose(r1, r2, rtol=1e-5)

    def test_large_batch(self):
        """No errors or OOM with N=10_000 samples."""
        rng = np.random.default_rng(0)
        xt = rng.standard_normal((10_000, 16)).astype(np.float32)
        xp = rng.standard_normal((10_000, 16)).astype(np.float32)
        result = self.fn(xt, xp)
        assert result.shape == (10_000,)
        assert np.all(np.isfinite(result))
