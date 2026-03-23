"""
Smoke tests that verify non-TF modules import cleanly and expose
the expected public API.  TF-dependent imports are skipped gracefully.
"""

import importlib
import sys
import types
import os

import pytest

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))


# ============================================================
# Helpers
# ============================================================

def _stub_tensorflow():
    """Install a minimal TF stub so TF-dependent modules can be imported."""
    tf_stub = types.ModuleType("tensorflow")
    tf_stub.config = types.SimpleNamespace(
        list_physical_devices=lambda *a, **k: [],
        experimental=types.SimpleNamespace(set_memory_growth=lambda *a, **k: None),
    )
    tf_stub.keras = types.SimpleNamespace(
        mixed_precision=types.SimpleNamespace(set_global_policy=lambda *a, **k: None),
        utils=types.SimpleNamespace(set_random_seed=lambda *a, **k: None),
        losses=types.SimpleNamespace(Loss=object),
        utils_register=None,
    )
    keras_stub = types.ModuleType("keras")
    keras_stub.layers = types.SimpleNamespace(TFSMLayer=None)
    keras_stub.models = types.SimpleNamespace(load_model=None)
    keras_stub.callbacks = types.SimpleNamespace(
        EarlyStopping=object,
        ReduceLROnPlateau=object,
        Callback=object,
    )

    sys.modules.setdefault("tensorflow", tf_stub)
    sys.modules.setdefault("keras", keras_stub)
    sys.modules.setdefault("keras.layers", keras_stub.layers)
    sys.modules.setdefault("keras.models", keras_stub.models)
    sys.modules.setdefault("keras.callbacks", keras_stub.callbacks)


# ============================================================
# cd2d_streamfunc — pure numpy/scipy
# ============================================================

class TestCd2dStreamfuncImports:
    """Verify cd2d_streamfunc exposes expected callable API."""

    def test_module_importable(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "cd2d_streamfunc", os.path.join(SRC, "cd2d_streamfunc.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert mod is not None

    def test_make_grid_callable(self):
        from cd2d_streamfunc import make_grid
        assert callable(make_grid)

    def test_monomial_multiindices_callable(self):
        from cd2d_streamfunc import monomial_multiindices
        assert callable(monomial_multiindices)

    def test_sample_coeffs_callable(self):
        from cd2d_streamfunc import sample_coeffs
        assert callable(sample_coeffs)

    def test_eval_velocity_callable(self):
        from cd2d_streamfunc import eval_velocity_from_streamfunc_coeffs
        assert callable(eval_velocity_from_streamfunc_coeffs)

    def test_gaussian_all_boundaries_callable(self):
        from cd2d_streamfunc import gaussian_all_boundaries
        assert callable(gaussian_all_boundaries)

    def test_assemble_fv_system_callable(self):
        from cd2d_streamfunc import assemble_fv_system
        assert callable(assemble_fv_system)

    def test_apply_dirichlet_callable(self):
        from cd2d_streamfunc import apply_dirichlet
        assert callable(apply_dirichlet)

    def test_solve_level_callable(self):
        from cd2d_streamfunc import solve_level
        assert callable(solve_level)

    def test_constant_diffusion_callable(self):
        from cd2d_streamfunc import constant_diffusion
        assert callable(constant_diffusion)
        d = constant_diffusion()
        assert callable(d)


# ============================================================
# create_splits — pure numpy
# ============================================================

class TestCreateSplitsImports:
    """Verify create_splits exposes expected callable API."""

    def test_module_importable(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "create_splits_", os.path.join(SRC, "create_splits.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert mod is not None

    def test_compute_splits_callable(self):
        from create_splits import compute_splits_from_index_array
        assert callable(compute_splits_from_index_array)

    def test_find_npy_files_callable(self):
        from create_splits import find_npy_files
        assert callable(find_npy_files)

    def test_load_and_get_n_callable(self):
        from create_splits import load_and_get_N
        assert callable(load_and_get_N)


# ============================================================
# compute_errors / evaluate_decoder — TF-dependent (stub or skip)
# ============================================================

class TestComputeErrorsImports:
    """Verify that the pure-numpy symbols exist in compute_errors."""

    def test_ree_rel_sq_exists(self):
        import importlib.util
        _stub_tensorflow()
        spec = importlib.util.spec_from_file_location(
            "_compute_errors_imp", os.path.join(SRC, "compute_errors.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "ree_rel_sq")
        assert callable(mod.ree_rel_sq)

    def test_normalize_field_exists(self):
        import importlib.util
        _stub_tensorflow()
        spec = importlib.util.spec_from_file_location(
            "_compute_errors_imp2", os.path.join(SRC, "compute_errors.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "normalize_field")
        assert callable(mod.normalize_field)

    def test_add_channel_dim_exists(self):
        import importlib.util
        _stub_tensorflow()
        spec = importlib.util.spec_from_file_location(
            "_compute_errors_imp3", os.path.join(SRC, "compute_errors.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "add_channel_dim")
        assert callable(mod.add_channel_dim)

    def test_batched_iter_exists(self):
        import importlib.util
        _stub_tensorflow()
        spec = importlib.util.spec_from_file_location(
            "_compute_errors_imp4", os.path.join(SRC, "compute_errors.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "batched_iter")
        assert callable(mod.batched_iter)


# ============================================================
# TF import itself — skip gracefully if not available
# ============================================================

class TestTensorflowAvailability:
    """Probe TF availability without failing the suite if absent."""

    def test_tensorflow_importable_or_skip(self):
        try:
            import tensorflow  # noqa: F401
            assert True  # TF is present
        except ImportError:
            pytest.skip("TensorFlow not installed in this environment")

    def test_keras_importable_or_skip(self):
        try:
            import keras  # noqa: F401
            assert True
        except ImportError:
            pytest.skip("Keras not installed in this environment")

    def test_pandas_importable(self):
        """pandas is used by compute_errors; verify it loads."""
        try:
            import pandas  # noqa: F401
            assert True
        except ImportError:
            pytest.skip("pandas not installed")
