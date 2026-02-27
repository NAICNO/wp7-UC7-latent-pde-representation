"""
Tests for create_splits.py:
  - compute_splits_from_index_array(index_array, val_frac, test_frac, rng)
  - find_npy_files(data_dir)
  - load_and_get_N(path)
"""

import os
import tempfile

import numpy as np
import pytest

from create_splits import (
    compute_splits_from_index_array,
    find_npy_files,
    load_and_get_N,
)


# ============================================================
# compute_splits_from_index_array
# ============================================================

class TestComputeSplitsFromIndexArray:
    """Unit tests for compute_splits_from_index_array."""

    def _rng(self, seed=0):
        return np.random.default_rng(seed)

    def test_returns_three_arrays(self):
        idx = np.arange(100)
        result = compute_splits_from_index_array(idx, 0.1, 0.1, self._rng())
        assert len(result) == 3

    def test_sizes_sum_to_n(self):
        N = 100
        idx = np.arange(N)
        train, val, test = compute_splits_from_index_array(idx, 0.1, 0.1, self._rng())
        assert len(train) + len(val) + len(test) == N

    def test_val_size_floor(self):
        """n_val = floor(val_frac * N)."""
        N = 100
        idx = np.arange(N)
        _, val, _ = compute_splits_from_index_array(idx, 0.15, 0.10, self._rng())
        assert len(val) == int(np.floor(0.15 * N))

    def test_test_size_floor(self):
        N = 100
        idx = np.arange(N)
        _, _, test = compute_splits_from_index_array(idx, 0.10, 0.20, self._rng())
        assert len(test) == int(np.floor(0.20 * N))

    def test_no_overlap_between_splits(self):
        idx = np.arange(200)
        train, val, test = compute_splits_from_index_array(idx, 0.1, 0.1, self._rng())
        sets = [set(train.tolist()), set(val.tolist()), set(test.tolist())]
        assert sets[0].isdisjoint(sets[1])
        assert sets[0].isdisjoint(sets[2])
        assert sets[1].isdisjoint(sets[2])

    def test_all_indices_present(self):
        """Union of splits must equal the original index set."""
        N = 150
        idx = np.arange(N)
        train, val, test = compute_splits_from_index_array(idx, 0.1, 0.1, self._rng())
        combined = np.sort(np.concatenate([train, val, test]))
        np.testing.assert_array_equal(combined, np.arange(N))

    def test_indices_are_integers(self):
        idx = np.arange(50)
        train, val, test = compute_splits_from_index_array(idx, 0.1, 0.1, self._rng())
        for split in (train, val, test):
            assert split.dtype in (np.int32, np.int64, int)

    def test_reproducible_with_same_rng(self):
        idx = np.arange(100)
        train1, val1, test1 = compute_splits_from_index_array(
            idx, 0.1, 0.1, np.random.default_rng(7)
        )
        train2, val2, test2 = compute_splits_from_index_array(
            idx, 0.1, 0.1, np.random.default_rng(7)
        )
        np.testing.assert_array_equal(train1, train2)
        np.testing.assert_array_equal(val1, val2)
        np.testing.assert_array_equal(test1, test2)

    def test_shuffled_result_differs_from_input_order(self):
        """Splits should not preserve the original input ordering."""
        N = 500
        idx = np.arange(N)
        train, val, test = compute_splits_from_index_array(idx, 0.1, 0.1, self._rng(42))
        all_out = np.concatenate([val, test, train])
        assert not np.array_equal(all_out, np.arange(N))

    def test_zero_val_frac(self):
        """val_frac=0 gives an empty val split."""
        idx = np.arange(100)
        train, val, test = compute_splits_from_index_array(idx, 0.0, 0.1, self._rng())
        assert len(val) == 0
        assert len(train) + len(test) == 100

    def test_zero_test_frac(self):
        """test_frac=0 gives an empty test split."""
        idx = np.arange(100)
        train, val, test = compute_splits_from_index_array(idx, 0.1, 0.0, self._rng())
        assert len(test) == 0

    def test_error_when_fracs_too_large(self):
        idx = np.arange(100)
        with pytest.raises(ValueError, match="too large"):
            compute_splits_from_index_array(idx, 0.5, 0.6, self._rng())

    def test_error_negative_val_frac(self):
        idx = np.arange(100)
        with pytest.raises(ValueError):
            compute_splits_from_index_array(idx, -0.1, 0.1, self._rng())

    def test_error_negative_test_frac(self):
        idx = np.arange(100)
        with pytest.raises(ValueError):
            compute_splits_from_index_array(idx, 0.1, -0.1, self._rng())

    def test_error_val_frac_ge_one(self):
        idx = np.arange(100)
        with pytest.raises(ValueError):
            compute_splits_from_index_array(idx, 1.0, 0.0, self._rng())

    def test_error_not_enough_for_train_guard(self):
        """
        The n_train <= 0 guard in create_splits is a defensive safety net.
        By the inequality floor(v*N) + floor(t*N) <= N*(v+t) < N (when v+t < 1),
        n_train is always >= 1 whenever the preceding fraction checks pass.
        This test documents the design and confirms the preceding guard catches
        the dangerous case (v+t >= 1) before the floor computation.
        """
        idx = np.arange(10)
        # v+t=1.0 is caught by the first guard ("too large"), not the floor guard.
        with pytest.raises(ValueError, match="too large"):
            compute_splits_from_index_array(idx, 0.5, 0.5, self._rng())

    def test_non_contiguous_input_indices(self):
        """Input indices need not be contiguous; all must still appear in output."""
        idx = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45])
        train, val, test = compute_splits_from_index_array(idx, 0.1, 0.1, self._rng())
        combined = set(train.tolist()) | set(val.tolist()) | set(test.tolist())
        assert combined == set(idx.tolist())

    def test_small_n_with_fracs(self):
        """N=10, val_frac=0.1, test_frac=0.1 → floor gives n_val=1, n_test=1."""
        idx = np.arange(10)
        train, val, test = compute_splits_from_index_array(idx, 0.1, 0.1, self._rng())
        assert len(val) == 1
        assert len(test) == 1
        assert len(train) == 8

    @pytest.mark.parametrize("val_frac,test_frac", [
        (0.1, 0.1), (0.2, 0.2), (0.15, 0.15), (0.0, 0.2),
    ])
    def test_parametrized_fracs(self, val_frac, test_frac):
        N = 200
        idx = np.arange(N)
        train, val, test = compute_splits_from_index_array(
            idx, val_frac, test_frac, self._rng()
        )
        assert len(train) + len(val) + len(test) == N


# ============================================================
# find_npy_files
# ============================================================

class TestFindNpyFiles:
    """Unit tests for find_npy_files(data_dir)."""

    def test_returns_sorted_list(self, tmp_path):
        (tmp_path / "b.npy").write_bytes(b"")
        (tmp_path / "a.npy").write_bytes(b"")
        files = find_npy_files(str(tmp_path))
        basenames = [os.path.basename(f) for f in files]
        assert basenames == sorted(basenames)

    def test_only_npy_files(self, tmp_path):
        (tmp_path / "data.npy").write_bytes(b"")
        (tmp_path / "readme.txt").write_text("hello")
        (tmp_path / "script.py").write_text("pass")
        files = find_npy_files(str(tmp_path))
        assert all(f.endswith(".npy") for f in files)
        assert len(files) == 1

    def test_empty_directory(self, tmp_path):
        files = find_npy_files(str(tmp_path))
        assert files == []

    def test_multiple_files(self, tmp_path):
        for name in ["a.npy", "b.npy", "c.npy"]:
            (tmp_path / name).write_bytes(b"")
        files = find_npy_files(str(tmp_path))
        assert len(files) == 3

    def test_full_paths_returned(self, tmp_path):
        (tmp_path / "x.npy").write_bytes(b"")
        files = find_npy_files(str(tmp_path))
        assert all(os.path.isabs(f) for f in files)


# ============================================================
# load_and_get_N
# ============================================================

class TestLoadAndGetN:
    """Unit tests for load_and_get_N(path)."""

    def test_returns_n_and_shape(self, tmp_path):
        arr = np.zeros((50, 16, 16), dtype=np.float64)
        p = str(tmp_path / "data.npy")
        np.save(p, arr)
        N, shape = load_and_get_N(p)
        assert N == 50
        assert shape == (50, 16, 16)

    def test_1d_array(self, tmp_path):
        arr = np.arange(30)
        p = str(tmp_path / "data.npy")
        np.save(p, arr)
        N, shape = load_and_get_N(p)
        assert N == 30
        assert shape == (30,)

    def test_raises_on_zero_samples(self, tmp_path):
        arr = np.zeros((0, 16), dtype=np.float64)
        p = str(tmp_path / "empty.npy")
        np.save(p, arr)
        with pytest.raises(ValueError, match="zero samples"):
            load_and_get_N(p)

    def test_raises_on_scalar(self, tmp_path):
        """A 0-d array should raise ValueError about dimensionality."""
        arr = np.float64(3.14)
        p = str(tmp_path / "scalar.npy")
        np.save(p, arr)
        with pytest.raises(ValueError):
            load_and_get_N(p)
