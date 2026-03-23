#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create consistent train/val/test splits for multi-modal data, with optional
pre-subsampling (limit total used samples before splitting).

Examples
--------
# Use ALL samples, split 80/10/10:
python do0403_splits.py --data_dir data --val_frac 0.10 --test_frac 0.10

# Use only 1000 of the available samples, chosen at random (seeded), then split:
python create_splits.py --data_dir data --val_frac 0.10 --test_frac 0.10 \
  --limit 1000 --subset-strategy random --seed 123
"""

import argparse
import os
import sys
from argparse import Namespace

import numpy as np

def find_npy_files(data_dir):
    return sorted(
        os.path.join(data_dir, fn)
        for fn in os.listdir(data_dir)
        if fn.endswith(".npy")
    )

def load_and_get_N(path):
    arr = np.load(path, mmap_mode="r")
    if arr.ndim < 1:
        raise ValueError(f"{path}: array must have at least 1 dimension, got shape {arr.shape}")
    N = arr.shape[0]
    if N <= 0:
        raise ValueError(f"{path}: zero samples?")
    return N, arr.shape

def compute_splits_from_index_array(index_array, val_frac, test_frac, rng):
    """Shuffle an index array and split into train/val/test (indices refer to original data)."""
    if not (0.0 <= val_frac < 1.0) or not (0.0 <= test_frac < 1.0):
        raise ValueError("val_frac and test_frac must be in [0,1).")
    if val_frac + test_frac >= 1.0:
        raise ValueError("val_frac + test_frac is too large; must leave samples for train.")

    idx = np.array(index_array, dtype=int)
    rng.shuffle(idx)

    N = idx.shape[0]
    n_val  = int(np.floor(val_frac * N))
    n_test = int(np.floor(test_frac * N))
    n_train = N - n_val - n_test
    if n_train <= 0:
        raise ValueError(f"Not enough samples for train (N_used={N}, n_val={n_val}, n_test={n_test}).")

    val   = idx[:n_val]
    test  = idx[n_val:n_val+n_test]
    train = idx[n_val+n_test:]
    return train, val, test

def main(data_dir='data', splits_file=None, val_frac=0.2, test_frac=0.2, seed=123, overwrite=True,
         check=False, limit=None, subset_strategy='random'):
    rng = np.random.default_rng(seed)

    splits_path = splits_file or os.path.join(data_dir, "05_splits.npz")

    # check
    if check:
            if not os.path.exists(splits_path):
                print(f"ERROR: splits file not found: {splits_path}", file=sys.stderr)
                sys.exit(1)
            data = np.load(splits_path, allow_pickle=True)
            train, val, test = data["train"], data["val"], data["test"]
            meta = data["meta"].item() if "meta" in data else {}
            print(f"Loaded: {splits_path}")
            print(f"  train: {len(train)} samples")
            print(f"  val:   {len(val)} samples")
            print(f"  test:  {len(test)} samples")
            if meta:
                print(f"Meta info:")
                for k, v in meta.items():
                    print(f"  {k}: {v}")
            return

    if not os.path.isdir(data_dir):
        print(f"ERROR: data_dir does not exist: {data_dir}", file=sys.stderr)
        sys.exit(1)

    if os.path.exists(splits_path) and not overwrite:
        print(f"ERROR: splits file already exists: {splits_path} (use --overwrite to replace).", file=sys.stderr)
        sys.exit(1)

    npy_files = find_npy_files(data_dir)
    if not npy_files:
        print(f"ERROR: No .npy files found in {data_dir}", file=sys.stderr)
        sys.exit(1)

    # Consistency check: all .npy files must share the same N
    Ns = []
    shapes = {}
    for f in npy_files:
        N_i, shp = load_and_get_N(f)
        Ns.append(N_i)
        shapes[os.path.basename(f)] = shp

    unique_N = set(Ns)
    if len(unique_N) != 1:
        print("ERROR: Inconsistent number of samples across files:", file=sys.stderr)
        for f in npy_files:
            print(f"  {os.path.basename(f)}: N={shapes[os.path.basename(f)][0]}, shape={shapes[os.path.basename(f)]}",
                  file=sys.stderr)
        sys.exit(1)

    N_all = unique_N.pop()
    print(f"Found {len(npy_files)} .npy files with consistent N={N_all}:")
    for f in npy_files:
        print(f"  {os.path.basename(f):>28}  shape={shapes[os.path.basename(f)]}")

    # ---- NEW: Pre-subsample before splitting ----
    if limit is None or limit <= 0 or limit >= N_all:
        used_indices = np.arange(N_all, dtype=int)
        N_used = N_all
        limit_effective = None
        subset_strategy = None
    else:
        N_used = int(limit)
        if subset_strategy == "random":
            used_indices = rng.choice(N_all, size=N_used, replace=False)
        else:  # "first"
            used_indices = np.arange(N_used, dtype=int)
        limit_effective = N_used
        subset_strategy = subset_strategy

    # Compute splits on the chosen subset; returned indices still refer to original data
    train, val, test = compute_splits_from_index_array(
        used_indices, val_frac, test_frac, rng
    )

    # Save
    os.makedirs(os.path.dirname(splits_path), exist_ok=True)
    meta = dict(
        N_all=int(N_all),
        N_used=int(N_used),
        val_frac=float(val_frac),
        test_frac=float(test_frac),
        seed=int(seed),
        files=[os.path.basename(f) for f in npy_files],
    )
    if limit_effective is not None:
        meta.update(limit=int(limit_effective), subset_strategy=subset_strategy)

    np.savez(splits_path, train=train, val=val, test=test, meta=meta)
    print(f"\nSaved splits to: {splits_path}")
    print(f"Sizes (on N_used={N_used}): train={len(train)}, val={len(val)}, test={len(test)}")


def get_args() -> Namespace:
    ap = argparse.ArgumentParser(
        description="Create train/val/test splits with consistency checks and optional pre-subsampling.")
    ap.add_argument("--data_dir", type=str, default="data", help="Directory containing .npy modality files.")
    ap.add_argument("--splits_file", type=str, default=None,
                    help="Output splits .npz path (default: <data_dir>/05_splits.npz).")
    ap.add_argument("--val_frac", type=float, default=0.20, help="Validation fraction (default 0.20).")
    ap.add_argument("--test_frac", type=float, default=0.20, help="Test fraction (default 0.20).")
    ap.add_argument("--seed", type=int, default=123, help="RNG seed for shuffling.")
    ap.add_argument("--no-overwrite", dest="overwrite", action="store_false",
                    help="Prevent overwriting an existing splits file (default: overwrite ON).")
    ap.add_argument("--check", action="store_true",
                    help="If set, only check an existing splits file and print sample counts.")
    ap.set_defaults(overwrite=True)
    ap.add_argument("--limit", type=int, default=None,
                    help="If set, pre-select only this many samples before splitting.")
    ap.add_argument("--subset-strategy", type=str, default="random", choices=["random", "first"],
                    help="How to choose the limited subset (default: random).")
    args = ap.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    main(args.data_dir, args.splits_file, args.val_frac, args.test_frac, args.seed, args.overwrite,
         args.check, args.limit, args.subset_strategy)

