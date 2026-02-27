#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
do0507_build_aligned_mean.py

Analyze alignment quality of second-level latents and build the canonical mean.

Computes:
  1) Pairwise REE matrices between modalities:
       RAW:        M_ij = ||Z_i - Z_j||_F^2 / (||Z_i||_F * ||Z_j||_F)
       CENTERED:   same, after subtracting each modality's column-mean
       PROCRUSTES: same, after optimal orthogonal rotation (on centered data)

  2) REE of each modality vs the mean latent:
       RAW:        r_i = ||Z_i - Z̄||_F^2 / (||Z_i||_F * ||Z̄||_F)
       CENTERED:   r_i = ||Z_i^c - Z̄^c||_F^2 / (||Z_i||_F * ||Z̄||_F)
       PROC-TO-MEAN: ||Z_i^c R_i - Z̄^c||_F^2 / (||Z_i||_F * ||Z̄||_F), R_i from train or all

Outputs:
  latents/lat_3_ld{ld}.npy
  results/lat_2_ree_matrix_raw.csv
  results/lat_2_ree_matrix_centered.csv
  results/lat_2_ree_matrix_procrustes.csv
  results/lat_2_ree_to_mean.csv
  (optional) latents/lat_2p_{mod}_ld{ld}.npy  (Procrustes-corrected)
"""

import os
import argparse
import numpy as np
import pandas as pd

DEFAULT_MODS = ["u16","u32","u64","u128","u256","coeff"]

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mods", nargs="+", default=DEFAULT_MODS,
                    help="Modalities to include (default: u16 u32 u64 u128 u256 coeff)")
    ap.add_argument("--ld", type=int, default=32, help="Latent dimensionality (default: 32)")
    ap.add_argument("--latents-dir", default="latents", help="Directory of lat_2_*.npy (default: latents)")
    ap.add_argument("--results-dir", default="results", help="Where to write CSVs (default: results)")
    ap.add_argument("--splits", default="data/05_splits.npz",
                    help="NPZ with train/val/test; if present, Procrustes uses TRAIN (default: data/05_splits.npz)")
    ap.add_argument("--save-procrustes-corrected", action="store_true",
                    help="Save Procrustes-corrected latents as lat_2p_{mod}_ld{ld}.npy")
    ap.add_argument("--mean-out", default=None,
                    help="Path for canonical mean latent (default: latents/foo_lat_3_ld{ld}.npy)")
    return ap.parse_args()

def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

def load_latents(mods, lat_dir, ld):
    arrays, names, files = [], [], []
    for m in mods:
        fn = os.path.join(lat_dir, f"lat_2_{m}_ld{ld}.npy")
        if not os.path.exists(fn):
            print(f"⚠️  Missing {fn} — skipping {m}")
            continue
        A = np.load(fn)
        if A.ndim != 2 or A.shape[1] != ld:
            raise ValueError(f"{fn} has shape {A.shape}, expected (N,{ld})")
        arrays.append(A.astype(np.float32))
        names.append(m)
        files.append(fn)
    if len(arrays) < 2:
        raise RuntimeError("Need at least two latent files to compare")
    # sanity: same N
    Ns = {A.shape[0] for A in arrays}
    if len(Ns) != 1:
        raise ValueError(f"All arrays must share N; got Ns = {Ns}")
    N = next(iter(Ns))
    return arrays, names, files, N

def ree(A, B, eps=1e-12):
    num = np.sum((A - B) ** 2)
    den = np.linalg.norm(A) * np.linalg.norm(B) + eps
    return num / den

def center(A):
    return A - A.mean(axis=0, keepdims=True)

def procrustes_R(Ac, Bc):
    # A^T B = U S V^T; R = U V^T ; fix improper rotation if det<0
    U, _, Vt = np.linalg.svd(Ac.T @ Bc, full_matrices=False)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt
    return R

def pairwise_matrix(arrs, metric_fn):
    M = len(arrs)
    Mtx = np.zeros((M, M), dtype=np.float32)
    for i in range(M):
        for j in range(i+1, M):
            Mtx[i, j] = Mtx[j, i] = metric_fn(arrs[i], arrs[j])
    return Mtx

def main(mods=("u16","u32","u64","u128","u256","coeff"), ld=32, latents_dir="latents", results_dir="results",
         splits="data/05_splits.npz", save_procrustes_corrected=False, mean_out=None):
    ensure_dirs(latents_dir, results_dir)

    arrays, mods, files, N = load_latents(mods, latents_dir, ld)
    M = len(arrays)
    print(f"Loaded {M} modalities, each with shape ({N},{ld})")

    # ===== Pairwise RAW REE matrix =====
    M_raw = pairwise_matrix(arrays, ree)
    df_raw = pd.DataFrame(M_raw, index=mods, columns=mods)
    raw_csv = os.path.join(results_dir, "lat_2_ree_matrix_raw.csv")
    #df_raw.to_csv(raw_csv, float_format="%.6g")
    print("\nPairwise latent REE matrix (RAW):")
    print(df_raw.round(6))
    #print(f"Saved to {raw_csv}")

    # ===== Centered variants =====
    arrays_c = [center(A) for A in arrays]
    M_cent = pairwise_matrix(arrays_c, lambda A, B: ree(A, B))
    df_cent = pd.DataFrame(M_cent, index=mods, columns=mods)
    cent_csv = os.path.join(results_dir, "lat_2_ree_matrix_centered.csv")
    #df_cent.to_csv(cent_csv, float_format="%.6g")
    print("\nPairwise latent REE matrix (CENTERED):")
    print(df_cent.round(6))
    #print(f"Saved to {cent_csv}")

    # ===== Procrustes (on centered arrays) =====
    def ree_procrustes(Ac, Bc, A_orig, B_orig):
        R = procrustes_R(Ac, Bc)
        num = np.sum((Ac @ R - Bc) ** 2)
        den = np.linalg.norm(A_orig) * np.linalg.norm(B_orig) + 1e-12
        return num / den

    M_pro = np.zeros((M, M), dtype=np.float32)
    for i in range(M):
        for j in range(i+1, M):
            M_pro[i, j] = M_pro[j, i] = ree_procrustes(arrays_c[i], arrays_c[j], arrays[i], arrays[j])
    df_pro = pd.DataFrame(M_pro, index=mods, columns=mods)
    pro_csv = os.path.join(results_dir, "lat_2_ree_matrix_procrustes.csv")
    #df_pro.to_csv(pro_csv, float_format="%.6g")
    print("\nPairwise latent REE matrix (PROCRUSTES after centering):")
    print(df_pro.round(6))
    #print(f"Saved to {pro_csv}")

    # ===== Canonical mean latent (RAW mean) =====
    stack = np.stack(arrays, axis=0)        # (M,N,D)
    mean_lat = np.mean(stack, axis=0)       # (N,D)
    mean_out = mean_out or os.path.join(latents_dir, f"foo_lat_3_ld{ld}.npy")
    #np.save(mean_out, mean_lat)
    #print(f"\nSaved canonical aligned latent: {mean_out}  shape={mean_lat.shape}")

    # ===== REE to mean: RAW, CENTERED, PROCRUSTES-to-mean =====
    mean_c = center(mean_lat)
    ree_raw_to_mean = []
    ree_cent_to_mean = []
    ree_proc_to_mean = []

    # choose train split for Procrustes (if provided), else use all rows
    idx_tr = None
    if os.path.exists(splits):
        try:
            d = np.load(splits, allow_pickle=True)
            idx_tr = d["train"].astype(int)
            print(f"Using TRAIN split of size {idx_tr.size} for Procrustes-to-mean.")
        except Exception as e:
            print(f"⚠️ Could not read {splits}: {e}. Falling back to ALL rows for Procrustes.")
            idx_tr = None

    # norms for denominator should use RAW norms for consistency
    norm_mean_raw = np.linalg.norm(mean_lat) + 1e-12

    # Prepare train-mean for rotation if train split is used
    if idx_tr is not None:
        mean_tr = mean_lat[idx_tr]
        mean_tr_c = center(mean_tr)
    else:
        mean_tr_c = mean_c  # fall back to all

    # compute vectors and optionally write Procrustes-corrected latents
    corrected = []
    for i, (A, A_c, m) in enumerate(zip(arrays, arrays_c, mods)):
        # RAW to mean
        ree_raw_to_mean.append(ree(A, mean_lat))

        # CENTERED to centered mean
        ree_cent_to_mean.append(ree(A_c, mean_c))

        # Procrustes-to-mean (fit R on train or all, apply to ALL rows)
        if idx_tr is not None:
            A_tr = A[idx_tr]
            A_tr_c = center(A_tr)
            R = procrustes_R(A_tr_c, mean_tr_c)
        else:
            R = procrustes_R(A_c, mean_c)

        num = np.sum((A_c @ R - mean_c) ** 2)
        den = (np.linalg.norm(A) * norm_mean_raw)
        ree_proc = num / (den + 1e-12)
        ree_proc_to_mean.append(ree_proc)

        if save_procrustes_corrected:
            # Apply affine: center → rotate → uncenter to match mean's location
            # X' = (X - mu_X) R + mu_mean
            mu_A = A.mean(axis=0, keepdims=True)
            mu_M = mean_lat.mean(axis=0, keepdims=True)
            A_corr = (A - mu_A) @ R + mu_M
            corrected.append((m, A_corr))

    # Save corrected latents if requested
    if save_procrustes_corrected and corrected:
        for m, A_corr in corrected:
            outp = os.path.join(latents_dir, f"lat_2p_{m}_ld{ld}.npy")
            np.save(outp, A_corr.astype(np.float32))
            print(f"Saved Procrustes-corrected latents: {outp}  shape={A_corr.shape}")

    # Pack and save REE-to-mean vector
    df_vec = pd.DataFrame({
        "modality": mods,
        "ree_to_mean_raw": np.array(ree_raw_to_mean, dtype=np.float32),
        "ree_to_mean_centered": np.array(ree_cent_to_mean, dtype=np.float32),
        "ree_to_mean_procrustes": np.array(ree_proc_to_mean, dtype=np.float32),
    })
    print(df_vec)
    vec_csv = os.path.join(results_dir, "lat_2_ree_to_mean.csv")
    #df_vec.to_csv(vec_csv, index=False, float_format="%.6g")
    print("\nREE of each modality vs mean latent (RAW / CENTERED / PROCRUSTES):")
    for m, r0, r1, rp in zip(mods, df_vec["ree_to_mean_raw"], df_vec["ree_to_mean_centered"], df_vec["ree_to_mean_procrustes"]):
        print(f"  {m:6s}: {r0:.6f}  /  {r1:.6f}  /  {rp:.6f}")
    #print(f"Saved to {vec_csv}")

if __name__ == "__main__":
    args = parse_args()
    main(args.mods, args.ld, args.latents_dir, args.results_dir, args.splits,
         args.save_procrustes_corrected, args.mean_out)

