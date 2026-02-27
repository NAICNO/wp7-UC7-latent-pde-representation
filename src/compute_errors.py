#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
do0512_eval_ree_enc_dec.py

Compute encoding/decoding REE for all modalities and splits.

Encoding REE (per modality m):
  REE_enc[m, split] = mean_i || enc_3_m(x_m[i]) - z[i] ||^2 / || z[i] ||^2

Decoding REE (per modality m):
  REE_dec[m, split] = mean_i || dec_3_m(z[i]) - x_m[i] ||^2 / || x_m[i] ||^2

Assumes:
  data/
    05_splits.npz  (train/val/test or train_idx/val_idx/test_idx)
    05_u16.npy, 05_u32.npy, 05_u64.npy, 05_u128.npy, 05_u256.npy
    05_streamfunction_coeffs.npy
  latents/
    lat_3_ld32.npy (use --ad to adapt)
  models/
    enc_3_<mod>_ld32.keras, dec_3_<mod>_ld32.keras
  exports/ (optional, preferred if present)
    enc_3_<mod>_ld32/ , dec_3_<mod>_ld32/

Usage:
  python do0512_eval_ree_enc_dec.py
"""

import os, re, glob, argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import TFSMLayer
from keras.models import load_model

MODS = ["u16","u32","u64","u128","u256","coeff"]
DATA_PATH = {
    "u16":   "data/05_u16.npy",
    "u32":   "data/05_u32.npy",
    "u64":   "data/05_u64.npy",
    "u128":  "data/05_u128.npy",
    "u256":  "data/05_u256.npy",
    "coeff": "data/05_streamfunction_coeffs.npy",
}
SPLITS_NPZ = "data/05_splits.npz"
LATENTS_PATH_TEMPLATE = "latents/lat_3_ld{ad}.npy"

def str_to_list(s: str):
    s = s.strip('[]{}() ')
    items = s.split(',')
    if len(items) == 1:
        return items[0]
    return [str(i).strip(' ') for i in items]

def load_splits(path=SPLITS_NPZ):
    d = np.load(path, allow_pickle=True)
    for a,b,c in [("train","val","test"), ("train_idx","val_idx","test_idx")]:
        if a in d and b in d and c in d:
            return d[a].astype(int), d[b].astype(int), d[c].astype(int)
    raise KeyError("train/val/test indices not found in splits npz")

def normalize_field(arr):
    # ensure (N,H,W) for images; coefficients stay (N,C)
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = np.asarray(arr[..., 0])
    return arr

def add_channel_dim(x):
    # (N,H,W) -> (N,H,W,1)
    return x[..., None].astype("float32")

def pick_endpoints():
    return ["serving_default","__call__","call","inference","predict"]

def load_from_export_or_keras(prefix, mod, models_dir="models", exports_dir="exports"):
    """
    Try exports/<prefix>_<mod>_ld*/ via TFSMLayer; else fallback to models/*.keras.
    Returns (callable_model, ad, source_str)
    callable_model(x) returns Tensor/ndarray.
    """
    # 1) SavedModel exports (preferred)
    if os.path.isdir(exports_dir):
        pat = re.compile(rf"^{re.escape(prefix)}_{re.escape(mod)}_ld(\d+)$")
        cands = []
        for d in os.listdir(exports_dir):
            m = pat.match(d)
            if m:
                full = os.path.join(exports_dir, d)
                if os.path.isdir(full):
                    cands.append((int(m.group(1)), full))
        if cands:
            cands.sort()
            ad, path = cands[-1]
            last_err = None
            for ep in pick_endpoints():
                try:
                    layer = TFSMLayer(path, call_endpoint=ep)
                    def _call(x, _layer=layer):
                        y = _layer(x, training=False)
                        if isinstance(y, dict):
                            keys = sorted(y.keys())
                            y = y[keys[0]]
                        return y
                    return _call, ad, f"export:{path}:{ep}"
                except Exception as e:
                    last_err = e
                    continue
            raise RuntimeError(f"Failed to load {prefix}_{mod} from export {path}: {last_err}")

    # 2) Fallback to .keras
    if os.path.isdir(models_dir):
        cands = sorted(glob.glob(os.path.join(models_dir, f"{prefix}_{mod}_ld*.keras")))
        if cands:
            fn = cands[-1]
            mdl = load_model(fn, compile=False)
            m = re.search(r"_ld(\d+)\.keras$", fn)
            ad = int(m.group(1)) if m else None
            return mdl, ad, f"keras:{fn}"

    raise FileNotFoundError(f"No export or keras model found for {prefix}_{mod}")

def batched_iter(idx, batch_size):
    for i in range(0, len(idx), batch_size):
        yield idx[i:i+batch_size]

def ree_rel_sq(y_true, y_pred, eps=1e-12):
    """
    Compute relative squared error per sample:
      ||e||^2 / (||y||^2 + eps)
    y_true, y_pred: numpy arrays with same shape per sample; any tail dims.
    Returns vector (N_batch,)
    """
    # flatten tail dims
    yt = y_true.reshape(y_true.shape[0], -1).astype(np.float32)
    yp = y_pred.reshape(y_pred.shape[0], -1).astype(np.float32)
    num = np.sum((yp - yt)**2, axis=1)
    den = np.sum(yt**2, axis=1) + eps
    return num / den

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mods', type=str, default='["u16","u32","u64","u128"]')
    ap.add_argument("--ad", type=int, default=32, help="Aligned latent dimension (matches lat_3 file)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    return ap.parse_args()

def main(mods=("u16","u32","u64","u128"), ad=32, batch=8, seed=7):

    # GPU friendly
    for g in tf.config.list_physical_devices("GPU"):
        try: tf.config.experimental.set_memory_growth(g, True)
        except: pass
    tf.keras.mixed_precision.set_global_policy("float32")
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    # Load splits
    idx_tr, idx_va, idx_te = load_splits(SPLITS_NPZ)

    # Load data arrays
    arrays = {}
    for m in mods:
        arr = np.load(DATA_PATH[m], mmap_mode="r")
        arrays[m] = normalize_field(arr)

    # Align N across everything
    N = min(arr.shape[0] for arr in arrays.values())
    for m in mods:
        if arrays[m].shape[0] != N:
            arrays[m] = arrays[m][:N]
    idx_tr = idx_tr[idx_tr < N]
    idx_va = idx_va[idx_va < N]
    idx_te = idx_te[idx_te < N]
    splits = {"train": idx_tr, "val": idx_va, "test": idx_te}

    # Load latents
    z_path = LATENTS_PATH_TEMPLATE.format(ad=ad)
    if not os.path.exists(z_path):
        raise FileNotFoundError(f"Latent file not found: {z_path}")
    Z = np.load(z_path, mmap_mode="r")
    if Z.shape[0] < N:
        N = Z.shape[0]
        for m in mods:
            arrays[m] = arrays[m][:N]
        idx_tr = idx_tr[idx_tr < N]
        idx_va = idx_va[idx_va < N]
        idx_te = idx_te[idx_te < N]
        splits = {"train": idx_tr, "val": idx_va, "test": idx_te}
    if Z.shape[1] != ad:
        raise ValueError(f"Latent dim mismatch: Z.shape={Z.shape}, --ad={ad}")

    # Load all encoders/decoders (prefer exports)
    encs = {}
    decs = {}
    ad_by_mod = {}
    for m in mods:
        enc, ad_e, src_e = load_from_export_or_keras("enc_3", m)
        dec, ad_d, src_d = load_from_export_or_keras("dec_3", m)
        ad_by_mod[m] = ad_e or ad_d or ad
        print(f"[load] enc_3[{m}] <- {src_e}")
        print(f"[load] dec_3[{m}] <- {src_d}")
        encs[m] = enc
        decs[m] = dec

    # Compute encoding REE and decoding REE
    enc_tab = {m: {} for m in mods}
    dec_tab = {m: {} for m in mods}

    for split_name, idx_vec in splits.items():
        if len(idx_vec) == 0:
            for m in mods:
                enc_tab[m][split_name] = np.nan
                dec_tab[m][split_name] = np.nan
            continue

        # ---- ENCODING: enc_3_m(x_m) vs Z ----
        for m in mods:
            bs = batch
            errs = []
            for bidx in batched_iter(idx_vec, bs):
                if m == "coeff":
                    x = arrays[m][bidx].astype(np.float32)                      # (B,C)
                else:
                    x = add_channel_dim(arrays[m][bidx])                        # (B,H,W,1)
                z_true = Z[bidx].astype(np.float32)                              # (B,ad)
                z_pred = encs[m](x)
                z_pred = np.array(z_pred)
                # reshape to (B,ad)
                z_pred = z_pred.reshape(z_pred.shape[0], -1).astype(np.float32)
                errs.append(ree_rel_sq(z_true, z_pred))
            enc_tab[m][split_name] = float(np.mean(np.concatenate(errs, axis=0)))

        # ---- DECODING: dec_3_m(Z) vs x_m ----
        for m in mods:
            bs = batch
            errs = []
            for bidx in batched_iter(idx_vec, bs):
                z_in = Z[bidx].astype(np.float32)                                # (B,ad)
                y_true = arrays[m][bidx]                                         # (B, H,W) or (B,C)
                y_pred = decs[m](z_in)
                y_pred = np.array(y_pred)
                if m != "coeff":
                    # (B,H,W,1) -> (B,H,W)
                    if y_pred.ndim == 4 and y_pred.shape[-1] == 1:
                        y_pred = y_pred[..., 0]
                errs.append(ree_rel_sq(y_true.astype(np.float32), y_pred.astype(np.float32)))
            dec_tab[m][split_name] = float(np.mean(np.concatenate(errs, axis=0)))

    # Pretty-print ASCII tables
    def print_table(title, table_dict):
        rows = mods
        cols = ["train","val","test"]
        # compute widths
        col_w = [max(len(c), 12) for c in cols]
        row_w = max(9, max(len(r) for r in rows))
        def fmt(v):
            if v != v:  # NaN
                return "   n/a     "
            return f"{v:>12.4f}"
        # header
        print("\n" + title)
        print("-" * (row_w + 3 + sum(w+3 for w in col_w)))
        header = " " * (row_w) + " | " + " | ".join(c.ljust(w) for c,w in zip(cols, col_w))
        print(header)
        print("-" * (row_w + 3 + sum(w+3 for w in col_w)))
        # rows
        for r in rows:
            vals = [fmt(table_dict[r][c]) for c in cols]
            line = r.ljust(row_w) + " | " + " | ".join(v.ljust(w) for v,w in zip(vals, col_w))
            print(line)
        print("-" * (row_w + 3 + sum(w+3 for w in col_w)))

    print_table("ENCODING REE (enc_3_mod(x_mod) vs latent z)", enc_tab)
    print_table("DECODING REE (dec_3_mod(z) vs modality)", dec_tab)
    return pd.DataFrame(enc_tab), pd.DataFrame(dec_tab)

if __name__ == "__main__":
    args = get_args()
    main(args.mods, args.ad, args.batch, args.seed)

