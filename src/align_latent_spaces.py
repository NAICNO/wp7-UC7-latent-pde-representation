#!/usr/bin/env python3
"""
Strict shared-latent training: decode FROM the per-item MEAN latent.

Outputs (as requested):
  - models/enc_2_<mod>_ld<ad>.keras
  - models/dec_2_<mod>_ld<ad>.keras
  - latents/lat_2_<mod>_ld<ad>.npy  (per-modality unit-sphere latents)
  - latents/lat_3_ld<ad>.npy        (joint latent: mean over modalities, L2-normalized)

Key design:
  • Encoders map RAW32 → Standardize → Dense(+optional NL) → L2-normalize (unit-sphere in R^ad)
  • Joint latent z̄ = L2norm(mean_k z_k)
  • Reconstruction uses z̄:  x̂_k = D_k(z̄)
  • Curriculum: total loss = (1-λ)*L_self + λ*L_mean + w_align*L_align
      - L_self = avg_k REE(x_k, D_k(z_k))
      - L_mean = avg_k REE(x_k, D_k(z̄))     # target regime (λ → 1)
      - L_align = avg_k REE(z_k, z̄)         # keeps codes tight around z̄
      - λ ramps from 0 → 1 over --joint-anneal-epochs (linear schedule)
  • Unit sphere latents at all times. No CSV logs. Early stopping on val total.

Requirements: tensorflow>=2.12, numpy
"""
import argparse
import json
import os
from typing import Dict, List

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow import keras as tfk  # the shim you’re on

# ---- Keras 3 / tf.keras compatibility shims ----
try:
    # Keras 3
    from keras.saving import register_keras_serializable as _register_ks
except Exception:
    # tf.keras 2.x
    from tensorflow.keras.utils import register_keras_serializable as _register_ks

def register_serializable(*args, **kwargs):
    return _register_ks(*args, **kwargs)

def export_model(model, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Keras 3 prefers model.export(); tf.keras 2.x uses SavedModel via model.save()
    try:
        model.export(path)     # Keras 3
    except Exception:
        model.save(path)       # tf.keras 2.x SavedModel dir

# -----------------------
# Config / constants
# -----------------------
MODALITY_FILES = {
    "u16": "lat_1_u16_ld32.npy",
    "u32": "lat_1_u32_ld32.npy",
    "u64": "lat_1_u64_ld32.npy",
    "u128": "lat_1_u128_ld32.npy",
    "u256": "lat_1_u256_ld32.npy",
    "coeff": "lat_1_coeff_ld32.npy",
}
LATENTS_DIR = "latents"
MODELS_DIR = "models"
SPLITS_PATH_DEFAULT = os.path.join("data", "05_splits.npz")
EPS = 1e-8

# -----------------------
# Utils
# -----------------------

def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


def l2_normalize_tensor(x: tf.Tensor, axis: int = -1, eps: float = EPS) -> tf.Tensor:
    return x / (tf.norm(x, axis=axis, keepdims=True) + eps)


def ree(y_true: tf.Tensor, y_pred: tf.Tensor, eps: float = EPS) -> tf.Tensor:
    num = tf.reduce_sum(tf.square(y_true - y_pred), axis=-1)
    den = tf.reduce_sum(tf.square(y_true), axis=-1) + eps
    return num / den

# -----------------------
# Data IO
# -----------------------

def load_modalities(lat_dir: str) -> Dict[str, np.ndarray]:
    data = {}
    for mod, fn in MODALITY_FILES.items():
        path = os.path.join(lat_dir, fn)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {path}")
        A = np.load(path)
        if A.ndim != 2 or A.shape[1] != 32:
            raise ValueError(f"{path} has shape {A.shape}, expected (N,32)")
        data[mod] = A.astype(np.float32)
    Ns = {X.shape[0] for X in data.values()}
    if len(Ns) != 1:
        raise ValueError(f"All modalities must share N; got {Ns}")
    return data


def load_splits(path: str, N: int):
    if not os.path.exists(path):
        idx = np.arange(N)
        return idx[: int(0.6 * N)], idx[int(0.6 * N) : int(0.8 * N)], idx[int(0.8 * N) :]
    d = np.load(path, allow_pickle=True)
    for t, v, te in [("train", "val", "test"), ("train_idx", "val_idx", "test_idx")]:
        if t in d and v in d and te in d:
            return d[t].astype(int), d[v].astype(int), d[te].astype(int)
    raise KeyError("train/val/test indices not found in splits npz")


def compute_stats(X_by_mod: Dict[str, np.ndarray], train_idx: np.ndarray):
    stats = {}
    for m, X in X_by_mod.items():
        Xtr = X[train_idx]
        mu = Xtr.mean(axis=0).astype(np.float32)
        sd = Xtr.std(axis=0).astype(np.float32)
        sd[sd < 1e-6] = 1.0
        stats[m] = {"mean": mu, "std": sd}
    return stats

# -----------------------
# Custom Layers (serialization-safe)
# -----------------------
@register_serializable(package="custom")
class Standardize(layers.Layer):
    def __init__(self, mean_vec: np.ndarray, std_vec: np.ndarray, name=None):
        super().__init__(name=name, dtype=tf.float32)
        self.mean_vec = np.array(mean_vec, dtype=np.float32)
        self.std_vec = np.array(std_vec, dtype=np.float32)
        self.mu = tf.constant(self.mean_vec)
        self.sd = tf.constant(self.std_vec)
    def call(self, x):
        return (x - self.mu) / self.sd
    def get_config(self):
        return {"mean_vec": self.mean_vec.tolist(), "std_vec": self.std_vec.tolist(), "name": self.name}

@register_serializable(package="custom")
class DeStandardize(layers.Layer):
    def __init__(self, mean_vec: np.ndarray, std_vec: np.ndarray, name=None):
        super().__init__(name=name, dtype=tf.float32)
        self.mean_vec = np.array(mean_vec, dtype=np.float32)
        self.std_vec = np.array(std_vec, dtype=np.float32)
        self.mu = tf.constant(self.mean_vec)
        self.sd = tf.constant(self.std_vec)
    def call(self, x):
        return x * self.sd + self.mu
    def get_config(self):
        return {"mean_vec": self.mean_vec.tolist(), "std_vec": self.std_vec.tolist(), "name": self.name}

@register_serializable(package="custom")
class L2Normalize(layers.Layer):
    def __init__(self, axis=-1, eps=EPS, name=None):
        super().__init__(name=name, dtype=tf.float32)
        self.axis = int(axis)
        self.eps = float(eps)
    def call(self, x):
        return l2_normalize_tensor(x, axis=self.axis, eps=self.eps)
    def get_config(self):
        return {"axis": self.axis, "eps": self.eps, "name": self.name}

@register_serializable(package="custom")
class Blend(layers.Layer):
    """Trainable scalar blend: sigma(alpha)*(a) + (1-sigma(alpha))*(b)"""
    def __init__(self, name=None):
        super().__init__(name=name, dtype=tf.float32)
    def build(self, input_shape):
        self.alpha_logit = self.add_weight(shape=(), initializer="zeros", trainable=True, name="alpha_logit")
        super().build(input_shape)
    def call(self, inputs):
        a, b = inputs
        a_s = tf.sigmoid(self.alpha_logit)
        return a_s * a + (1.0 - a_s) * b
    def get_config(self):
        return {"name": self.name}

def mlp(x, hidden, depth, activation, name_prefix):
    h = x
    for i in range(depth):
        h = layers.Dense(hidden, activation=activation, name=f"{name_prefix}_h{i+1}")(h)
    return h

# -----------------------
# Model builders
# -----------------------

def build_encoder(ad: int, nonlinear: bool, mean_vec: np.ndarray, std_vec: np.ndarray, hidden: int = 64, activation: str = "gelu", depth: int = 1):
    x_raw = layers.Input(shape=(32,), name="enc_in_raw")
    x = Standardize(mean_vec, std_vec, name="standardize")(x_raw)
    z_lin = layers.Dense(ad, use_bias=True, activation=None, name="enc_linear")(x)
    if nonlinear:
        # was: one Dense -> z_nl
        h1 = mlp(z_lin, hidden=hidden, depth=depth, activation=activation, name_prefix="enc")
        z_nl = layers.Dense(ad, use_bias=True, activation=None, name="enc_out")(h1)
        z = Blend(name="blend")([z_lin, z_nl])
    else:
        z = z_lin
    z = L2Normalize(name="enc_norm")(z)
    return keras.Model(x_raw, z, name="encoder")


def build_decoder(ad: int, nonlinear: bool, mean_vec: np.ndarray, std_vec: np.ndarray, hidden: int = 64, activation: str = "gelu", depth: int = 1):
    z = layers.Input(shape=(ad,), name="dec_in")
    if nonlinear:
        h1 = mlp(z, hidden=hidden, depth=depth, activation=activation, name_prefix="dec")
        out_std = layers.Dense(32, activation=None, name="dec_out")(h1)
        skip    = layers.Dense(32, activation=None, name="dec_skip")(z)
        y_std   = layers.Add(name="sum")([out_std, skip])
    else:
        y_std = layers.Dense(32, activation=None, name="dec_linear")(z)
    y_raw = DeStandardize(mean_vec, std_vec, name="destandardize")(y_std)
    return keras.Model(z, y_raw, name="decoder")

# -----------------------
# Evaluation helpers
# -----------------------

def evaluate_losses(models_by_mod, X_by_mod, idx, use_cross: bool, w_align: float, lam_joint: float) -> Dict[str, float]:
    mods = list(models_by_mod.keys())
    vals = {"self": 0.0, "mean": 0.0, "align": 0.0, "cross": 0.0}
    count = 0
    for start in range(0, len(idx), 256):
        bi = idx[start:start+256]
        Y = {m: tf.convert_to_tensor(X_by_mod[m][bi]) for m in mods}
        Z, Yhat_self, Yhat_mean = {}, {}, {}
        for m in mods:
            enc, dec = models_by_mod[m]["enc"], models_by_mod[m]["dec"]
            z = enc(Y[m], training=False)
            Z[m] = z
            Yhat_self[m] = dec(z, training=False)
        z_stack = tf.stack([Z[m] for m in mods], axis=0)
        z_bar = l2_normalize_tensor(tf.reduce_mean(z_stack, axis=0), axis=-1)
        for m in mods:
            Yhat_mean[m] = models_by_mod[m]["dec"](z_bar, training=False)
        self_loss = tf.add_n([tf.reduce_mean(ree(Y[m], Yhat_self[m])) for m in mods]) / len(mods)
        mean_loss = tf.add_n([tf.reduce_mean(ree(Y[m], Yhat_mean[m])) for m in mods]) / len(mods)
        align_loss = tf.add_n([tf.reduce_mean(ree(z_bar, Z[m])) for m in mods]) / len(mods)
        vals["self"] += float(self_loss.numpy())
        vals["mean"] += float(mean_loss.numpy())
        vals["align"] += float(align_loss.numpy())
        count += 1
    for k in vals:
        vals[k] /= max(count, 1)
    vals["total"] = (1 - lam_joint) * vals["self"] + lam_joint * vals["mean"] + w_align * vals["align"]
    return vals

# -----------------------
# Training
# -----------------------

def train_decode_from_mean(X_by_mod, train_idx, val_idx, ad, nonlinear_mode, w_align, lr, epochs, batch_size,
                           activation, seed, stats, joint_anneal_epochs, hidden, coeff_hidden, coeff_depth):
    tf.keras.utils.set_random_seed(seed)
    mods = list(X_by_mod.keys())

    nl_per_mod = {m: False for m in mods}
    if nonlinear_mode == "all":
        nl_per_mod = {m: True for m in mods}
    elif nonlinear_mode == "coeff":
        nl_per_mod["coeff"] = True

    models_by_mod = {}
    opt = keras.optimizers.Adam(learning_rate=lr)

    # Build encoders/decoders
    for m in mods:
        nl = nl_per_mod[m]
        hid = (coeff_hidden if (m == "coeff" and nl) else hidden)
        dep = (coeff_depth if (m == "coeff" and nl) else 1)
        enc = build_encoder(ad=ad, nonlinear=nl, mean_vec=stats[m]["mean"], std_vec=stats[m]["std"], hidden=hid, depth=dep, activation=activation)
        dec = build_decoder(ad=ad, nonlinear=nl, mean_vec=stats[m]["mean"], std_vec=stats[m]["std"], hidden=hid, depth=dep, activation=activation)
        models_by_mod[m] = {"enc": enc, "dec": dec}

    best_val = np.inf
    best_weights = {m: (models_by_mod[m]["enc"].get_weights(), models_by_mod[m]["dec"].get_weights()) for m in mods}
    patience, wait = 100, 0

    # --- NEW: LR-on-plateau scheduler (no CLI) ---
    PLATEAU_PATIENCE = 10   # epochs with no val improvement before we reduce LR
    LR_FACTOR         = 0.5 # multiply LR by this
    MIN_LR            = 1e-6
    MIN_DELTA         = 1e-7  # same idea you used for ES
    plateau_wait      = 0

    train_idx = np.array(train_idx)

    for epoch in range(1, epochs + 1):
        # Linear anneal for λ (0→1 over joint_anneal_epochs)
        lam = 1.0 if joint_anneal_epochs <= 0 else min(1.0, epoch / float(joint_anneal_epochs))
        perm = np.random.permutation(train_idx)
        for start in range(0, len(perm), batch_size):
            bi = perm[start:start+batch_size]
            Y = {m: tf.convert_to_tensor(X_by_mod[m][bi]) for m in mods}
            with tf.GradientTape() as tape:
                Z, Yhat_self, Yhat_mean = {}, {}, {}
                for m in mods:
                    enc, dec = models_by_mod[m]["enc"], models_by_mod[m]["dec"]
                    z = enc(Y[m], training=True)
                    Z[m] = z
                    Yhat_self[m] = dec(z, training=True)
                z_stack = tf.stack([Z[m] for m in mods], axis=0)
                z_bar = l2_normalize_tensor(tf.reduce_mean(z_stack, axis=0), axis=-1)
                for m in mods:
                    Yhat_mean[m] = models_by_mod[m]["dec"](z_bar, training=True)
                self_loss = tf.add_n([tf.reduce_mean(ree(Y[m], Yhat_self[m])) for m in mods]) / len(mods)
                mean_loss = tf.add_n([tf.reduce_mean(ree(Y[m], Yhat_mean[m])) for m in mods]) / len(mods)
                align_loss = tf.add_n([tf.reduce_mean(ree(z_bar, Z[m])) for m in mods]) / len(mods)
                total = (1 - lam) * self_loss + lam * mean_loss + w_align * align_loss
            vars_all = []
            for m in mods:
                vars_all += models_by_mod[m]["enc"].trainable_variables + models_by_mod[m]["dec"].trainable_variables
            grads = tape.gradient(total, vars_all)
            opt.apply_gradients(zip(grads, vars_all))

        train_metrics = evaluate_losses(models_by_mod, X_by_mod, train_idx, use_cross=False, w_align=w_align, lam_joint=lam)
        train_total = train_metrics["total"]
        val_metrics = evaluate_losses(models_by_mod, X_by_mod, val_idx, use_cross=False, w_align=w_align, lam_joint=lam)
        val_total = val_metrics["total"]

        print(f"Epoch {epoch:03d} | λ={lam:.3f} "
            f"| train={train_total:.4f} (self={train_metrics['self']:.4f}, mean={train_metrics['mean']:.4f}, align={train_metrics['align']:.4f}) "
            f"| val={val_total:.4f} (self={val_metrics['self']:.4f}, mean={val_metrics['mean']:.4f}, align={val_metrics['align']:.4f}) "
            f"| patience left={patience - wait}"
        )

        if val_total + MIN_DELTA < best_val:
            best_val = val_total
            wait = 0
            plateau_wait = 0          # reset plateau counter on improvement
            for m in mods:
                best_weights[m] = (
                    models_by_mod[m]["enc"].get_weights(),
                    models_by_mod[m]["dec"].get_weights(),
                )
        else:
            wait += 1
            plateau_wait += 1

            # ---- Reduce LR on plateau ----
            if plateau_wait >= PLATEAU_PATIENCE:
                new_lr = float(opt.learning_rate.numpy()) * LR_FACTOR
                new_lr = max(new_lr, MIN_LR)
                opt.learning_rate.assign(new_lr)
                print(f"[LR scheduler] Plateau {plateau_wait} epochs → reducing LR to {new_lr:.2e}")
                plateau_wait = 0

            # Early stopping (unchanged)
            if wait >= patience:
                print(f"Early stopping at epoch {epoch} (best val={best_val:.4f})")
                break

    for m in mods:
        E_w, D_w = best_weights[m]
        models_by_mod[m]["enc"].set_weights(E_w)
        models_by_mod[m]["dec"].set_weights(D_w)

    return models_by_mod


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ad", type=int, default=32)
    ap.add_argument("--nonlinear", choices=["all","none","coeff"], default="none")
    ap.add_argument("--align-weight", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=10000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--activation", type=str, default="gelu")
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--splits", type=str, default=SPLITS_PATH_DEFAULT)
    ap.add_argument("--joint-anneal-epochs", type=int, default=10, help="epochs to ramp λ from 0→1; set 0 to use λ=1 from start")
    ap.add_argument("--coeff-hidden", type=int, default=128, help="hidden width for coeff enc/dec when nonlinear")
    ap.add_argument("--coeff-depth", type=int, default=2, help="number of hidden layers for coeff enc/dec when nonlinear")

    return ap.parse_args()
# -----------------------
# Main
# -----------------------

def main(ad=32, nonlinear='none', align_weight=1.0, epochs=10000, batch_size=64, hidden=64, activation='gelu',
         lr=1e-2, seed=42, splits=SPLITS_PATH_DEFAULT, joint_anneal_epochs=10, coeff_hidden=128, coeff_depth=2):
    ensure_dirs(LATENTS_DIR, MODELS_DIR)

    # Load RAW data
    X_raw = load_modalities(LATENTS_DIR)
    N = next(iter(X_raw.values())).shape[0]

    # Splits & stats (stats on RAW train)
    train_idx, val_idx, test_idx = load_splits(splits, N)
    stats = compute_stats(X_raw, train_idx)

    # Train enc/dec with decode-from-mean
    models_by_mod = train_decode_from_mean(
        X_by_mod=X_raw,
        train_idx=train_idx,
        val_idx=val_idx,
        ad=ad,
        nonlinear_mode=nonlinear,
        w_align=align_weight,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        activation=activation,
        seed=seed,
        stats=stats,
        joint_anneal_epochs=joint_anneal_epochs,
        hidden=hidden,
        coeff_hidden=coeff_hidden,
        coeff_depth=coeff_depth
    )

    # Save models with enc_2_/dec_2_ naming
    config = {
        "ad": ad,
        "nonlinear": nonlinear,
        "align_weight": align_weight,
        "epochs": epochs,
        "batch_size": batch_size,
        "hidden": hidden,
        "activation": activation,
        "lr": lr,
        "seed": seed,
        "modalities": list(MODALITY_FILES.keys()),
        "joint_anneal_epochs": joint_anneal_epochs,
    }
    with open(os.path.join(MODELS_DIR, "config_decode_from_mean.json"), "w") as f:
        json.dump(config, f, indent=2)

    for m in MODALITY_FILES.keys():
        enc = models_by_mod[m]["enc"]
        dec = models_by_mod[m]["dec"]

        export_dir = "exports"
        os.makedirs(export_dir, exist_ok=True)

        enc_path_keras = os.path.join(MODELS_DIR, f"enc_2_{m}_ld{ad}.keras")
        dec_path_keras = os.path.join(MODELS_DIR, f"dec_2_{m}_ld{ad}.keras")
        enc.save(enc_path_keras)
        dec.save(dec_path_keras)

        # Prefer Keras 3 .export() to produce clean, colon-free SavedModels
        try:
            enc.export(os.path.join(export_dir, f"enc_2_{m}_ld{ad}"))
            dec.export(os.path.join(export_dir, f"dec_2_{m}_ld{ad}"))
            print(f"[OK] Exported modern SavedModels for {m}")
        except Exception as e:
            print(f"[WARN] Could not export {m}: {e}")

    # Export per-modality lat_2 and the joint lat_3
    Z_all = []
    for m in MODALITY_FILES.keys():
        Zm = models_by_mod[m]["enc"](X_raw[m], training=False).numpy().astype(np.float32)
        np.save(os.path.join(LATENTS_DIR, f"lat_2_{m}_ld{ad}.npy"), Zm)
        print(f"Saved latents/lat_2_{m}_ld{ad}.npy shape={Zm.shape}")
        Z_all.append(Zm)
    Z_all = np.stack(Z_all, axis=0)              # (M, N, ad)
    Z_joint = Z_all.mean(axis=0)
    Z_joint = Z_joint / (np.linalg.norm(Z_joint, axis=1, keepdims=True) + 1e-8)
    np.save(os.path.join(LATENTS_DIR, f"lat_3_ld{ad}.npy"), Z_joint.astype(np.float32))
    print(f"Saved latents/lat_3_ld{ad}.npy shape={Z_joint.shape}")


if __name__ == "__main__":
    args=get_args()
    main()

