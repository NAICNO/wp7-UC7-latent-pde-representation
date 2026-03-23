#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
do0415_coeff_ae.py
Tiny MLP autoencoder for 14-D streamfunction coefficients.
- Input: c in R^14 (use *post-scaled* coefficients)
- Standardize on train (save mu, sigma)
- Recon loss on standardized coefficients
- Latent z in R^k with whitening penalties (mean/var/cov)
- Saves ONE best encoder/decoder; exports all latents

Outputs:
- models/coeff_encoder_ld{K}.keras
- models/coeff_decoder_ld{K}.keras
- models/coeff_std.json   (train mu, sigma)
- latents/latents_coeff_ld{K}.npy
"""

import os, json, argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ----------------------------
# Utils
# ----------------------------
def ensure_dirs():
    os.makedirs("models", exist_ok=True)
    os.makedirs("latents", exist_ok=True)
    os.makedirs("results", exist_ok=True)

def load_splits(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Splits file not found: {path}")
    if path.endswith(".json"):
        with open(path, "r") as f:
            s = json.load(f)
        return (np.array(s["train"], dtype=int),
                np.array(s["val"], dtype=int),
                np.array(s["test"], dtype=int))
    d = np.load(path, allow_pickle=True)
    return d["train"].astype(int), d["val"].astype(int), d["test"].astype(int)

def standardize_fit(x_train):
    mu = x_train.mean(axis=0)
    sigma = x_train.std(axis=0, ddof=0)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    return mu.astype("float32"), sigma.astype("float32")

def standardize_apply(x, mu, sigma):
    return ((x - mu) / sigma).astype("float32", copy=False)

K = keras.ops

class LatentWhitenPenalty(layers.Layer):
    def __init__(self, lam_mu=1e-3, lam_var=1e-3, lam_cov=1e-3, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.lam_mu = lam_mu
        self.lam_var = lam_var
        self.lam_cov = lam_cov
        self.eps = eps

    def call(self, z):
        # z: (B, k)
        z = K.cast(z, "float32")
        B = K.shape(z)[0]

        # mean ~ 0, var ~ 1
        m = K.mean(z, axis=0)              # (k,)
        v = K.var(z, axis=0)               # (k,)
        loss_mu  = self.lam_mu  * K.sum(K.square(m))
        loss_var = self.lam_var * K.sum(K.square(v - 1.0))

        # covariance off-diagonal ~ 0
        zc  = z - m[None, :]
        Bt  = K.maximum(K.cast(B - 1, "float32"), self.eps)
        cov = K.matmul(K.transpose(zc), zc) / Bt   # (k,k)

        # Build identity mask; use TF here (allowed inside Layer.call)
        k = z.shape[-1] if z.shape[-1] is not None else tf.shape(z)[-1]
        I = tf.eye(k, dtype=z.dtype)

        sum_cov_sq  = K.sum(K.square(cov))           # ||C||_F^2
        sum_diag_sq = K.sum(K.square(cov * I))       # ||diag(C)||_F^2
        offdiag_sq  = sum_cov_sq - sum_diag_sq       # ||offdiag(C)||_F^2
        loss_cov    = self.lam_cov * offdiag_sq

        self.add_loss(loss_mu + loss_var + loss_cov)
        return z  # passthrough

# ----------------------------
# Latent whitening penalties
# ----------------------------
def latent_whiten_penalty(z, lam_mu=1e-3, lam_var=1e-3, lam_cov=1e-3, eps=1e-6):
    """
    Add small penalties so batch latents are ~N(0,I).
    z: (B,k)
    """
    B = tf.cast(tf.shape(z)[0], tf.float32)
    # mean ~ 0
    m = tf.reduce_mean(z, axis=0)                       # (k,)
    loss_mu = lam_mu * tf.reduce_sum(tf.square(m))
    # var ~ 1
    v = tf.math.reduce_variance(z, axis=0)              # (k,)
    loss_var = lam_var * tf.reduce_sum(tf.square(v - 1.0))
    # covariance off-diagonal ~ 0
    zc = z - m[None, :]
    cov = (tf.transpose(zc) @ zc) / (B - 1.0 + eps)     # (k,k)
    offdiag = cov - tf.linalg.diag(tf.linalg.diag_part(cov))
    loss_cov = lam_cov * tf.reduce_sum(tf.square(offdiag))
    return loss_mu + loss_var + loss_cov

# ----------------------------
# Models
# ----------------------------
def build_coeff_encoder(latent_dim, hidden=64, layernorm=True):
    inp = keras.Input(shape=(14,), name="coeff_in")
    x = layers.Dense(hidden, activation="gelu")(inp)
    if layernorm: x = layers.LayerNormalization(axis=-1, center=False, scale=False)(x)
    x = layers.Dense(hidden, activation="gelu")(x)
    if layernorm: x = layers.LayerNormalization(axis=-1, center=False, scale=False)(x)
    z = layers.Dense(latent_dim, activation=None, name=f"z_ld{latent_dim}")(x)
    #return keras.Model(inp, z, name=f"coeff_encoder_ld{latent_dim}")
    return keras.Model(inp, z, name=f"enc_1_coeff_ld{latent_dim}")

def build_coeff_decoder(latent_dim, hidden=64, layernorm=True):
    inp = keras.Input(shape=(latent_dim,), name="z_in")
    x = layers.Dense(hidden, activation="gelu")(inp)
    if layernorm: x = layers.LayerNormalization(axis=-1, center=False, scale=False)(x)
    x = layers.Dense(hidden, activation="gelu")(x)
    if layernorm: x = layers.LayerNormalization(axis=-1, center=False, scale=False)(x)
    out = layers.Dense(14, activation=None, name="coeff_hat_std")(x)  # predicts standardized coeffs
    #return keras.Model(inp, out, name=f"coeff_decoder_ld{latent_dim}")
    return keras.Model(inp, out, name=f"dec_coeff_ld{latent_dim}")

# ----------------------------
# Train / Eval
# ----------------------------
def train(coeff_file, splits, latent, hidden, batch, epochs, patience, lr, lam_mu, lam_var, lam_cov, pred_batch, no_layernorm):
    ensure_dirs()

    # ---- Load data ----
    if not os.path.exists(coeff_file):
        raise FileNotFoundError(f"Missing: {coeff_file}")
    X = np.load(coeff_file).astype("float32")
    if X.ndim != 2 or X.shape[1] != 14:
        raise ValueError(f"Expected (N,14) coefficients, got {X.shape}")

    idx_tr, idx_va, idx_te = load_splits(splits)
    Xtr, Xva, Xte = X[idx_tr], X[idx_va], X[idx_te]

    # ---- Standardize on train split ----
    mu, sigma = standardize_fit(Xtr)
    Xtr_s = standardize_apply(Xtr, mu, sigma)
    Xva_s = standardize_apply(Xva, mu, sigma)
    Xte_s = standardize_apply(Xte, mu, sigma)

    # Save standardization stats
    #std_path = "models/coeff_std.json"
    #with open(std_path, "w") as f:
    #    json.dump({"mu": mu.tolist(), "sigma": sigma.tolist()}, f)
    #print(f"Saved standardization to {std_path}")

    # ---- Build models ----
    enc = build_coeff_encoder(latent, hidden=hidden, layernorm=not no_layernorm)
    dec = build_coeff_decoder(latent, hidden=hidden, layernorm=not no_layernorm)

    # Full model for training
    inp = keras.Input(shape=(14,), name="coeff_std_in")
    z = enc(inp)
    # Add latent regularizers
    z = LatentWhitenPenalty(lam_mu=lam_mu, lam_var=lam_var, lam_cov=lam_cov)(z)
    yhat = dec(z)
    ae = keras.Model(inp, yhat, name=f"coeff_ae_ld{latent}")
    ae.summary()

    # Recon loss = MSE on standardized coeffs
    ae.compile(optimizer=keras.optimizers.Adam(lr), loss="mse")

    # ---- Callbacks ----
    rlr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1)
    es  = keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, min_delta=1e-6,
                                        restore_best_weights=True, verbose=1)

    # ---- Fit ----
    ae.fit(
        Xtr_s, Xtr_s,
        validation_data=(Xva_s, Xva_s),
        epochs=epochs,
        batch_size=batch,
        shuffle=True,
        callbacks=[rlr, es],
        verbose=1
    )
    ae.summary()

    # ---- Save ONE best encoder/decoder ----
    #enc_path = f"models/encoder_coeff_coefflatent{latent}.keras"
    #dec_path = f"models/decoder_coeff_coefflatent{latent}.keras"
    enc_path = f"models/enc_1_coeff_ld{latent}.keras"
    dec_path = f"models/dec_1_coeff_ld{latent}.keras"
    enc.save(enc_path)
    dec.save(dec_path)
    print(f"Saved encoder: {enc_path}")
    print(f"Saved decoder: {dec_path}")

    # ---- Export latents for ALL samples (standardize first) ----
    X_all_s = standardize_apply(X, mu, sigma)
    lat = enc.predict(X_all_s, batch_size=pred_batch, verbose=0).astype("float32", copy=False)
    #lat_path = f"latents/latents_coeff_coefflatent{latent}.npy"
    lat_path = f"latents/lat_1_coeff_ld{latent}.npy"
    np.save(lat_path, lat)
    print(f"Saved latents: {lat_path}  shape={lat.shape}")

    # ---- Optional quick metrics (RMSE on standardized coeffs) ----
    def summarize(name, Xs):
        if Xs.shape[0] == 0:
            return (name, np.nan, np.nan, np.nan, np.nan, np.nan)
        Yp = ae.predict(Xs, batch_size=pred_batch, verbose=0)
        err = np.mean((Yp - Xs)**2, axis=1)   # per-sample MSE in standardized space
        q = np.percentile(err, [0,25,50,75,100])
        return (name, float(err.mean()), float(err.std(ddof=1) if err.size>1 else 0.0),
                float(q[0]), float(q[2]), float(q[4]))
    for nm, Xs in [("train", Xtr_s), ("validate", Xva_s), ("test", Xte_s)]:
        nm, mean, std, p0, p50, p100 = summarize(nm, Xs)
        print(f"{nm:9s}  MSE_std: mean={mean:.6f} std={std:.6f}  p0={p0:.6f}  p50={p50:.6f}  p100={p100:.6f}")

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coeff_file", default="data/05_streamfunction_coeffs.npy", help="Path to .npy with coeff data")
    ap.add_argument("--splits", default="data/05_splits.npz", help="NPZ or JSON with arrays train/val/test")
    ap.add_argument("--latent", type=int, default=32, help="Latent dimension k")
    ap.add_argument("--hidden", type=int, default=28, help="Hidden width for MLP layers")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=10000)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lam-mu",   dest="lam_mu",   type=float, default=1e-3)
    ap.add_argument("--lam-var",  dest="lam_var",  type=float, default=1e-3)
    ap.add_argument("--lam-cov",  dest="lam_cov",  type=float, default=1e-3)
    ap.add_argument("--pred-batch", type=int, default=256, help="Batch size for predict() when exporting latents")
    ap.add_argument("--no-layernorm", action="store_true", help="Disable LayerNorm in hidden layers")
    return ap.parse_args()

def main(coeff_file='data/05_streamfunction_coeffs.np', splits='data/05_splits.npz', latent=32, hidden=28, batch=128,
         epochs=10000, patience=50, lr=3e-4, lam_mu=1e-3, lam_var=1e-3,
         lam_cov=1e-3, pred_batch=256, no_layernorm=False):
    # Safer default: allow memory growth on GPUs
    gpus = tf.config.list_physical_devices('GPU')
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass
    tf.keras.mixed_precision.set_global_policy("float32")
    train(coeff_file, splits, latent, hidden, batch, epochs, patience, lr, lam_mu, lam_var, lam_cov,
          pred_batch, no_layernorm)

if __name__ == "__main__":
    args = get_args()
    main(args.coeff_file, args.splits, args.latent, args.hidden, args.batch, args.epochs, args.patience,
         args.lr, args.lam_mu, args.lam_var, args.lam_cov, args.pred_batch, args.no_layernorm)

