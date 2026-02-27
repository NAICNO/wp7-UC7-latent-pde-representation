#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
do0410-u-autoencoder.py
Staged Conv3D pyramid encoder + stand-alone decoder for u16/u32/u64/u128/u256.
- 9-level pyramid (depth D=9), Conv3D blocks with spatial-only pooling.
- Each stage: H,W down by 2; channels x4 (capped).
- Spatial average -> (D,C) -> Dense -> latent(=--latent).
- REE loss. Saves encoder/decoder separately. Saves one latents file for ALL samples.
"""

import os, json, argparse
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from numpy.lib.format import open_memmap

gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

tf.keras.mixed_precision.set_global_policy("float32")
print("Mixed precision disabled (safe default).")

# ----------------------------
# Utils
# ----------------------------
def side_from_modality(name: str) -> int:
    # expects "u256", "u128", ...
    s = ''.join([ch for ch in name if ch.isdigit()])
    return int(s) if s else 256

def parse_to_level(s: str) -> int:
    # "1x1" -> 1, "4x4" -> 4
    s = s.lower().strip()
    if "x" in s:
        a,b = s.split("x")
        a = int(a); b = int(b)
        if a != b:
            raise ValueError("--to-level must be NxN")
        return a
    v = int(s)
    return v

def ensure_dirs():
    os.makedirs("models", exist_ok=True)
    os.makedirs("latents", exist_ok=True)

# ----------------------------
# Loss / Metrics
# ----------------------------
def ree_loss(y_true, y_pred, eps=1e-8):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    num = tf.reduce_sum(tf.square(y_pred - y_true), axis=[1,2,3])
    den = tf.reduce_sum(tf.square(y_true),        axis=[1,2,3]) + eps
    return tf.reduce_mean(num / den)

# ----------------------------
# Custom Layers (serializable; avoid Lambda lambdas)
# ----------------------------
@keras.utils.register_keras_serializable(package="custom")
class PyramidToDepth(layers.Layer):
    """
    Build 9-level average-pooling pyramid, upsample each level to original H,W,
    and stack as depth: output shape (B, D=9, H, W, 1).
    """
    def __init__(self, n_levels=9, **kwargs):
        super().__init__(**kwargs)
        self.n_levels = n_levels

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"n_levels": self.n_levels})
        return cfg

    def call(self, x):
        # x: (B,H,W,1)
        B = tf.shape(x)[0]
        H = tf.shape(x)[1]
        W = tf.shape(x)[2]

        levels = []
        cur = x
        levels.append(cur)
        for _ in range(1, self.n_levels):
            # avg pool 2x2, stride 2
            cur = tf.nn.avg_pool2d(cur, ksize=2, strides=2, padding='SAME')
            levels.append(cur)

        # upsample each to (H,W) bilinear
        ups = []
        for lv in levels:
            ups.append(tf.image.resize(lv, size=(H, W), method='bilinear'))
        # stack along a new depth axis: (B,H,W,1) x 9 -> (B,9,H,W,1)
        # First concat on channel then reshape+transpose to keep graphs simple:
        stacked = tf.concat(ups, axis=-1)       # (B,H,W,9)
        stacked = tf.expand_dims(stacked, axis=1)  # (B,1,H,W,9)
        stacked = tf.transpose(stacked, perm=[0,4,2,3,1])  # (B,9,H,W,1)
        return stacked

@keras.utils.register_keras_serializable(package="custom")
class SpaceGlobalAverage(layers.Layer):
    """Average over spatial H,W only for a 5D tensor (B,D,H,W,C) -> (B,D,C)."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_config(self):
        return super().get_config()

    def call(self, x):
        # x: (B,D,H,W,C)
        return tf.reduce_mean(x, axis=[2,3])  # -> (B,D,C)

# ----------------------------
# Blocks
# ----------------------------
def conv3d_block(x, out_ch, k=(3,3,3), act="gelu", name=None):
    y = layers.Conv3D(out_ch, k, padding="same", activation=None, name=None if name is None else name+"_conv")(x)
    y = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=None if name is None else name+"_bn")(y)
    y = layers.Activation(act, name=None if name is None else name+"_act")(y)
    return y

def resid3d(x, out_ch, k=(3,3,3), depth_mixer=False, name=None):
    y = conv3d_block(x, out_ch, k, name=None if name is None else name+"_main")
    if depth_mixer:
        y = conv3d_block(y, out_ch, k=(3,1,1), name=None if name is None else name+"_dmix")
    skip = layers.Conv3D(out_ch, (1,1,1), padding="same", activation=None,
                         name=None if name is None else name+"_skip")(x)
    y = layers.Add(name=None if name is None else name+"_add")([y, skip])
    y = layers.Activation("gelu", name=None if name is None else name+"_out")(y)
    return y

def stage3d(x, out_ch, pool=True, depth_mixer=False, name=None):
    x = resid3d(x, out_ch, depth_mixer=depth_mixer, name=None if name is None else name+"_resid")
    if pool:
        x = layers.AveragePooling3D(pool_size=(1,2,2), name=None if name is None else name+"_pool")(x)
    return x

# ----------------------------
# Models
# ----------------------------
def build_encoder(input_side, latent_dim, to_level, c0=2, cmax=128, depth_mixer=False, ch_mult=4):
    """
    Encoder:
    - Input: (H,W,1)
    - Pyramid -> (B, D=9, H, W, 1)
    - Stages: each downsample halves H,W and multiplies channels by ch_mult (capped at cmax)
    - Stop when H=W=to_level
    - Spatial average -> (B,D,C) -> Flatten -> Dense(latent_dim)
    """
    if ch_mult < 1:
        raise ValueError(f"ch_mult must be >= 1, got {ch_mult}")

    inp = keras.Input(shape=(input_side, input_side, 1), name="input")

    x = PyramidToDepth(n_levels=9, name="pyr_to_depth")(inp)  # (B,9,H,W,1)

    # Initial stem
    ch = min(int(c0), int(cmax))
    x = resid3d(x, ch, depth_mixer=depth_mixer, name="stem")

    # Stage loop
    cur_side = int(input_side)
    stage_id = 0
    while cur_side > to_level:
        stage_id += 1
        next_ch = min(ch * ch_mult, cmax)
        x = stage3d(x, next_ch, pool=True, depth_mixer=depth_mixer, name=f"s{stage_id}")
        ch = next_ch
        cur_side //= 2
        if cur_side < to_level:
            raise ValueError(f"to-level {to_level} must divide input side {input_side} by powers of 2.")

    # Now (B, D=9, to_level, to_level, ch)
    x_hw = SpaceGlobalAverage(name="space_avg")(x)   # (B, D, ch)
    x_flat = layers.Flatten(name="flatten_DxC")(x_hw)  # (B, D*C)
    z = layers.Dense(latent_dim, activation=None, name="latent")(x_flat)

    enc = keras.Model(inp, z, name=f"encoder_{input_side}_ulatent{latent_dim}")
    return enc, ch, to_level

def build_decoder(output_side, latent_dim, start_side, start_ch):
    """
    Decoder (stand-alone):
    - Input: (latent_dim,)
    - Dense -> (start_side, start_side, start_ch)
    - Upsample (×2) + Conv2D blocks until original side
    - Output: (H,W,1) in original scale
    """
    z_in = keras.Input(shape=(latent_dim,), name="z")
    x = layers.Dense(start_side * start_side * start_ch, activation="gelu")(z_in)
    x = layers.Reshape((start_side, start_side, start_ch))(x)

    ch = start_ch
    cur = start_side
    blk = 0
    while cur < output_side:
        blk += 1
        # upsample by 2
        x = layers.UpSampling2D(size=(2,2), interpolation="bilinear", name=f"up{blk}")(x)
        # two convs
        x = layers.Conv2D(ch, 3, padding="same", activation="gelu", name=f"dec{blk}_c1")(x)
        x = layers.Conv2D(ch, 3, padding="same", activation="gelu", name=f"dec{blk}_c2")(x)
        # keep channels reasonable
        ch = max(ch // 2, 32)  # gradually reduce; keeps params in check
        cur *= 2

    out = layers.Conv2D(1, 1, padding="same", activation=None, dtype="float32", name="out")(x)
    dec = keras.Model(z_in, out, name=f"decoder_{output_side}_ld{latent_dim}")
    return dec

def build_autoencoder(modality, latent_dim, to_level_str, c0=2, cmax=128, depth_mixer=False, ch_mult=4):
    side = side_from_modality(modality)
    to_level = parse_to_level(to_level_str)
    if (side % to_level) != 0:
        raise ValueError(f"--to-level {to_level_str} must divide input side {side} by powers of 2.")

    # Encoder uses ch_mult to control per-stage channel growth
    enc, last_ch, start_side = build_encoder(
        input_side=side,
        latent_dim=latent_dim,
        to_level=to_level,
        c0=c0,
        cmax=cmax,
        depth_mixer=depth_mixer,
        ch_mult=ch_mult
    )

    # Decoder stays as-is (it sizes itself from start_side/last_ch)
    dec = build_decoder(output_side=side, latent_dim=latent_dim,
                        start_side=start_side, start_ch=last_ch)

    # Full AE for training
    inp = keras.Input(shape=(side, side, 1), name="input_full")
    z = enc(inp)
    out = dec(z)
    ae = keras.Model(inp, out, name=f"ae_ms3d_{side}_ld{latent_dim}")
    return ae, enc, dec, side

# ----------------------------
# Data IO
# ----------------------------
def load_splits(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Splits file not found: {path}")
    # try JSON first
    if path.endswith(".json"):
        with open(path, "r") as f:
            s = json.load(f)
        return np.array(s["train"], dtype=int), np.array(s["val"], dtype=int), np.array(s["test"], dtype=int)
    # else npz
    d = np.load(path, allow_pickle=True)
    return d["train"].astype(int), d["val"].astype(int), d["test"].astype(int)

def load_data(data_file):
    X = np.load(data_file)
    if X.ndim == 3:
        X = X[..., None]
    X = X.astype("float32", copy=False)
    return X

# ----------------------------
# Callbacks
# ----------------------------
# -------------------- callback: show patience left --------------------
class OneLinePrinter(tf.keras.callbacks.Callback):
    def __init__(self, es):
        super().__init__()
        self.es = es

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        patience_left = max(self.es.patience - getattr(self.es, "wait", 0), 0)
        print(f"{epoch+1:04d} - ree_loss: {logs.get('loss'):.5f} - " f"val_ree_loss: {logs.get('val_loss'):.5f} - patience_left: {patience_left}")

def reconstruct_batched(encoder, decoder, x, batch_size=64):
    zs, xhats = [], []
    for i in range(0, x.shape[0], batch_size):
        z_b = encoder.predict(x[i:i+batch_size], verbose=0)
        zs.append(z_b)
    zs = np.concatenate(zs, axis=0)
    for i in range(0, zs.shape[0], batch_size):
        xhat_b = decoder.predict(zs[i:i+batch_size], verbose=0)
        xhats.append(xhat_b)
    return np.concatenate(xhats, axis=0)

def ree_per_sample(xt: np.ndarray, xp: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    yt = xt.reshape(xt.shape[0], -1)
    yp = xp.reshape(xp.shape[0], -1)
    num = np.linalg.norm(yp - yt, axis=1)**2
    den = np.linalg.norm(yt, axis=1)**2 + eps
    return num / den

def save_latents_streaming(encoder, X, out_path, batch_size=64, device="/GPU:0"):
    """
    Export latents for ALL samples to a proper .npy (with header), streamed to disk.
    """
    N = X.shape[0]
    latent_dim = int(encoder.output_shape[-1])

    # proper .npy with header, still memory-mapped for streaming writes
    z_mm = open_memmap(out_path, mode="w+", dtype="float32", shape=(N, latent_dim))
    idx = 0

    ds = tf.data.Dataset.from_tensor_slices(X).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    with tf.device(device):
        for xb in ds:
            zb = encoder(xb, training=False).numpy().astype("float32", copy=False)
            n = zb.shape[0]
            z_mm[idx:idx+n] = zb
            idx += n

    z_mm.flush()
    print(f"Saved latents for ALL samples (streaming): {out_path}")
    print("Latents shape:", (N, latent_dim), "(index-aligned with your data and splits)")


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", required=True, choices=["u16","u32","u64","u128","u256"])
    ap.add_argument("--data_file", required=True, help="Path to .npy with data (N,H,W) or (N,H,W,1)")
    ap.add_argument("--splits", default="data/05_splits.npz", help="NPZ or JSON with arrays train/val/test")
    ap.add_argument("--latent", type=int, default=32)
    ap.add_argument("--to-level", default="1x1", help="One of 1x1,2x2,4x4,8x8,16x16")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=10000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--c0", type=int, default=2, help="Start channels at the largest spatial stage")
    ap.add_argument("--cmax", type=int, default=128, help="Max channels cap")
    ap.add_argument("--patience", type=int, default=50, help="Patience")
    ap.add_argument("--depth-mixer", action="store_true", help="Add (3,1,1) Conv3D in residual blocks")
    ap.add_argument("--no-mixed-precision", action="store_true", help="Disable mixed precision")
    ap.add_argument("--pred-batch", type=int, default=8, help="Batch size for encoder.predict when exporting latents.")
    ap.add_argument("--predict-device", choices=["gpu","cpu"], default="gpu", help="Device to use for latent export.")
    ap.add_argument("--ch-mult", type=int, default=4, help="Channel multiplier per downsample stage in the encoder (default: 4)")
    return ap.parse_args()

# ----------------------------
# Training
# ----------------------------
def main(modality, data_file, splits="data/05_splits.npz", latent=32, to_level="1x1", batch=16,
         epochs=10000, lr=3e-4, c0=2, cmax=128, patience=50, depth_mixer=False, no_mixed_precision=False,
         pred_batch=8, predict_device="gpu", ch_mult=4):


    #if not args.no_mixed_precision:
        #try:
            #tf.keras.mixed_precision.set_global_policy("mixed_float16")
            #print("Mixed precision policy:", tf.keras.mixed_precision.global_policy())
        #except Exception as e:
            #print("Mixed precision not set:", e)

    ensure_dirs()

    X = load_data(data_file)
    N, H, W, C = X.shape
    side = side_from_modality(modality)
    assert H == side and W == side and C == 1, f"Data shape must be (N,{side},{side},1); got {X.shape}"

    idx_tr, idx_va, idx_te = load_splits(splits)

    x_tr = X[idx_tr]
    x_va = X[idx_va]

    ae, enc, dec, side = build_autoencoder(
        modality=modality,
        latent_dim=latent,
        to_level_str=to_level,
        c0=c0,
        cmax=cmax,
        depth_mixer=depth_mixer,
        ch_mult=ch_mult
    )

    ae.summary()

    # Optimizer
    opt = keras.optimizers.Adam(learning_rate=lr)
    ae.compile(optimizer=opt, loss=ree_loss)

    # Callbacks
    rlr = keras.callbacks.ReduceLROnPlateau( monitor="val_loss", factor=0.5, patience=3, verbose=1, min_lr=1e-7)
    es = keras.callbacks.EarlyStopping( monitor="val_loss", patience=patience, min_delta=1e-5, restore_best_weights=True, verbose=1)
    ol = OneLinePrinter(es)

    # Fit
    ae.fit(
        x_tr, x_tr,
        validation_data=(x_va, x_va),
        epochs=epochs,
        batch_size=batch,
        callbacks=[rlr, es, ol],
        shuffle=True,
        verbose=0
    )

    # ---------- Extended metrics (per-split REE) ----------
    x_te = X[idx_te]

    def summarize(split_name: str, x_split: np.ndarray):
        if x_split.shape[0] == 0:
            return {"split": split_name, "n": 0, "mean": np.nan, "std": np.nan,
                    "p0": np.nan, "p25": np.nan, "p50": np.nan, "p75": np.nan, "p100": np.nan}
        xhat = reconstruct_batched(enc, dec, x_split, batch_size=64)
        ree = ree_per_sample(x_split, xhat)
        q = np.percentile(ree, [0, 25, 50, 75, 100])
        return {"split": split_name, "n": int(ree.size),
                "mean": float(np.mean(ree)),
                "std": float(np.std(ree, ddof=1)) if ree.size > 1 else 0.0,
                "p0": float(q[0]), "p25": float(q[1]), "p50": float(q[2]),
                "p75": float(q[3]), "p100": float(q[4])}

    summ_train = summarize("train", x_tr)
    summ_val   = summarize("validate", x_va)
    summ_test  = summarize("test", x_te)

    rows = []
    for s in (summ_train, summ_val, summ_test):
        rows.append({
            "k": int(latent), "H": int(H), "W": int(W),
            "split": s["split"], "n": s["n"], "mean": s["mean"], "std": s["std"],
            "p0": s["p0"], "p25": s["p25"], "p50": s["p50"], "p75": s["p75"], "p100": s["p100"]
        })
    df = pd.DataFrame(rows)

    #metrics_csv = os.path.join("results", "do0413_metrics.csv")
    #os.makedirs(os.path.dirname(metrics_csv), exist_ok=True)
    #write_header = not os.path.exists(metrics_csv)
    #df.to_csv(metrics_csv, index=False, mode='a' if not write_header else 'w', header=write_header)
    #print("\nSaved AE metrics to:", metrics_csv)
    #print(df.to_string(index=False))

    # Save final encoder/decoder (these are the BEST due to restore_best_weights=True)
    enc_final = f"models/enc_1_u{side}_ld{latent}.keras"
    dec_final = f"models/dec_1_u{side}_ld{latent}.keras"
    enc.save(enc_final)
    dec.save(dec_final)

    # NEW: export self-contained inference SavedModels (no custom code needed to load)
    export_dir = f"exports"
    os.makedirs(export_dir, exist_ok=True)

    enc_saved = os.path.join(export_dir, f"enc_1_u{side}_ld{latent}")
    dec_saved = os.path.join(export_dir, f"dec_1_u{side}_ld{latent}")

    # Ensure we export inference functions (no optimizer/compile state)
    # Keras v3: OK to call tf.saved_model.save on a Keras Model for inference.
    enc.export(enc_saved)  
    dec.export(dec_saved)


    print(f"Saved KERAS:   {enc_final}  |  {dec_final}")
    print(f"Saved TF SM:   {enc_saved}  |  {dec_saved}")

    # --- Export latents with the same BEST encoder ---
    lat_path = f"latents/lat_1_u{side}_ld{latent}.npy"
    save_latents_streaming(enc, X, lat_path, batch_size=pred_batch, device="/GPU:0" if predict_device == "gpu" else "/CPU:0")

    # now it's safe to clear if you want
    keras.backend.clear_session()

if __name__ == "__main__":
    args = get_args()
    main(args.modality, args.data_file, args.splits, args.latent, args.to_level, args.batch, args.epochs, args.lr,
         args.c0, args.cmax, args.patience, args.depth_mixer, args.no_mixed_precision, args.pred_batch,
         args.predict_device, args.ch_mult)

