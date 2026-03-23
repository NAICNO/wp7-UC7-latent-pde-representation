#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
finetune_decoder_from_latent.py  (Keras 3, true end-to-end fine-tuning for decoders)

Trainable chain:
    z_aligned(32)  --dec_2(.keras)-->  h  --dec_1(.keras)-->  y_mod  (image or coeff)

Loss / metric:
    REE(y_pred, y_true)    # relative Euclidean error in OUTPUT space

Inputs:
- models/dec_1_<mod>_ld32.keras
- models/dec_2_<mod>_ld<ad>.keras   (we infer <ad> from this filename)
- latents/lat_3_ld32.npy            # X
- data/05_<mod>.npy                 # Y
- data/05_splits.npz                # train/val/test

Outputs:
- models/dec_3_<mod>_ld<ad>.keras
- exports/dec_3_<mod>_ld<ad>/
- results/finetune_dec_chain_<mod>_ld<ad>.csv
"""

import os
import re
import sys
import argparse
import numpy as np
import tensorflow as tf
from keras import Model, Input, layers
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.saving import register_keras_serializable
import pandas as pd

# ----------------------------
# Custom layers (stubs for deserialization)
# ----------------------------
@register_keras_serializable(package="custom")
class Standardize(layers.Layer):
    def __init__(self, mean_vec, std_vec, name=None):
        super().__init__(name=name, dtype=tf.float32)
        self.mean_vec = np.array(mean_vec, dtype=np.float32)
        self.std_vec  = np.array(std_vec,  dtype=np.float32)
        self.mu = tf.constant(self.mean_vec)
        self.sd = tf.constant(self.std_vec)
    def call(self, x): return (x - self.mu) / self.sd
    def get_config(self): return {"mean_vec": self.mean_vec.tolist(), "std_vec": self.std_vec.tolist(), "name": self.name}

@register_keras_serializable(package="custom")
class DeStandardize(layers.Layer):
    def __init__(self, mean_vec, std_vec, name=None):
        super().__init__(name=name, dtype=tf.float32)
        self.mean_vec = np.array(mean_vec, dtype=np.float32)
        self.std_vec  = np.array(std_vec,  dtype=np.float32)
        self.mu = tf.constant(self.mean_vec)
        self.sd = tf.constant(self.std_vec)
    def call(self, x): return x * self.sd + self.mu
    def get_config(self): return {"mean_vec": self.mean_vec.tolist(), "std_vec": self.std_vec.tolist(), "name": self.name}

@register_keras_serializable(package="custom")
class L2Normalize(layers.Layer):
    def __init__(self, axis=-1, eps=1e-8, name=None):
        super().__init__(name=name, dtype=tf.float32)
        self.axis = int(axis); self.eps = float(eps)
    def call(self, x): return x / (tf.norm(x, axis=self.axis, keepdims=True) + self.eps)
    def get_config(self): return {"axis": self.axis, "eps": self.eps, "name": self.name}

@register_keras_serializable(package="custom")
class Blend(layers.Layer):
    def build(self, input_shape):
        self.alpha_logit = self.add_weight(shape=(), initializer="zeros", trainable=True, name="alpha_logit")
    def call(self, inputs):
        a, b = inputs
        w = tf.sigmoid(self.alpha_logit)
        return w * a + (1.0 - w) * b

@register_keras_serializable(package="custom")
class PyramidToDepth(layers.Layer):
    def __init__(self, n_levels=9, **kwargs):
        super().__init__(**kwargs)
        self.n_levels = int(n_levels)
    def get_config(self): return {"n_levels": self.n_levels, "name": self.name}
    def call(self, x):
        # Safe functional body for possible tracing; not used in decoder path.
        B = tf.shape(x)[0]; H = tf.shape(x)[1]; W = tf.shape(x)[2]
        lvls = [x]
        for _ in range(1, self.n_levels):
            x = tf.nn.avg_pool2d(x, 2, 2, 'SAME'); lvls.append(x)
        ups = [tf.image.resize(v, (H, W), method='bilinear') for v in lvls]
        st = tf.concat(ups, axis=-1); st = tf.expand_dims(st, 1); st = tf.transpose(st, [0,4,2,3,1])
        return st

# ----------------------------
# Config
# ----------------------------
MOD_CHOICES = ["u16","u32","u64","u128","u256","coeff"]

DATA_PATH = {
    "u16":   "data/05_u16.npy",
    "u32":   "data/05_u32.npy",
    "u64":   "data/05_u64.npy",
    "u128":  "data/05_u128.npy",
    "u256":  "data/05_u256.npy",
    "coeff": "data/05_streamfunction_coeffs.npy",
}

OUTPUT_SHAPE = {
    "u16": (16,16,1), "u32": (32,32,1), "u64": (64,64,1),
    "u128": (128,128,1), "u256": (256,256,1), "coeff": (14,)  # match your 05_* arrays
}

def ensure_dirs():
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("exports", exist_ok=True)

def load_splits(path="data/05_splits.npz"):
    d = np.load(path, allow_pickle=True)
    for a,b,c in [("train","val","test"), ("train_idx","val_idx","test_idx")]:
        if a in d and b in d and c in d:
            return d[a].astype(int), d[b].astype(int), d[c].astype(int)
    raise KeyError("train/val/test indices not found in splits npz")

def make_dataset(X, Y, idx, batch, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((X[idx], Y[idx]))
    if shuffle:
        ds = ds.shuffle(min(len(idx), 4*batch))
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)

def infer_ad_from_dec2(mod: str) -> int:
    # find models/dec_2_<mod>_ld<ad>.keras
    pat = re.compile(rf"^dec_2_{re.escape(mod)}_ld(\d+)\.keras$")
    for fn in os.listdir("models"):
        m = pat.match(fn)
        if m:
            return int(m.group(1))
    raise FileNotFoundError(f"No dec_2_{mod}_ld*.keras found in models/")

# ----------------------------
# REE in output space
# ----------------------------
@tf.keras.utils.register_keras_serializable(package="custom")
def REE(y_true, y_pred):
    eps = 1e-8
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    diff = tf.reshape(y_pred - y_true, [tf.shape(y_true)[0], -1])
    num  = tf.sqrt(tf.reduce_sum(tf.square(diff), axis=1))
    den  = tf.sqrt(tf.reduce_sum(tf.square(tf.reshape(y_true,[tf.shape(y_true)[0],-1])),axis=1))
    return tf.reduce_mean(num / (den + eps))

# ----------------------------
# Patience printer (auto-detect val metric name)
# ----------------------------
class PatiencePrinter(tf.keras.callbacks.Callback):
    def __init__(self, patience=30, monitor=None):
        super().__init__()
        self.patience = int(patience)
        self.user_monitor = monitor
        self.best = np.inf
        self.wait = 0
        self.key = None
    def _detect_key(self, logs):
        if self.user_monitor and self.user_monitor in logs:
            return self.user_monitor
        for k in ("val_ree", "val_REE"):
            if k in logs: return k
        for k in logs.keys():
            kl = k.lower()
            if "val" in kl and "ree" in kl: return k
        return None
    def on_train_begin(self, logs=None):
        print(f"[PatiencePrinter] armed with patience={self.patience}", flush=True)
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if self.key is None:
            self.key = self._detect_key(logs)
            if self.key is None:
                print("[PatiencePrinter] waiting to detect a 'val_*ree*' metric...", flush=True)
                return
            else:
                print(f"[PatiencePrinter] monitoring: {self.key}", flush=True)
        cur = logs.get(self.key, None)
        if cur is None: return
        if float(cur) < self.best - 1e-12:
            self.best = float(cur); self.wait = 0; improved = " ✅ improved"
        else:
            self.wait += 1; improved = ""
        left = max(0, self.patience - self.wait)
        print(f"Epoch {epoch+1:03d}: {self.key}={float(cur):.5f}  → patience left {left:2d}/{self.patience}{improved}", flush=True)

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mod", required=True, choices=MOD_CHOICES)
    ap.add_argument("--lat3", default="latents/lat_3_ld32.npy",
                    help="Aligned latent inputs (N,32)")
    ap.add_argument("--epochs", type=int, default=10000)
    ap.add_argument("--batch",  type=int, default=128)
    ap.add_argument("--lr",     type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--splits", default="data/05_splits.npz")
    ap.add_argument("--suffix", default="")
    return ap.parse_args()

# ----------------------------
# Main
# ----------------------------
def main(mod, lat3="latents/lat_3_ld32.npy", epochs=10000, batch=128, lr=1e-3, patience=50,
         splits="data/05_splits.npz", suffix=""):
    for g in tf.config.list_physical_devices("GPU"):
        try: tf.config.experimental.set_memory_growth(g, True)
        except: pass
    tf.keras.mixed_precision.set_global_policy("float32")

    ensure_dirs()

    # Data
    Z = np.load(lat3).astype("float32")           # (N,32)
    Y = np.load(DATA_PATH[mod]).astype("float32") # (N, H,W,1) or (N,14)
    if Y.ndim == 3: Y = Y[..., None]                   # ensure channel dim for images
    if Z.shape[0] != Y.shape[0]:
        raise ValueError(f"N mismatch: Z={Z.shape[0]} vs Y={Y.shape[0]}")
    idx_tr, idx_va, idx_te = load_splits(splits)

    # Models
    ad = infer_ad_from_dec2(mod)
    dec1_path = os.path.join("models", f"dec_1_{mod}_ld32.keras")
    dec2_path = os.path.join("models", f"dec_2_{mod}_ld{ad}.keras")
    if not os.path.exists(dec1_path): raise FileNotFoundError(dec1_path)
    if not os.path.exists(dec2_path): raise FileNotFoundError(dec2_path)

    dec1 = tf.keras.models.load_model(dec1_path, compile=False)
    dec2 = tf.keras.models.load_model(dec2_path, compile=False)
    dec1.trainable = True
    dec2.trainable = True

    # Chain: z(32) -> dec2 -> dec1 -> y_mod
    inp = Input(shape=(32,), name="z_aligned_in")
    h   = dec2(inp)
    y   = dec1(h)
    chain = Model(inp, y, name=f"dec_chain_{mod}")
    chain.summary()

    # Compile (REE on outputs)
    chain.compile(optimizer=Adam(learning_rate=lr), loss=REE, metrics=[REE])

    # Datasets
    ds_tr = make_dataset(Z, Y, idx_tr, batch, True)
    ds_va = make_dataset(Z, Y, idx_va, batch, False)
    ds_te = make_dataset(Z, Y, idx_te, batch, False)

    # Callbacks
    monitor_key = "val_ree"
    ckpt_path = f"models/.tmp_dec_3_{mod}_ld{ad}{suffix}.keras"
    cbs = [
        EarlyStopping(monitor=monitor_key, patience=patience, restore_best_weights=True, mode="min"),
        ModelCheckpoint(ckpt_path, monitor=monitor_key, save_best_only=True, mode="min"),
        ReduceLROnPlateau(monitor=monitor_key, factor=0.5,
                          patience=max(5, patience//3), verbose=1, mode="min"),
        PatiencePrinter(patience=patience, monitor=monitor_key),
    ]

    # Fit
    hist = chain.fit(ds_tr, validation_data=ds_va, epochs=epochs, callbacks=cbs, verbose=2)

    # Evaluate
    tr = chain.evaluate(ds_tr, verbose=0, return_dict=True)
    va = chain.evaluate(ds_va, verbose=0, return_dict=True)
    te = chain.evaluate(ds_te, verbose=0, return_dict=True)

    def pick_ree(d):
        for k in ("REE","ree","val_REE","val_ree"):
            if k in d: return float(d[k])
        for k in reversed(list(d.keys())):
            if k not in ("loss","learning_rate"):
                return float(d[k])
        return float(d.get("loss", float("nan")))

    print(f"\nFinal REE — {mod}: train={pick_ree(tr):.5f}  val={pick_ree(va):.5f}  test={pick_ree(te):.5f}")

    # Save artifacts
    out_path = f"models/dec_3_{mod}_ld{ad}{suffix}.keras"
    chain.save(out_path)
    print(f"Saved: {out_path}")

    export_dir = os.path.join("exports", f"dec_3_{mod}_ld{ad}{suffix}")
    try:
        chain.export(export_dir)
        print(f"Exported SavedModel → {export_dir}")
    except Exception as e:
        print(f"[WARN] Export failed: {e}")

    # History
    pd.DataFrame(hist.history).to_csv(f"results/finetune_dec_chain_{mod}_ld{ad}{suffix}.csv", index=False)

if __name__ == "__main__":
    args = get_args()
    main(args.mod, args.lat3, args.epochs, args.batch, args.lr, args.patience, args.splits, args.suffix)

