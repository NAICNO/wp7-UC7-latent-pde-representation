#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
finetune_encoder_to_latent.py  (Keras 3, true end-to-end fine-tuning)

Trainable chain:
    X_mod  --enc_1(.keras)-->  z1(32)  --enc_2(.keras)-->  z2(32)
Loss:
    REE(z2, lat_3)   # relative Euclidean error in latent space

Requirements:
- models/enc_1_<mod>_ld32.keras   (from do0504/do0505)
- models/enc_2_<mod>_ld32.keras   (from do0506)
- latents/lat_3_ld32.npy          (from do0506)
- data/05_<mod>.npy               (raw inputs)
- data/05_splits.npz              (train/val/test indices)

No adapter heads. No legacy optimizers. Pure Keras 3.
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from keras import Model, Input
from keras import layers
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# ----------------------------
# Custom layers (deserialization stubs)
#   Names & configs must match those used in do0504 / do0506.
# ----------------------------
import re

def infer_aligned_dim_from_enc2(mod: str) -> int:
    enc2_path = os.path.join("models", f"enc_2_{mod}_ld32.keras")
    # If the ld isn’t always 32, generalize by glob or regex:
    if not os.path.exists(enc2_path):
        # try to find any enc_2_{mod}_ldXX.keras
        pat = re.compile(rf"^enc_2_{re.escape(mod)}_ld(\d+)\.keras$")
        for fn in os.listdir("models"):
            m = pat.match(fn)
            if m:
                return int(m.group(1))
        raise FileNotFoundError(f"No enc_2_{mod}_ld*.keras found in models/")
    # default case (you said enc_2 carries the joint latent dim in the name)
    m = re.search(r"_ld(\d+)\.keras$", enc2_path)
    return int(m.group(1)) if m else 32

from keras.saving import register_keras_serializable

@register_keras_serializable(package="custom")
class PyramidToDepth(layers.Layer):
    def __init__(self, n_levels=9, **kwargs):
        super().__init__(**kwargs)
        self.n_levels = int(n_levels)
    def get_config(self):
        return {"n_levels": self.n_levels, "name": self.name}
    def call(self, x):
        # Not used at runtime here (only needed for deserialization graph),
        # but provide a functional body for safety in case of tracing.
        B = tf.shape(x)[0]; H = tf.shape(x)[1]; W = tf.shape(x)[2]
        levels = [x]
        for _ in range(1, self.n_levels):
            x = tf.nn.avg_pool2d(x, ksize=2, strides=2, padding='SAME')
            levels.append(x)
        ups = [tf.image.resize(lv, (H, W), method='bilinear') for lv in levels]
        stacked = tf.concat(ups, axis=-1)          # (B,H,W,9)
        stacked = tf.expand_dims(stacked, axis=1)  # (B,1,H,W,9)
        stacked = tf.transpose(stacked, [0,4,2,3,1])  # (B,9,H,W,1)
        return stacked

@register_keras_serializable(package="custom")
class SpaceGlobalAverage(layers.Layer):
    def call(self, x):
        return tf.reduce_mean(x, axis=[2,3])

@register_keras_serializable(package="custom")
class Standardize(layers.Layer):
    def __init__(self, mean_vec, std_vec, name=None):
        super().__init__(name=name, dtype=tf.float32)
        self.mean_vec = np.array(mean_vec, dtype=np.float32)
        self.std_vec  = np.array(std_vec,  dtype=np.float32)
        self.mu = tf.constant(self.mean_vec)
        self.sd = tf.constant(self.std_vec)
    def call(self, x): return (x - self.mu) / self.sd
    def get_config(self):
        return {"mean_vec": self.mean_vec.tolist(), "std_vec": self.std_vec.tolist(), "name": self.name}

@register_keras_serializable(package="custom")
class DeStandardize(layers.Layer):
    def __init__(self, mean_vec, std_vec, name=None):
        super().__init__(name=name, dtype=tf.float32)
        self.mean_vec = np.array(mean_vec, dtype=np.float32)
        self.std_vec  = np.array(std_vec,  dtype=np.float32)
        self.mu = tf.constant(self.mean_vec)
        self.sd = tf.constant(self.std_vec)
    def call(self, x): return x * self.sd + self.mu
    def get_config(self):
        return {"mean_vec": self.mean_vec.tolist(), "std_vec": self.std_vec.tolist(), "name": self.name}

@register_keras_serializable(package="custom")
class L2Normalize(layers.Layer):
    def __init__(self, axis=-1, eps=1e-8, name=None):
        super().__init__(name=name, dtype=tf.float32)
        self.axis = int(axis); self.eps = float(eps)
    def call(self, x):
        return x / (tf.norm(x, axis=self.axis, keepdims=True) + self.eps)
    def get_config(self): return {"axis": self.axis, "eps": self.eps, "name": self.name}

@register_keras_serializable(package="custom")
class Blend(layers.Layer):
    def build(self, input_shape):
        self.alpha_logit = self.add_weight(shape=(), initializer="zeros", trainable=True, name="alpha_logit")
    def call(self, inputs):
        a, b = inputs
        w = tf.sigmoid(self.alpha_logit)
        return w * a + (1.0 - w) * b

# ----------------------------
# Config
# ----------------------------
MOD_CHOICES = ["u16","u32","u64","u128","u256","coeff"]

# DATA_PATH = {
#     "u16":   "data/05_u16.npy",
#     "u32":   "data/05_u32.npy",
#     "u64":   "data/05_u64.npy",
#     "u128":  "data/05_u128.npy",
#     "u256":  "data/05_u256.npy",
#     "coeff": "data/05_streamfunction_coeffs.npy",
# }

INPUT_SHAPE = {
    "u16": (16,16,1), "u32": (32,32,1), "u64": (64,64,1),
    "u128": (128,128,1), "u256": (256,256,1), "coeff": (14,)  # coeff raw is 14-D (pre-standardization in do0505)
}

def ensure_dirs():
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

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

# ----------------------------
# Loss / metric: REE in latent space
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
# Patience printer callback (monitor val_REE)
# ----------------------------
import sys

class PatiencePrinter(tf.keras.callbacks.Callback):
    """
    Prints patience left for EarlyStopping.
    Auto-detects the validation REE key on first epoch.
    """
    def __init__(self, patience=30, monitor=None):
        super().__init__()
        self.patience = int(patience)
        self.user_monitor = monitor  # explicit key or None for auto
        self.best = np.inf
        self.wait = 0
        self.key = None

    def _detect_key(self, logs):
        # explicit key wins
        if self.user_monitor and self.user_monitor in logs:
            return self.user_monitor
        # common names
        for k in ("val_ree", "val_REE"):
            if k in logs:
                return k
        # any key containing both 'val' and 'ree'
        for k in logs.keys():
            kl = k.lower()
            if "val" in kl and "ree" in kl:
                return k
        return None

    def on_train_begin(self, logs=None):
        # announce on start (helps confirm the callback is active)
        print(f"[PatiencePrinter] armed with patience={self.patience}", flush=True)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if self.key is None:
            self.key = self._detect_key(logs)
            if self.key is None:
                # tell you we didn't find it yet
                print("[PatiencePrinter] waiting to detect a 'val_*ree*' metric...", flush=True)
                return
            else:
                print(f"[PatiencePrinter] monitoring: {self.key}", flush=True)

        cur = logs.get(self.key, None)
        if cur is None:
            return

        # update patience state
        if float(cur) < self.best - 1e-12:
            self.best = float(cur)
            self.wait = 0
            improved = " ✅ improved"
        else:
            self.wait += 1
            improved = ""

        left = max(0, self.patience - self.wait)
        print(
            f"Epoch {epoch+1:03d}: {self.key}={float(cur):.5f}  "
            f"→ patience left {left:2d}/{self.patience}{improved}",
            flush=True
        )

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mod", required=True, choices=MOD_CHOICES)
    ap.add_argument("--lat3", default="latents/lat_3_ld32.npy")
    ap.add_argument('--data_prefix', type=str, default="data/05_", help='Prefix of the .npy files')
    ap.add_argument("--epochs", type=int, default=10000)
    ap.add_argument("--batch",  type=int, default=128)
    ap.add_argument("--lr",     type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--splits", default="data/05_splits.npz")
    ap.add_argument("--freeze-enc1", action="store_true", help="Start with enc_1 frozen (optional).")
    ap.add_argument("--unfreeze-epoch", type=int, default=0, help="If >0 and enc_1 frozen, unfreeze from this epoch.")
    return ap.parse_args()

# ----------------------------
# Main
# ----------------------------
def main(mod, lat3="latents/lat_3_ld32.npy", data_prefix="data/05_", epochs=10000, batch=128, lr=1e-3,
         patience=50, splits="data/05_splits.npz", freeze_enc1=False, unfreeze_epoch=0):
    # practical GPU setup + float32 (no mixed precision surprises)
    for g in tf.config.list_physical_devices("GPU"):
        try: tf.config.experimental.set_memory_growth(g, True)
        except: pass
    tf.keras.mixed_precision.set_global_policy("float32")

    ensure_dirs()

    # --- data ---
    data_file = f"{data_prefix}{mod}.npy"
    if mod == "coeff":
        data_file = f"{data_prefix}streamfunction_coeffs.npy"
    X = np.load(data_file).astype("float32")
    if X.ndim == 3: X = X[..., None]  # make (N,H,W,1) for uXX
    Z_star = np.load(lat3).astype("float32")  # (N,32)
    if X.shape[0] != Z_star.shape[0]:
        raise ValueError(f"N mismatch: X={X.shape[0]} vs lat_3={Z_star.shape[0]}")
    idx_tr, idx_va, idx_te = load_splits(splits)

    # --- load trainable .keras encoders ---
    enc1_path = os.path.join("models", f"enc_1_{mod}_ld32.keras")
    enc2_path = os.path.join("models", f"enc_2_{mod}_ld32.keras")
    ad = infer_aligned_dim_from_enc2(mod)
    if not os.path.exists(enc1_path): raise FileNotFoundError(enc1_path)
    if not os.path.exists(enc2_path): raise FileNotFoundError(enc2_path)

    enc1 = tf.keras.models.load_model(enc1_path, compile=False)
    enc2 = tf.keras.models.load_model(enc2_path, compile=False)

    # optional freeze/unfreeze strategy for enc1
    enc1.trainable = not freeze_enc1

    # enc2 is always fine-tuned
    enc2.trainable = True

    # --- build chain ---
    inp = Input(shape=INPUT_SHAPE[mod], name=f"{mod}_in")
    z1  = enc1(inp)
    z2  = enc2(z1)
    chain = Model(inp, z2, name=f"chain_{mod}_enc1_enc2")
    chain.summary()

    # compile with REE on latent
    chain.compile(optimizer=Adam(learning_rate=lr), loss=REE, metrics=[REE])

    # datasets
    ds_tr = make_dataset(X, Z_star, idx_tr, batch, True)
    ds_va = make_dataset(X, Z_star, idx_va, batch, False)
    ds_te = make_dataset(X, Z_star, idx_te, batch, False)

    # callbacks
    ckpt_path = f"models/.tmp_enc_3_{mod}_ld{ad}.keras"
    monitor_key = "val_ree"
    cbs = [
        tf.keras.callbacks.EarlyStopping(monitor=monitor_key, patience=patience, restore_best_weights=True, mode="min"),
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor=monitor_key, save_best_only=True, mode="min"),
        tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor_key, factor=0.5, patience=max(5, patience//3), verbose=1, mode="min"),
        PatiencePrinter(patience=patience, monitor=monitor_key),
    ]

    # optional unfreeze of enc1 after some epochs
    if freeze_enc1 and unfreeze_epoch > 0:
        class UnfreezeEnc1(tf.keras.callbacks.Callback):
            def on_epoch_begin(self, epoch, logs=None):
                if epoch + 1 == unfreeze_epoch:
                    print(f"[Info] Unfreezing enc_1 at epoch {epoch+1}")
                    enc1.trainable = True
                    chain.compile(optimizer=Adam(learning_rate=lr), loss=REE, metrics=[REE])
        cbs.insert(0, UnfreezeEnc1())

    # fit
    hist = chain.fit(ds_tr, validation_data=ds_va, epochs=epochs, callbacks=cbs, verbose=2)

    # eval
    tr = chain.evaluate(ds_tr, verbose=0, return_dict=True)
    va = chain.evaluate(ds_va, verbose=0, return_dict=True)
    te = chain.evaluate(ds_te, verbose=0, return_dict=True)

    def pick_ree(d):
        # Prefer explicit metric keys; fall back to last non-loss metric
        for k in ("REE", "ree", "val_REE", "val_ree"):
            if k in d:
                return float(d[k])
        for k in reversed(list(d.keys())):
            if k not in ("loss", "learning_rate"):
                return float(d[k])
        return float(d.get("loss", float("nan")))

    print(f"\nFinal REE — {mod}: " f"train={pick_ree(tr):.5f}  val={pick_ree(va):.5f}  test={pick_ree(te):.5f}")


    # save final chain
    out_path = f"models/enc_3_{mod}_ld{ad}.keras"
    chain.save(out_path)
    print(f"Saved: {out_path}")

    export_dir = os.path.join("exports", f"enc_3_{mod}_ld{ad}")
    os.makedirs("exports", exist_ok=True)
    try:
        chain.export(export_dir)  # Keras 3 export
        print(f"Exported SavedModel → {export_dir}")
    except Exception as e:
        print(f"[WARN] Export of enc_3 failed: {e}")

if __name__ == "__main__":
    args = get_args()
    main(args.mod, args.lat3, args.data_prefix, args.epochs, args.batch, args.lr, args.patience, args.splits,
         args.freeze_enc1, args.unfreeze_epoch)

