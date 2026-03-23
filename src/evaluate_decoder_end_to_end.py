#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
do0512_u_decoder_end_to_end.py  (clean, single-metric version)

Train a NEW supervised decoder dec_4_<mod> to map aligned latent z (lat_3_ld<ad>.npy)
→ modality target (data/05_*.npy), WITHOUT using dec_1/dec_2.

# Single metric everywhere (train/val/final):
# RelMSE = sum(||y - y_hat||^2) / sum(||y||^2)  (implemented as constant * MSE)

Outputs:
  models/dec_4_<mod>_ld<ad>.keras
  exports/dec_4_<mod>_ld<ad>/    (SavedModel for inference)

Usage:
  python do0512_u_decoder_end_to_end.py --mod u256 --ad 32
"""

import os, re, argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback

# -----------------------
# Paths / modalities
# -----------------------
DATA_PATH = {
    "u16":   "data/05_u16.npy",
    "u32":   "data/05_u32.npy",
    "u64":   "data/05_u64.npy",
    "u128":  "data/05_u128.npy",
    "u256":  "data/05_u256.npy",
    "coeff": "data/05_streamfunction_coeffs.npy",
}
SPLITS_NPZ = "data/05_splits.npz"
LATENTS_TMPL = "latents/lat_3_ld{ad}.npy"

# -----------------------
# Utils
# -----------------------
def ensure_dirs():
    os.makedirs("models", exist_ok=True)
    os.makedirs("exports", exist_ok=True)

def side_from_mod(mod):
    if mod == "coeff":
        return None
    return int(re.sub(r"\D", "", mod))

def load_splits(path=SPLITS_NPZ):
    d = np.load(path, allow_pickle=True)
    # allow either train/val/test or train_idx/val_idx/test_idx
    if all(k in d for k in ("train", "val", "test")):
        return d["train"].astype(int), d["val"].astype(int), d["test"].astype(int)
    if all(k in d for k in ("train_idx", "val_idx", "test_idx")):
        return d["train_idx"].astype(int), d["val_idx"].astype(int), d["test_idx"].astype(int)
    raise KeyError("train/val/test (or *_idx) not found in splits npz")

def as_4d(x):
    """(N,H,W) → (N,H,W,1)"""
    if x.ndim == 3:
        x = x[..., None]
    return x.astype("float32", copy=False)

# ----- Single metric (and loss): RelMSE% -----
def relmse_loss(y_true, y_pred, eps=1e-12):
    """
    RelMSE% = 100 * sum(||y_pred - y_true||^2) / sum(||y_true||^2), computed per batch.
    We use this BOTH as the training loss and the validation metric.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    num = tf.reduce_sum(tf.square(y_pred - y_true))         # batch sum ||err||^2
    den = tf.reduce_sum(tf.square(y_true)) + eps            # batch sum ||y||^2
    return num / den

def mean_per_pixel_energy(dataset):
    num = 0.0
    den = 0
    for _, y in dataset:
        y = tf.cast(y, tf.float32)
        num += float(tf.reduce_sum(tf.square(y)).numpy())
        den += int(tf.size(y).numpy())
    return num / max(den, 1)

# --- replace your ScaledMSE with this ---
@tf.keras.utils.register_keras_serializable(package="custom")
class ScaledMSE(tf.keras.losses.Loss):
    """
    Relative MSE (unitless):  sum ||y - ŷ||^2 / sum ||y||^2  (implemented as constant * MSE)
    The constant is computed once from the train split:  scale = 1 / E_train[ y^2 ].
    """
    def __init__(self, scale=None, dataset=None, name="relmse"):
        super().__init__(name=name)
        if (scale is None) == (dataset is None):
            raise ValueError("Provide exactly one of `scale` or `dataset`.")
        if dataset is not None:
            # average per-element energy over the TRAIN split
            scale = 1.0 / (mean_per_pixel_energy(dataset) + 1e-12)
        # store as python float (for serialization); create tf constant for runtime
        self._scale_value = float(scale)
        self.scale = tf.constant(self._scale_value, dtype=tf.float32)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        mse = tf.reduce_mean(tf.square(y_pred - y_true))
        return self.scale * mse  # identical gradients/minima to MSE

    # exact-split evaluation using the SAME constant scaling
    def eval(self, model, ds):
        num = 0.0
        den = 0
        for xb, yb in ds:
            yp = model(xb, training=False)
            num += float(tf.reduce_sum(tf.square(tf.cast(yp, tf.float32) - tf.cast(yb, tf.float32))))
            den += int(tf.size(yb).numpy())
        mse = num / max(den, 1)
        return float(self._scale_value * mse)   # <- return python float

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"scale": self._scale_value})
        return cfg

# -----------------------
# Pretty printing with patience
# -----------------------
class PatiencePrinter(Callback):
    def __init__(self, es: EarlyStopping):
        super().__init__()
        self.es = es

    def _get_lr(self):
        opt = self.model.optimizer
        lr = opt.learning_rate
        try:
            if callable(lr):
                step = tf.cast(opt.iterations, tf.float32)
                return float(tf.keras.backend.get_value(lr(step)))
            return float(tf.keras.backend.get_value(lr))
        except Exception:
            return float(tf.keras.backend.get_value(tf.convert_to_tensor(lr)))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        left = max(self.es.patience - getattr(self.es, "wait", 0), 0)
        tr_pct = logs.get("loss", float("nan"))      # we use loss==RelMSE%
        va_pct = logs.get("val_loss", float("nan"))  # val_loss==RelMSE%
        # print(
        #     f"Epoch {epoch+1} | "
        #     f"relmse={tr_pct:.5f} | val_relmse={va_pct:.5f} | "
        #     f"lr={self._get_lr():.2e} | patience_left={left}"
        # )

# -----------------------
# Decoder (z → modality)
# -----------------------
def conv2_block(x, ch, name):
    x = layers.Conv2D(ch, 3, padding="same", activation="gelu", name=name+"_c1")(x)
    x = layers.Conv2D(ch, 3, padding="same", activation="gelu", name=name+"_c2")(x)
    return x

def build_dec4_for_image(output_side: int, ad: int, start_side: int = 8, base_ch: int = 192):
    """
    Progressive upsampling decoder:
      z (ad,) → Dense → (start_side,start_side,base_ch)
      repeat: Up2x + Conv2D blocks until output_side
      final 1x1 conv to single channel.
    """
    z = layers.Input(shape=(ad,), name="z")
    x = layers.Dense(start_side * start_side * base_ch, activation="gelu", name="dense_proj")(z)
    x = layers.Reshape((start_side, start_side, base_ch), name="resh")(x)

    ch = base_ch
    cur = start_side
    blk = 0
    while cur < output_side:
        blk += 1
        x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name=f"up{blk}")(x)
        x = conv2_block(x, ch, name=f"blk{blk}")
        ch = max(int(ch * 0.75), 48)  # gentle channel decay
        cur *= 2

    out = layers.Conv2D(1, 1, padding="same", activation=None, dtype="float32", name="out")(x)
    return Model(z, out, name=f"dec4_img_{output_side}_ld{ad}")

def build_dec4_for_coeff(coeff_dim: int, ad: int, hidden: int = 256, depth: int = 2):
    """
    Simple MLP for coefficients: z → MLP → y (coeff_dim,)
    """
    z = layers.Input(shape=(ad,), name="z")
    x = z
    for i in range(depth):
        x = layers.Dense(hidden, activation="gelu", name=f"h{i+1}")(x)
    y = layers.Dense(coeff_dim, activation=None, dtype="float32", name="out")(x)
    return Model(z, y, name=f"dec4_coeff_{coeff_dim}_ld{ad}")

# -----------------------
# Dataset
# -----------------------
def make_ds(z, y, idx, batch, shuffle=True):
    z_s = z[idx].astype("float32", copy=False)
    y_s = y[idx].astype("float32", copy=False)
    ds = tf.data.Dataset.from_tensor_slices((z_s, y_s))
    if shuffle:
        ds = ds.shuffle(min(len(idx), 10000), reshuffle_each_iteration=True)
    ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
    return ds

# -----------------------
# Exact split evaluation (same metric)
# -----------------------
def eval_relmse_split(model, ds):
    num = 0.0
    den = 0.0
    for xb, yb in ds:
        yp = model(xb, training=False)
        err = tf.square(tf.cast(yp, tf.float32) - tf.cast(yb, tf.float32))
        num += float(tf.reduce_sum(err).numpy())
        den += float(tf.reduce_sum(tf.square(tf.cast(yb, tf.float32))).numpy())
    return num / max(den, 1e-12)

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mod", required=True, choices=list(DATA_PATH.keys()))
    ap.add_argument("--ad", type=int, default=32, help="aligned latent dim (matches lat_3)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=10000)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--start-side", type=int, default=8, help="initial spatial size for image decoders")
    ap.add_argument("--base-ch", type=int, default=192, help="initial channel width for image decoders")
    ap.add_argument("--coeff-hidden", type=int, default=256)
    ap.add_argument("--coeff-depth", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--stop", action="store_true", help="Stop execution after model summary.")
    return ap.parse_args()

# -----------------------
# Main
# -----------------------
def main(mod, ad=32, batch=16, epochs=10000, patience=50, lr=1e-3, start_side=8, base_ch=192, coeff_hidden=256,
         coeff_depth=2, seed=7, stop=False):

    # Safety & memory
    for g in tf.config.list_physical_devices("GPU"):
        try: tf.config.experimental.set_memory_growth(g, True)
        except: pass
    tf.keras.mixed_precision.set_global_policy("float32")
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    ensure_dirs()

    # Load z (aligned latents)
    z_path = LATENTS_TMPL.format(ad=ad)
    if not os.path.exists(z_path):
        raise FileNotFoundError(f"Missing latents: {z_path}")
    Z = np.load(z_path, mmap_mode="r")
    N = Z.shape[0]

    # Load y target
    y_path = DATA_PATH[mod]
    if not os.path.exists(y_path):
        raise FileNotFoundError(f"Missing modality data: {y_path}")
    Y = np.load(y_path, mmap_mode="r")
    if mod != "coeff":
        Y = as_4d(Y)  # (N,H,W,1)

    if Y.shape[0] < N:
        N = Y.shape[0]
        Z = Z[:N]
    elif Z.shape[0] < Y.shape[0]:
        Y = Y[:Z.shape[0]]
        N = Z.shape[0]

    # Splits
    tr_idx, va_idx, te_idx = load_splits(SPLITS_NPZ)
    tr_idx = tr_idx[tr_idx < N]; va_idx = va_idx[va_idx < N]; te_idx = te_idx[te_idx < N]

    # Build decoder
    if mod == "coeff":
        coeff_dim = int(Y.shape[1])
        dec4 = build_dec4_for_coeff(coeff_dim, ad, hidden=coeff_hidden, depth=coeff_depth)
    else:
        side = side_from_mod(mod)
        dec4 = build_dec4_for_image(side, ad, start_side=start_side, base_ch=base_ch)

    # Data
    ds_tr = make_ds(Z, Y, tr_idx, batch=batch, shuffle=True)
    ds_va = make_ds(Z, Y, va_idx, batch=batch, shuffle=False)
    ds_te = make_ds(Z, Y, te_idx, batch=batch, shuffle=False)

    # Compile — single metric used as loss
    #opt = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
    opt = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-4, clipnorm=1.0)
    loss = ScaledMSE(dataset=ds_tr)
    dec4.compile(optimizer=opt, loss=loss, metrics=[])
    #dec4.summary()
    if stop: exit()

    # Callbacks (monitor the same single metric on val)
    es = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True, min_delta=1e-4, verbose=0)
    rlr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, cooldown=1, verbose=1)
    pp = PatiencePrinter(es)

    # Train
    dec4.fit(ds_tr, validation_data=ds_va, epochs=epochs, callbacks=[pp, rlr, es], verbose=0)

    # evaluate
    tr_rel = loss.eval(dec4, ds_tr)
    va_rel = loss.eval(dec4, ds_va)
    te_rel = loss.eval(dec4, ds_te)
    print(f"\nFinal relmse — train={tr_rel:.6f}  val={va_rel:.6f}  test={te_rel:.6f}")
    df_vec = pd.DataFrame({'Final relmse - train': [tr_rel], 'Final relmse - val': [va_rel], 'Final relmse - test': [te_rel]})


    # Save
    keras_name = f"models/dec_4_{mod}_ld{ad}.keras"
    dec4.save(keras_name)
    print("Saved:", keras_name)
    # Export (SavedModel)
    export_dir = f"exports/dec_4_{mod}_ld{ad}"
    try:
        dec4.export(export_dir)   # Keras 3
    except Exception:
        dec4.save(export_dir)     # Fallback SavedModel
    print("Exported:", export_dir)
    return df_vec

if __name__ == "__main__":
    args = get_args()
    main(args.mod, ad=args.ad, batch=args.batch, epochs=args.epochs, patience=args.patience, lr=args.lr,
         start_side=args.start_side, base_ch=args.base_ch, coeff_hidden=args.coeff_hidden, coeff_depth=args.coeff_depth,
         seed=args.seed, stop=args.stop)

