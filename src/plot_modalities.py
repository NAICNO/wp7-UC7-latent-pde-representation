#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_modalities.py

Looping 2×6 plot:
  For each entity in the chosen split (train/val/test, shuffled by default):

  Top row   : ORIGINAL modalities
              ONE random modality (colored title) is encoded with enc_3_<mod>
              → aligned latent z (ld = 32)
  Bottom row: RECONSTRUCTIONS from z via dec_3_<mod>

Modalities (columns):
  [u16, u32, u64, u128, u256, coeff]
- u* plotted via imshow
- coeff plotted as velocity quiver from streamfunction coefficients
"""

import os, re, glob, argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import TFSMLayer
from keras.models import load_model

# ---------------- config ----------------
# MODS = ["u16","u32","u64","u128", "u256", "coeff"]
# DATA_PATH = {
#     "u16":   "data/05_u16.npy",
#     "u32":   "data/05_u32.npy",
#     "u64":   "data/05_u64.npy",
#     "u128":  "data/05_u128.npy",
#     "u256":  "data/05_u256.npy",
#     "coeff": "data/05_streamfunction_coeffs.npy",
# }
EXTENT = [-1,1,-1,1]

# --------------- helpers ----------------
# def str_to_list(s: str):
#     s = s.strip('[]{}() ')
#     items = s.split(',')
#     if len(items) == 1:
#         return items[0]
#     return [str(i).strip(' ') for i in items]

def load_splits(path="data/05_splits.npz"):
    d = np.load(path, allow_pickle=True)
    for a,b,c in [("train","val","test"),("train_idx","val_idx","test_idx")]:
        if a in d and b in d and c in d:
            return d[a].astype(int), d[b].astype(int), d[c].astype(int)
    raise KeyError("train/val/test not found in splits npz")

def normalize_field(a):
    if a.ndim==4 and a.shape[-1]==1: a=a[...,0]
    if a.ndim!=3: raise ValueError(f"Unexpected shape {a.shape}")
    return a

def infer_degree_from_len(m):
    for n in range(1,20):
        if (n+1)*(n+2)//2 - 1 == m: return n
    raise ValueError(f"Bad coeff length {m}")

def monomial_indices(n):
    out=[]
    for i in range(n+1):
        for j in range(n+1-i):
            if i==0 and j==0: continue
            out.append((i,j))
    return out

def _powers(A,maxd):
    P=[np.ones_like(A)]
    for _ in range(maxd): P.append(P[-1]*A)
    return P

def velocity_from_coeff(cvec,n):
    n_sf=infer_degree_from_len(len(cvec))
    IDX=monomial_indices(n_sf)
    x=np.linspace(-1,1,n); y=np.linspace(-1,1,n)
    X,Y=np.meshgrid(x,y,indexing="ij")
    Xp=_powers(X,max(i for i,_ in IDX))
    Yp=_powers(Y,max(j for _,j in IDX))
    vx=np.zeros_like(X); vy=np.zeros_like(Y)
    for a,(i,j) in zip(cvec,IDX):
        if j>=1: vx+=(j*a)*Xp[i]*Yp[j-1]
        if i>=1: vy-=(i*a)*Xp[i-1]*Yp[j]
    return vx,vy

def _pick_endpoints():
    return ["serving_default","__call__","call","inference","predict"]

def load_from_export_or_keras(prefix,mod):
    pat=re.compile(rf"^{re.escape(prefix)}_{re.escape(mod)}_ld(\d+)$")
    if os.path.isdir("exports"):
        cands=[]
        for d in os.listdir("exports"):
            m=pat.match(d)
            if m: cands.append((int(m.group(1)),os.path.join("exports",d)))
        if cands:
            cands.sort(); ad,path=cands[-1]
            for ep in _pick_endpoints():
                try:
                    layer=TFSMLayer(path,call_endpoint=ep)
                    def _call(x,_layer=layer):
                        y=_layer(x,training=False)
                        if isinstance(y,dict): y=y[sorted(y.keys())[0]]
                        return y
                    return _call,ad,f"export:{path}:{ep}"
                except: continue
    if os.path.isdir("models"):
        cands=sorted(glob.glob(f"models/{prefix}_{mod}_ld*.keras"))
        if cands:
            fn=cands[-1]
            mdl=load_model(fn,compile=False)
            m=re.search(r"_ld(\d+)\.keras$",fn)
            ad=int(m.group(1)) if m else None
            return mdl,ad,f"keras:{fn}"
    raise FileNotFoundError(f"No {prefix}_{mod}")


def get_args():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mods', type=str, nargs='+', default=("u16","u32","u64","u128"))
    ap.add_argument("--split",choices=["train","val","test"],required=True)
    ap.add_argument('--data_prefix', type=str, default="data/05_", help='Prefix of the .npy files')
    ap.add_argument("--seed",type=int,default=7)
    ap.add_argument('--num_plots', type=int, default=5, help='Number of plots to show')
    ap.add_argument("--save",action="store_true")
    ap.add_argument("--outdir",default="results")
    ap.add_argument("--dpi",type=int,default=150)
    ap.add_argument("--cmap",default="viridis")
    ap.add_argument("--quiver-step",type=int,default=12)
    ap.add_argument("--source-color",default="crimson")
    return ap.parse_args()

# --------------- main ----------------
def main(split, mods=("u16","u32","u64","u128"), data_prefix="data/05_", seed=7, num_plots=5, save=False, outdir="results", dpi=150, cmap="viridis", quiver_step=12,
         source_color="crimson"):
    for g in tf.config.list_physical_devices("GPU"):
        try: tf.config.experimental.set_memory_growth(g,True)
        except: pass
    tf.keras.mixed_precision.set_global_policy("float32")

    rng=np.random.default_rng(seed)

    # --- load data ---
    idx_tr,idx_va,idx_te=load_splits(f"{data_prefix}{'splits'}.npz")
    arrays={}
    for m in mods: #"u16","u32","u64","u128","u256"]:
        if m=="coeff": continue
        arrays[m]=normalize_field(np.load(f"{data_prefix}{m}.npy",mmap_mode="r"))
    arrays["coeff"]=np.load(f"{data_prefix}{'streamfunction_coeffs'}.npy",allow_pickle=False)
    N=min(a.shape[0] for a in arrays.values())
    for k in arrays:
        if arrays[k].shape[0]!=N: arrays[k]=arrays[k][:N]
    idx_tr=idx_tr[idx_tr<N]; idx_va=idx_va[idx_va<N]; idx_te=idx_te[idx_te<N]

    if split=="train": split_idx=idx_tr
    elif split=="val": split_idx=idx_va
    else: split_idx=idx_te
    indices=list(split_idx)[:num_plots]
    rng.shuffle(indices)

    # --- load all decoders once ---
    decoders={}
    for m in mods:
        d,_,src=load_from_export_or_keras("dec_3",m)
        decoders[m]=d
        print(f"[load] dec_3[{m}] <- {src}")

    # --- loop over samples ---
    for loop_i,gidx in enumerate(indices,1):
        src_mod=rng.choice(mods)
        enc3,ad,srcinfo=load_from_export_or_keras("enc_3",src_mod)
        print(f"[{loop_i}/{len(indices)}] enc_3[{src_mod}] <- {srcinfo}")

        # encode
        if src_mod=="coeff": x_src=arrays["coeff"][gidx][None,...].astype("float32")
        else: x_src=arrays[src_mod][gidx][None,...,None].astype("float32")
        z=tf.convert_to_tensor(enc3(x_src))
        z=tf.reshape(z,(tf.shape(z)[0],-1))

        # decode all
        preds={}
        for m in mods:
            y=np.array(decoders[m](z))
            if m=="coeff": preds[m]=y[0]
            else:
                y=y[0]
                if y.ndim==3 and y.shape[-1]==1: y=y[...,0]
                preds[m]=y

        # --- figure ---
        fig,axs=plt.subplots(2,len(mods),figsize=(6*3.2,2*3.2),constrained_layout=False)
        fig.subplots_adjust(left=0.05,right=0.95,wspace=0.3,hspace=0.3)
        axs=np.asarray(axs)

        def draw_row(axrow,which):
            for j,mod in enumerate(mods):
                ax=axrow[j]; ax.set_xticks([]); ax.set_yticks([])
                if which=="top" and mod==src_mod:
                    ax.set_title(mod,color=source_color,fontsize=11,fontweight="bold")
                else: ax.set_title(mod,fontsize=11)
                if mod=="coeff":
                    coeff=arrays["coeff"][gidx] if which=="top" else preds["coeff"]
                    n=256; vx,vy=velocity_from_coeff(coeff,n)
                    x=np.linspace(-1,1,n); y=np.linspace(-1,1,n)
                    X,Y=np.meshgrid(x,y,indexing="xy")
                    step=quiver_step
                    ax.quiver(X[::step,::step],Y[::step,::step],
                              vx.T[::step,::step],vy.T[::step,::step],
                              angles="xy",scale_units="xy",scale=None,width=0.0025,color="k")
                    ax.set_xlim(-1,1); ax.set_ylim(-1,1); ax.set_aspect("equal","box")
                else:
                    U=arrays[mod][gidx] if which=="top" else preds[mod]
                    im=ax.imshow(U.T,origin="lower",extent=EXTENT,cmap=cmap,
                                 vmin=float(np.min(U)),vmax=float(np.max(U)))
                    cbar=fig.colorbar(im,ax=ax,fraction=0.046,pad=0.02)
                    cbar.ax.set_ylabel("u",rotation=0,labelpad=10,fontsize=9)

        draw_row(axs[0,:],"top")
        draw_row(axs[1,:],"bottom")
        fig.suptitle(f"Sample {loop_i}/{len(indices)} · split={split} · src={src_mod} (ld={ad}) · global idx={gidx}",
                     fontsize=12)
        fig.text(0.995,0.01,f"split: {split} | src: {src_mod}",ha="right",va="bottom",
                 fontsize=10,bbox=dict(boxstyle="round,pad=0.25",facecolor="white",alpha=0.7,ec="0.7"))

        if save:
            os.makedirs(outdir,exist_ok=True)
            out=os.path.join(outdir,f"do0510_{split}_{loop_i:05d}_src-{src_mod}.png")
            fig.savefig(out,dpi=dpi,bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: {out}")
        else:
            print(f"Showing {loop_i}/{len(indices)} (src={src_mod}) — close window for next.")
            plt.show()

if __name__=="__main__":
    args = get_args()
    main(args.split, args.mods, args.data_prefix, args.seed, args.num_plots, args.save, args.outdir, args.dpi, args.cmap, args.quiver_step, )

