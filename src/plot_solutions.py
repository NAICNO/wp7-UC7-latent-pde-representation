#!/usr/bin/env python3
"""
Plot 5 resolutions side-by-side (left->right: 16, 32, 64, 128, 256)
from:
  data/04-solutions-16.npy
  data/04-solutions-32.npy
  data/04-solutions-64.npy
  data/04-solutions-128.npy
  data/04-solutions-256.npy

One figure per sample index. Either show interactively, or save PNGs to results/.
"""

import os
import sys
import numpy as np
import argparse
import matplotlib.pyplot as plt

def get_args():
    # ------------ user options ------------
    # save = False               # e.g. True to save images instead of showing; None/False to show
    # results_dir = "results"   # where to save if save=True
    # selection = 'rnd'          # e.g. [0,1,2,3,4] to only plot those indices; or None for all
    # cmap = "viridis"
    # extent = [-1, 1, -1, 1]   # consistent display coordinates
    # dpi = 150                 # PNG DPI when saving
    # --------------------------------------
    p = argparse.ArgumentParser(description="Plot solutions")
    p.add_argument('--data_prefix', type=str, default="data/05_", help='Prefix of the .npy files')
    p.add_argument('--levels', type=int, nargs='+', default=[16, 32, 64, 128, 256], help='Resolutions to plot')
    p.add_argument("--save", type=bool, default=False, help="True to save images instead of showing; None/False to show")
    p.add_argument("--results_dir", type=str, default="results", help="Where to save if save=True")
    p.add_argument("--selection", type=str, default="rnd", help="E.g. [0,1,2,3,4] to only plot those indices; or None for all")
    p.add_argument("--cmap", type=str, default="viridis", help="The colormap")
    p.add_argument("--extent", type=int, nargs=4, default=[-1, 1, -1, 1], help="Consistent display coordinates")
    p.add_argument("--dpi", type=int, default=150, help="PNG DPI when saving")
    return p.parse_args()

def str_to_list(s: str):
    s = s.strip('[]{}() ')
    items = s.split(',')
    if len(items) == 1:
        return items[0]
    return [int(i) for i in items]

def main(data_prefix="data/05_", levels=(16, 32, 64, 128, 256), save=False, results_dir="results", selection='rnd', cmap="viridis", extent=(-1, 1, -1, 1), dpi=150):
    # Files in increasing resolution order
    entries = []
    for level in levels:
        entries.append((level, f"{data_prefix}u{level}.npy"))

    # --- check files exist ---
    missing = [p for _, p in entries if not os.path.exists(p)]
    if missing:
        sys.exit("ERROR: Missing files:\n  " + "\n  ".join(missing))

    # --- load (memory-mapped) ---
    arrays = {}
    ns_all = []
    for res, path in entries:
        arr = np.load(path, mmap_mode="r")   # shape: (nsamples, res, res)
        if arr.ndim != 3 or arr.shape[1] != res or arr.shape[2] != res:
            sys.exit(f"ERROR: Unexpected shape in {path}: {arr.shape}, expected (N,{res},{res})")
        arrays[res] = arr
        ns_all.append(arr.shape[0])

    # Use the minimum number of samples across all files (safest)
    ns = min(ns_all)
    if any(n != ns for n in ns_all):
        print("[WARN] Number of samples differs across files. Using ns =", ns)

    # apply selection if provided
    if selection is None:
        indices = list(range(ns))
        print("Plotting all samples.")
    selection_list = selection
    if isinstance(selection, str):
        selection_list = str_to_list(selection)
    if isinstance(selection_list, list) or isinstance(selection, tuple):
        # filter to valid indices
        selection = [i for i in selection_list if 0 <= i < ns]
        if not selection:
            sys.exit("ERROR: selection is empty or out of range.")
        indices = selection
    elif selection_list == 'rnd':
        import random
        indices = list(range(ns))
        random.shuffle(indices)
    else:
        indices = list(range(ns))

    print(f"Loaded {ns} samples (will plot {len(indices)}).")

    if save:
        os.makedirs(results_dir, exist_ok=True)

    pad = len(str(ns))

    # --- main loop: one row (5 images) per sample ---
    for k in indices:
        # fetch the five resolutions in order
        panels = [(res, arrays[res][k]) for res, _ in entries]

        # consistent color scaling across all five for fair visual comparison
        #vmin = min(u.min() for _, u in panels)
        #vmax = max(u.max() for _, u in panels)

        fig, axs = plt.subplots(1, len(entries), figsize=(5 * 3.6, 3.6), constrained_layout=True)
        if not isinstance(axs, (list, np.ndarray)):
            axs = [axs]

        for ax, (res, u) in zip(axs, panels):
            vmin=u.min(); vmax=u.max()
            im = ax.imshow(u, origin="lower", cmap=cmap, extent=extent, vmin=vmin, vmax=vmax)
            ax.set_title(f"{res}×{res}, (min,max)=({u.min():.2e}, {u.max():.2e})", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])

        # Optional: one shared colorbar for the row
        cbar = fig.colorbar(im, ax=axs, fraction=0.025, pad=0.02)
        cbar.ax.set_ylabel("u", rotation=0, labelpad=10)

        if save:
            out_path = os.path.join(results_dir, f"solutions_row_{str(k).zfill(pad)}.png")
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            print(f"[{indices.index(k)+1}/{len(indices)}] Saved: {out_path}")
        else:
            print(f"Showing sample {indices.index(k)+1}/{len(indices)} (index={k}). Close to continue…")
            plt.show()

if __name__ == '__main__':
    args = get_args()
    main(data_prefix=args.data_prefix, levels=args.levels, save=args.save, results_dir=args.results_dir,
         selection=(1, 2, 3), cmap=args.cmap, extent=args.extent, dpi=args.dpi)