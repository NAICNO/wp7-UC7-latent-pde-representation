#!/usr/bin/env python3
"""
do0401_cd2d_streamfunc.py — CD2D solutions with divergence-free convection from a polynomial streamfunction

Generates a family of 2D steady convection–diffusion solutions on [-1,1]^2
with Dirichlet BCs, constant diffusion, and divergence-free advection
v = (∂ψ/∂y, -∂ψ/∂x) derived from a polynomial streamfunction ψ(x,y) of total
degree n_sf. The constant term of ψ is irrelevant and omitted.

Key points
----------
• Domain: [-1,1]^2, cell-centered grid; levels from --levels (default 16 32 64 128 256)
• Diffusion: constant 1
• Advection: conservative full upwind (Godunov) on faces
• Streamfunction: ψ(x,y) = Σ_{i+j≤n_sf, (i,j)≠(0,0)} a_{ij} x^i y^j
  n_coeff = 0.5*(n_sf+1)*(n_sf+2) - 1
• Coeff sampling: i.i.d. Normal(0,1) by default (see --coeff-mode); per-sample velocity
  field is scaled so RMS speed on cell centers equals --vel-scale
• BCs (fixed across samples): Non-zero Dirichlet on ALL four boundaries (Gaussian bumps)
• Outputs:
  - Solutions: <out-prefix>-<n>.npy with shape (N, n, n) (header-safe open_memmap)
  - Coeffs: data/04_streamfunction_coeffs.npy with shape (N, n_coeff)
• Optional post-processing: --center, --l2norm (saved with allow_pickle=False)

Example
-------
python do0401_cd2d_streamfunc.py \
  --n-sol 360 --levels 16 32 64 128 \
  --n-sf 4 --vel-scale 1e5 --workers 24 \
  --out-prefix data/04-solutions --center --l2norm
"""

import os
import argparse
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from time import time
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
from numpy.lib.format import open_memmap

# Avoid BLAS oversubscription with processes
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# ---------------- CLI ----------------
def get_args():
    p = argparse.ArgumentParser(description="Generate CD2D solutions with streamfunction-driven divergence-free advection (header-safe .npy).")
    p.add_argument("--n-sol", type=int, default=1000, help="Number of samples.")
    p.add_argument("--levels", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    p.add_argument("--workers", type=int, default=min(24, os.cpu_count() or 24))
    p.add_argument("--seed", type=int, default=12345, help="Random seed for coefficient sampling and shuffling.")

    # Streamfunction / velocity scaling
    p.add_argument("--n-sf", type=int, default=4, help="Total polynomial degree of the streamfunction ψ.")
    p.add_argument("--coeff-mode", type=str, default="uniform", choices=["normal","uniform"],
                   help="Coefficient sampling distribution (i.i.d. per monomial).")
    p.add_argument("--vel-scale", type=float, default=100_000.0,
                   help="Target RMS speed on cell centers after scaling each sample's velocity field.")

    # Output
    p.add_argument("--out-prefix", type=str, default="data/05_u")
    p.add_argument("--coeff-out", type=str, default="data/05_streamfunction_coeffs.npy",
                   help="Path for saving used (scaled) streamfunction coefficients (N, n_coeff).")

    # Solver & debug
    p.add_argument("--solver", type=str, default="auto", choices=["auto","scipy","pardiso"],
                   help="Linear solver backend.")
    p.add_argument("--jacobi-scale", action="store_true",
                   help="Apply Jacobi row scaling before solve (stability aid).")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--debug-k", type=int, default=0)

    # Optional post-processing
    p.add_argument("--center", action="store_true", help="Mean-center each sample before final save.")
    p.add_argument("--l2norm", action="store_true", help="L2-normalize each sample after centering (if any).")

    return p.parse_args()

# --------------- Geometry & fields ---------------
def make_grid(n: int):
    """Cell-centered grid on [-1,1]^2 with n×n cells."""
    x = np.linspace(-1.0, 1.0, n)
    y = np.linspace(-1.0, 1.0, n)
    h = x[1] - x[0]
    Xc, Yc = np.meshgrid(x, y, indexing="ij")
    # Face coordinates
    x_e = 0.5 * (x[:-1] + x[1:])            # (n-1,)
    y_n = 0.5 * (y[:-1] + y[1:])            # (n-1,)
    Xef, Yef = np.meshgrid(x_e, y,  indexing="ij")   # (n-1, n)  x-faces (east/west)
    Xnf, Ynf = np.meshgrid(x,   y_n, indexing="ij")  # (n, n-1)  y-faces (north/south)
    return x, y, Xc, Yc, Xef, Yef, Xnf, Ynf, h

def constant_diffusion():
    def d_on_centers(X, Y):
        return np.ones_like(X, dtype=np.float64)
    return d_on_centers

def gaussian_all_boundaries(n: int, sigma: float = 0.25):
    """Dirichlet g[i,j] on [-1,1]^2: non-zero on *all four* boundaries.
       Left/right (x=±1): Gaussian in y, peak 1 at y=0; Bottom/top (y=±1): Gaussian in x, peak 1 at x=0.
       Corners take the max of the two (so peak remains 1).
    """
    x = np.linspace(-1.0, 1.0, n)
    y = np.linspace(-1.0, 1.0, n)
    g = np.zeros((n, n), dtype=np.float64)
    gy = np.exp(-0.5 * (y / sigma) ** 2)  # along y
    gx = np.exp(-0.5 * (x / sigma) ** 2)  # along x
    g[0, :]  = np.maximum(g[0, :],  gy)
    g[-1, :] = np.maximum(g[-1, :], gy)
    g[:, 0]  = np.maximum(g[:, 0],  gx)
    g[:, -1] = np.maximum(g[:, -1], gx)
    return g

# ---------- Streamfunction monomials & velocity evaluation ----------
def monomial_multiindices(n_sf: int):
    """List of (i,j) with i+j ≤ n_sf, excluding (0,0)."""
    idx = []
    for i in range(n_sf + 1):
        for j in range(n_sf + 1 - i):
            if i == 0 and j == 0:
                continue
            idx.append((i, j))
    return idx  # length = 0.5*(n_sf+1)*(n_sf+2)-1

def sample_coeffs(n_sf: int, n_sol: int, mode: str = "normal", seed: int = 12345):
    """Return (coeffs, idx) where coeffs has shape (n_sol, n_coeff)."""
    rng = np.random.default_rng(seed)
    idx = monomial_multiindices(n_sf)
    n_coeff = len(idx)
    if mode == "normal":
        C = rng.normal(0.0, 1.0, size=(n_sol, n_coeff))
    else:  # uniform
        C = rng.uniform(-1.0, 1.0, size=(n_sol, n_coeff))
    return C.astype(np.float64), idx

def eval_velocity_from_streamfunc_coeffs(coeffs, idx, Xc, Yc, Xef, Yef, Xnf, Ynf, target_rms=None):
    """
    Given streamfunction coefficients (aligned with idx list), evaluate
    v = (∂ψ/∂y, -∂ψ/∂x) on centers and faces. Optionally scale so that
    RMS speed on centers equals target_rms (if provided).

    Returns:
      (vx_c, vy_c), (vx_f, vy_f), scaled_coeffs
      where vx_f is on x-faces (shape of Xef), vy_f on y-faces (shape of Xnf)
    """
    # Precompute powers for centers, x-faces, y-faces
    def powers(A, max_deg):
        # returns list [A^0, A^1, ..., A^max_deg]
        P = [np.ones_like(A, dtype=np.float64)]
        for _ in range(max_deg):
            P.append(P[-1] * A)
        return P

    max_i = max(i for i, _ in idx)
    max_j = max(j for _, j in idx)

    XcP = powers(Xc, max_i); YcP = powers(Yc, max_j)
    XefP = powers(Xef, max_i); YefP = powers(Yef, max_j)
    XnfP = powers(Xnf, max_i); YnfP = powers(Ynf, max_j)

    # v_x = ∂ψ/∂y = sum a_{ij} * j * x^i * y^{j-1}  (j>=1)
    # v_y = -∂ψ/∂x = -sum a_{ij} * i * x^{i-1} * y^j (i>=1)
    vx_c = np.zeros_like(Xc, dtype=np.float64)
    vy_c = np.zeros_like(Yc, dtype=np.float64)
    vx_f = np.zeros_like(Xef, dtype=np.float64)  # on x-faces
    vy_f = np.zeros_like(Xnf, dtype=np.float64)  # on y-faces

    for a, (i, j) in zip(coeffs, idx):
        if j >= 1:
            jfac = j * a
            vx_c += jfac * XcP[i]  * YcP[j - 1]
            vx_f += jfac * XefP[i] * YefP[j - 1]
        if i >= 1:
            ifac = i * a
            vy_c -= ifac * XcP[i - 1]  * YcP[j]
            vy_f -= ifac * XnfP[i - 1] * YnfP[j]

    # Scale to target RMS speed on centers if requested
    scale = 1.0
    if target_rms is not None:
        speed2 = vx_c**2 + vy_c**2
        rms = float(np.sqrt(np.mean(speed2)))
        if rms > 0:
            scale = target_rms / rms
            vx_c *= scale; vy_c *= scale
            vx_f *= scale; vy_f *= scale

    # Return scaled coefficients actually used (psi → scale*psi ⇒ coeffs → scale*coeffs)
    scaled_coeffs = np.asarray(coeffs) * scale
    return (vx_c, vy_c), (vx_f, vy_f), scaled_coeffs

# --------------- FV assembly (diffusion + conservative upwind convection) ---------------
def assemble_fv_system(n, vx_f, vy_f, dfun):
    x, y = np.linspace(-1.0, 1.0, n), np.linspace(-1.0, 1.0, n)
    Xc, Yc = np.meshgrid(x, y, indexing="ij")
    h = x[1] - x[0]

    # Diffusion at centers and faces (arithmetic avg for faces)
    d_c = dfun(Xc, Yc)
    De = 0.5 * (d_c[1:, :] + d_c[:-1, :])    # (n-1, n) east/west faces
    Dn = 0.5 * (d_c[:, 1:] + d_c[:, :-1])    # (n, n-1) north/south faces

    # Interior indices (exclude boundary)
    I = np.arange(1, n - 1)
    J = np.arange(1, n - 1)
    II, JJ = np.meshgrid(I, J, indexing="ij")
    p  = (II * n + JJ).ravel(order="C")
    pE = p + 1
    pW = p - 1
    pN = p - n
    pS = p + n

    vec = lambda a: np.asarray(a).ravel(order="C")

    # Diffusion contributions (5-pt)
    De_c = vec(De[II, JJ])
    Dw_c = vec(De[II - 1, JJ])
    Dn_c = vec(Dn[II, JJ])
    Ds_c = vec(Dn[II, JJ - 1])

    diff_diag = (De_c + Dw_c + Dn_c + Ds_c) / (h * h)
    diff_e = -De_c / (h * h)
    diff_w = -Dw_c / (h * h)
    diff_n = -Dn_c / (h * h)
    diff_s = -Ds_c / (h * h)

    # Conservative upwind advection (Godunov)
    vxe = vec(vx_f[II, JJ])
    vxw = vec(vx_f[II - 1, JJ])
    vyn = vec(vy_f[II, JJ])
    vys = vec(vy_f[II, JJ - 1])

    adv_diag = (np.maximum(vxe, 0.0) - np.minimum(vxw, 0.0) +
                np.maximum(vyn, 0.0) - np.minimum(vys, 0.0)) / h
    adv_e =  np.minimum(vxe, 0.0) / h
    adv_w = -np.maximum(vxw, 0.0) / h
    adv_n =  np.minimum(vyn, 0.0) / h
    adv_s = -np.maximum(vys, 0.0) / h

    rows = np.concatenate([p, p, p, p, p])
    cols = np.concatenate([p, pS, pN, pE, pW])
    data = np.concatenate([
        diff_diag + adv_diag,
        diff_e + adv_e,
        diff_w + adv_w,
        diff_n + adv_n,
        diff_s + adv_s,
    ])

    A = sp.coo_matrix((data, (rows, cols)), shape=(n * n, n * n)).tocsr()
    A.sort_indices(); A.sum_duplicates()
    return A, h

def apply_dirichlet(A, s_grid, g_grid=None):
    """Apply Dirichlet BCs via row overwrite.
       s_grid is the cell-integrated RHS (here zeros); g_grid gives boundary values.
    """
    n2 = A.shape[0]
    n = int(np.sqrt(n2))
    if g_grid is None:
        g_grid = np.zeros((n, n), dtype=np.float64)
    bmask2d = np.zeros((n, n), dtype=bool)
    bmask2d[0, :] = True; bmask2d[-1, :] = True; bmask2d[:, 0] = True; bmask2d[:, -1] = True
    bmask = bmask2d.ravel(order="C")
    rows = np.flatnonzero(bmask)

    b = s_grid.ravel(order="C").astype(np.float64, copy=True)
    A = A.tolil(copy=False)
    A[rows, :] = 0.0; A[rows, rows] = 1.0
    b[bmask] = g_grid.ravel(order="C")[bmask]
    A = A.tocsr(); A.sort_indices(); A.sum_duplicates()
    return A, b

# --------------- Solve (with optional debug) ---------------
def solve_level(n, coeffs, idx, vel_scale, jacobi_scale=False, solver="auto", debug=False, dbg_tag=""):
    # Build coordinates
    x, y = np.linspace(-1.0, 1.0, n), np.linspace(-1.0, 1.0, n)
    Xc, Yc = np.meshgrid(x, y, indexing="ij")
    x_e = 0.5 * (x[:-1] + x[1:])
    y_n = 0.5 * (y[:-1] + y[1:])
    Xef, Yef = np.meshgrid(x_e, y,  indexing="ij")
    Xnf, Ynf = np.meshgrid(x,   y_n, indexing="ij")

    # Evaluate velocity from streamfunction and scale to target RMS
    (_, _), (vx_f, vy_f), _ = eval_velocity_from_streamfunc_coeffs(
        coeffs, idx, Xc, Yc, Xef, Yef, Xnf, Ynf, target_rms=vel_scale
    )

    dfun = constant_diffusion()

    # Assemble system
    A, h = assemble_fv_system(n, vx_f, vy_f, dfun)
    # No source term
    s = np.zeros((n, n), dtype=np.float64)
    g = gaussian_all_boundaries(n, sigma=0.05)
    A, b = apply_dirichlet(A, s, g_grid=g)

    # Optional scaling
    if jacobi_scale:
        A = A.tocsr()
        D = np.abs(A.diagonal()) + 1e-30
        Dinv = 1.0 / D
        A_sys = (sp.diags(Dinv) @ A).tocsc()
        b_sys = Dinv * b
    else:
        A_sys, b_sys = A.tocsc(), b

    # Solve
    u_flat = None
    if solver in ("pardiso", "auto"):
        try:
            from pypardiso import spsolve as pardiso_spsolve
            u_flat = pardiso_spsolve(A_sys, b_sys)
        except Exception:
            if solver == "pardiso":
                raise
    if u_flat is None:
        u_flat = spla.spsolve(A_sys, b_sys)

    u = u_flat.reshape((n, n))

    if debug:
        # Simple diagnostics
        u_inf = float(np.max(np.abs(u)))
        r = A @ u_flat - b
        r_inf = float(np.max(np.abs(r)))
        print(f"[dbg {dbg_tag} n={n}] h={h:.3e}  |u|_inf={u_inf:.3e}  |Au-b|_inf={r_inf:.3e}")

    return u

# --------------- Worker ---------------
def _compute_and_write(k, *, coeff_k, idx, levels, n_sol, out_paths, vel_scale, solver, jacobi_scale, debug, debug_k):
    # single-thread per worker
    for var in ("MKL_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"

    for n, path in zip(levels, out_paths):
        u = solve_level(
            n, coeff_k, idx, vel_scale,
            jacobi_scale=jacobi_scale, solver=solver,
            debug=(debug and k == debug_k), dbg_tag=f"sample{k}"
        )
        mm = open_memmap(path, mode='r+', dtype=np.float64, shape=(n_sol, n, n))
        mm[k] = u; mm.flush()
    return k

# --------------- Main ---------------
def main(n_sol=1000, levels=(16, 32, 64, 128, 256), workers=24, seed=12345, n_sf=4, coeff_mode="uniform",
         vel_scale=100000, out_prefix="data/05_u", coeff_out="data/coeffs.npy", solver="auto", jacobi_scale=False,
        debug=False, debug_k=0, center=False, l2norm=False):
    t0 = time()
    levels = tuple(sorted(set(int(n) for n in levels)))

    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(coeff_out) or ".", exist_ok=True)

    # Prepare outputs (solutions)
    out_paths = []
    for n in levels:
        path = f"{out_prefix}{n}.npy"
        mm = open_memmap(path, mode='w+', dtype=np.float64, shape=(n_sol, n, n)); mm.flush(); del mm
        out_paths.append(path)
    out_paths = tuple(out_paths)

    # Sample base coefficients (unscaled)
    C_raw, idx = sample_coeffs(n_sf, n_sol, mode=coeff_mode, seed=seed)

    # We will fill C_used with the actually used (scaled-to-RMS) coefficients.
    C_used = np.empty_like(C_raw)

    # Build coordinates at one representative level to compute scaling per sample.
    # Use the finest level for stable RMS estimation.
    n_ref = max(levels)
    x_ref = np.linspace(-1.0, 1.0, n_ref)
    y_ref = np.linspace(-1.0, 1.0, n_ref)
    Xc_ref, Yc_ref = np.meshgrid(x_ref, y_ref, indexing="ij")
    x_e_ref = 0.5 * (x_ref[:-1] + x_ref[1:])
    y_n_ref = 0.5 * (y_ref[:-1] + y_ref[1:])
    Xef_ref, Yef_ref = np.meshgrid(x_e_ref, y_ref,  indexing="ij")
    Xnf_ref, Ynf_ref = np.meshgrid(x_ref,   y_n_ref, indexing="ij")

    # Precompute per-sample scaling by evaluating RMS speed on centers at reference level
    scales = np.empty((n_sol,), dtype=np.float64)
    for k in range(n_sol):
        (_, _), (_, _), coeffs_scaled = eval_velocity_from_streamfunc_coeffs(
            C_raw[k], idx, Xc_ref, Yc_ref, Xef_ref, Yef_ref, Xnf_ref, Ynf_ref, target_rms=vel_scale
        )
        # scale factor s satisfies coeffs_scaled = s * C_raw[k]
        s = (coeffs_scaled[0] / C_raw[k][0]) if C_raw[k][0] != 0 else 1.0
        scales[k] = s
        C_used[k] = coeffs_scaled

    # Launch workers; pass unscaled coeffs and let each call recompute scaling internally
    # (small overhead; ensures consistency even if levels differ)
    worker = partial(
        _compute_and_write,
        idx=idx,
        levels=levels, n_sol=n_sol, out_paths=out_paths,
        vel_scale=float(vel_scale),
        solver=solver, jacobi_scale=jacobi_scale,
        debug=debug, debug_k=debug_k
    )

    n_workers = max(1, int(workers))
    print(f"Generating {n_sol} solutions at levels {list(levels)} with {n_workers} workers...")
    done = 0
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = [ex.submit(worker, k, coeff_k=C_raw[k]) for k in range(n_sol)]
        for f in as_completed(futs):
            _ = f.result(); done += 1
            if done % 50 == 0 or done == n_sol:
                print(f"[{done}/{n_sol}] samples done")

    # Save the used (scaled) coefficients, header-safe (no pickle)
    tmpc = coeff_out + ".tmp.npy"
    np.save(tmpc, C_used, allow_pickle=False)
    os.replace(tmpc, coeff_out)
    print(f"Wrote streamfunction coefficients to: {coeff_out} (shape {C_used.shape})")

    # Optional post-processing (no-pickle save with atomic replace)
    if center or l2norm:
        for path, n in zip(out_paths, levels):
            mm = np.load(path, mmap_mode='r', allow_pickle=False)
            arr = np.array(mm)
            if center:
                arr -= np.mean(arr, axis=(1, 2), keepdims=True)
            if l2norm:
                norms = np.linalg.norm(arr, axis=(1, 2), keepdims=True)
                arr /= (norms + 1e-12)
            tmp = path.replace('.npy', '.tmp.npy')
            np.save(tmp, arr, allow_pickle=False)
            os.replace(tmp, path)
            print(f"Wrote post-processed (no-pickle) array to: {path}")

    dt = time() - t0
    print(f"Total time: {dt:.1f}s | per-sample: {dt/float(n_sol):.2f}s")
    print("Notes:")
    print("  • Divergence-free advection from ψ(x,y) polynomial of degree n_sf; constant term omitted.")
    print("  • Each sample’s velocity field scaled to RMS speed = --vel-scale on centers (reference = finest level).")
    print("  • Gaussian Dirichlet on ALL four boundaries (σ=0.25); no source term.")
    print("  • Files: " + ", ".join(out_paths))
    print("  • Coeffs: " + coeff_out)

if __name__ == "__main__":
    args = get_args()
    main(n_sol=args.n_sol, levels=args.levels, workers=args.workers, seed=args.seed, n_sf=args.n_sf, coeff_mode=args.coeff_mode,
         vel_scale=args.vel_scale, out_prefix=args.out_prefix, coeff_out=args.coeff_out, solver=args.solver, jacobi_scale=args.jacobi_scale,
         debug=args.debug, debug_k=args.debug_k, center=args.center, l2norm=args.l2norm)

