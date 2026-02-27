# Analyzing Results

```{objectives}
- Interpret Relative Energy Error (REE) values and understand what constitutes good vs. poor reconstruction
- Analyze latent-space alignment quality using pairwise metrics and Procrustes diagnostics
- Evaluate cross-modal reconstruction fidelity and identify failure modes
- Use visualization scripts to build intuition about learned representations
- Distinguish between alignment quality issues and reconstruction quality issues
```

```{admonition} What to expect from a successful experiment
:class: tip

A well-trained pipeline should exhibit:
- **REE below 0.01 (1%)** for same-modality reconstruction (encode and decode the same resolution)
- **REE below 0.05 (5%)** for cross-modal reconstruction (encode one resolution, decode another)
- **Pairwise latent REE** that decreases substantially after Procrustes alignment
- **Test set performance** comparable to validation set (no overfitting)
```

## Scope of Analysis

The analysis addresses three complementary questions:

1. **Are modality-specific latent representations well aligned?** (geometric consistency)
2. **Does the joint latent representation preserve sufficient information for reconstruction?** (information preservation)
3. **How do errors vary across modalities, resolutions, and data splits?** (generalization)

Visualization-only scripts (`plot_solutions.py`, called in notebook section "3. Plot solutions") play a supportive role and are not part of the quantitative evaluation.

---

## Latent-Space Alignment Analysis

### Pairwise Modality Consistency

**Script:** `analyze_latent_alignment.py`
**Notebook section:** "7. Analyze the alignments"

This analysis operates on second-level modality latents (`lat_2_*_ld32.npy`) and computes:

- **Pairwise REE** between all modality pairs (raw latent vectors)
- **REE after centering** (mean subtraction) to remove global offset differences
- **REE after Procrustes alignment** to measure best-case geometric agreement

**Outputs:**

- Pairwise REE matrices (CSV) for each alignment type
- REE of each modality relative to the canonical mean latent
- The canonical joint latent representation (`lat_3_ld32.npy`)

### Interpreting Alignment Metrics

| Observation | Interpretation | Action |
|-------------|---------------|--------|
| Low raw REE (<0.01) | Strong intrinsic alignment | Latent spaces naturally agree |
| High raw REE, low Procrustes REE | Differences are primarily rotational | Alignment is working correctly |
| High Procrustes REE | Fundamental structural disagreement | Check autoencoder training quality |
| One modality with much higher REE-to-mean | Outlier modality | May need more training epochs or different architecture |
| Train REE << Test REE | Overfitting in alignment | Reduce model capacity or add regularization |

```{admonition} What Procrustes alignment tells you
:class: note

Procrustes finds the optimal rotation and reflection to align two point clouds. If the REE drops dramatically after Procrustes, it means the two latent spaces have learned similar *structure* but different *orientation*. The alignment network's job is essentially to learn this rotation -- so large Procrustes improvement is expected before alignment training and should diminish after.
```

---

## Reconstruction and Encoding Errors

### Encoding Accuracy

**Script:** `compute_errors.py`
**Notebook section:** "11. Compute errors"

For each modality and data split (train/val/test), the encoding REE measures:

```
REE_encode = || encoder(x) - z_joint ||^2 / || z_joint ||^2
```

where `z_joint` is the canonical joint latent. This quantifies how consistently different encoders map physical inputs to the shared latent space.

### Decoding Accuracy

Using the same script, decoding REE is computed:

```
REE_decode = || decoder(z_joint) - x ||^2 / || x ||^2
```

This measures how faithfully each decoder reconstructs its modality from the joint latent.

### Reading the Error Tables

Results are reported per modality and per split. A typical output looks like:

| Modality | Train REE | Val REE | Test REE |
|----------|-----------|---------|----------|
| u_32 | 0.003 | 0.004 | 0.004 |
| u_64 | 0.005 | 0.006 | 0.007 |
| u_128 | 0.008 | 0.010 | 0.011 |
| u_256 | 0.012 | 0.015 | 0.016 |
| streamfunc | 0.001 | 0.001 | 0.002 |

**What to look for:**

- **Monotonic increase with resolution**: Higher-resolution fields are harder to reconstruct, so REE typically increases from u_16 to u_256. This is expected.
- **Train-test gap**: A small gap (within 2x) is normal. A large gap indicates overfitting.
- **Streamfunction REE**: Should be very low since coefficient vectors are already low-dimensional.

### What Good Results Look Like

- Same-modality reconstruction REE: **0.001 - 0.01** (excellent), **0.01 - 0.05** (acceptable)
- Cross-modal reconstruction REE: **0.01 - 0.05** (good), **0.05 - 0.10** (marginal)
- Encoding REE: **< 0.01** (encoders agree on the joint latent)

### What Bad Results Look Like

- REE above 0.10 for any modality: the autoencoder is not compressing well enough
- Test REE more than 5x train REE: severe overfitting
- Cross-modal REE much larger than same-modal REE: alignment failed
- One modality with dramatically higher error: that modality's autoencoder needs retraining

---

## Qualitative Cross-Modality Inspection

**Script:** `plot_modalities.py`
**Notebook section:** "10. Plot modalities"

This script provides visual validation:

1. Selects a random source modality and sample
2. Encodes the sample into the joint latent space
3. Decodes from the joint latent into **all** target modalities
4. Displays original fields and reconstructions side-by-side

### What to Look for in Plots

**Good signs:**

- Reconstructed solution fields visually match originals at all resolutions
- Boundary layers and sharp features are preserved (not smoothed away)
- Cross-resolution reconstructions maintain consistent physical structure

**Warning signs:**

- **Systematic smoothing**: Sharp features (boundary layers, internal layers) are blurred. This suggests the autoencoder latent dimension is too small.
- **Checkerboard artifacts**: Grid-scale oscillations in reconstructions. This suggests the decoder architecture has issues (common with transposed convolutions).
- **Resolution-dependent bias**: Low-resolution reconstructions look fine but high-resolution ones fail. This suggests the high-resolution autoencoder needs more capacity.
- **Modality confusion**: Reconstructed streamfunction coefficients do not correspond to the velocity field seen in the solution. This indicates an alignment failure.

---

## Supporting Visualization

**Script:** `plot_solutions.py`
**Notebook section:** "3. Plot solutions"

Visualizes raw PDE solutions at multiple spatial resolutions. Used to:

- Build intuition about how solutions vary across the parameter space
- Provide reference fields for visual comparison with reconstructions
- Observe how advection-dominated solutions differ from diffusion-dominated ones

No quantitative metrics are derived from this script.

---

## Latent Space Visualization Tips

While the latent space is typically 32-dimensional, useful low-dimensional projections include:

- **PCA (first 2-3 components)**: Shows the dominant variation directions. If modality latents cluster by modality rather than by sample, alignment has failed.
- **t-SNE or UMAP**: Shows local neighborhood structure. Aligned modalities should intermingle; unaligned ones form separate clusters.
- **Pairwise scatter plots**: Plot latent dimension *i* vs. dimension *j* for different modalities. Aligned modalities should show correlated clouds.

```{admonition} Visualization is not evaluation
:class: warning

Low-dimensional projections can be misleading. Two latent spaces that look overlapping in a PCA projection may still have high REE in the full 32-dimensional space. Always use quantitative metrics (REE tables) as the primary evaluation and visualization as a supplementary check.
```

---

## Result Artifacts

Generated analysis artifacts include:

| Artifact | Location | Description |
|----------|----------|-------------|
| Pairwise REE matrices | `results/*.csv` | Raw, centered, and Procrustes-aligned REE between all modality pairs |
| Error summaries | `results/*.csv` | Per-modality, per-split encoding and decoding REE |
| Joint latent | `latents/lat_3_ld32.npy` | Canonical joint latent representation |
| Reconstruction plots | `results/*.png` | Cross-modal reconstruction visualizations |
| Solution visualizations | `results/*.png` | Raw PDE solutions at multiple resolutions |

---

## Summary

The analysis framework combines three complementary approaches:

1. **Geometric latent-space diagnostics** -- Are the latent representations structurally aligned across modalities?
2. **Quantitative reconstruction error metrics** -- Can the original data be faithfully recovered from the joint latent?
3. **Qualitative cross-modality visual inspection** -- Do reconstructions look physically reasonable?

Together, these analyses validate whether the learned joint latent space provides a consistent, resolution-agnostic representation of PDE solutions suitable for downstream tasks such as interpolation, comparison, and transfer learning.

```{keypoints}
- REE below 0.01 (1%) indicates good same-modality reconstruction; below 0.05 (5%) for cross-modal
- Procrustes REE improvement indicates rotational differences (expected; alignment corrects these)
- Error should increase monotonically with grid resolution (higher resolution = harder to compress)
- Train-test gap within 2x is normal; larger gaps indicate overfitting
- Visual inspection catches artifacts (smoothing, checkerboards) that REE numbers may not reveal
- Always combine quantitative metrics with qualitative visual checks
```
