# Machine Learning Methodology for PDE Representation

```{objectives}
- Understand how PDE data is represented as multiple modalities and learned with ML
- Learn the two-level autoencoder architecture: modality-specific encoding followed by latent alignment
- Understand why Relative Energy Error (REE) is used instead of standard MSE
- Know how models are trained, aligned, fine-tuned, and evaluated end-to-end
- Appreciate the cross-modal transfer capability enabled by the shared latent space
```

```{admonition} Key insight: representation learning, not prediction
:class: important

This project does **not** train a model to predict PDE solutions from parameters (that would be a neural operator / surrogate model). Instead, it learns a **shared geometric structure** across modalities -- a latent space where different views of the same PDE state are mapped to nearby points. This distinction is fundamental to understanding the methodology.
```

## Methodology Overview

The machine learning methodology is designed to learn compact, **resolution-invariant** representations of PDE solutions. The core idea is to treat different numerical realizations of the same PDE state as multiple **modalities** and to learn a shared latent representation that captures their common structure.

The approach proceeds in three levels:

1. **Level 1**: Train independent autoencoders per modality (compression)
2. **Level 2**: Align the resulting latent spaces on a shared hypersphere (alignment)
3. **Level 3**: Fine-tune end-to-end encoder-decoder chains (refinement)

## Data Representation

### Modalities

Each PDE sample is represented through several modalities, all index-aligned:

| Modality | Shape | Description |
|----------|-------|-------------|
| `u_16` | (N, 16, 16) | Solution field on 16x16 grid |
| `u_32` | (N, 32, 32) | Solution field on 32x32 grid |
| `u_64` | (N, 64, 64) | Solution field on 64x64 grid |
| `u_128` | (N, 128, 128) | Solution field on 128x128 grid |
| `u_256` | (N, 256, 256) | Solution field on 256x256 grid |
| `streamfunc` | (N, K) | Streamfunction coefficient vector |

**Index alignment** is critical: sample *i* in every file corresponds to the same underlying PDE state (same velocity field, same boundary conditions, same physical solution observed at different resolutions).

### Dataset Splitting

A single, global split is created by `create_splits.py` and reused for **all** experiments:

- **Training set**: Used for gradient-based optimization
- **Validation set**: Used for early stopping and hyperparameter selection
- **Test set**: Used for final evaluation only

This guarantees consistent evaluation across modalities, prevents information leakage, and ensures reproducibility. The split indices are saved as `.npy` files and loaded by every downstream script.

## Level 1: Modality-Specific Autoencoders

### Architecture

Two distinct architectures are used, matched to the structure of each modality:

**Convolutional autoencoders** (for grid-based solution fields):

- Encoder: series of convolutional layers with spatial downsampling (strided convolutions)
- Bottleneck: flattened to a dense latent vector of dimension *d* (typically 32)
- Decoder: transposed convolutions with spatial upsampling back to original grid size
- One autoencoder is trained per grid resolution (16x16, 32x32, ..., 256x256)

**MLP autoencoders** (for streamfunction coefficients):

- Encoder: fully connected layers with decreasing width
- Bottleneck: dense latent vector of dimension *d* (same as solution autoencoders)
- Decoder: fully connected layers with increasing width
- Latent whitening regularization encourages isotropic, well-conditioned latent distributions

### Loss Function: Relative Energy Error (REE)

Standard MSE penalizes large-magnitude solutions disproportionately. Instead, the project uses **Relative Energy Error**:

```
REE(x, x_hat) = || x - x_hat ||^2 / || x ||^2
```

This is scale-invariant: a 1% reconstruction error is treated equally whether the solution has magnitude 0.01 or 100. REE is used for both training loss and evaluation metrics throughout the project.

```{admonition} Why not standard MSE?
:class: tip

PDE solutions can vary enormously in magnitude depending on the diffusion coefficient and velocity field parameters. A fixed MSE threshold would be meaningless: 0.001 MSE might be excellent for a large-amplitude solution but catastrophic for a small one. REE normalizes by the solution energy, making error comparisons meaningful across the parameter space.
```

### Training Details

- Each modality autoencoder is trained independently
- Latent dimension is fixed across all modalities (typically *d* = 32)
- Validation REE is monitored for early stopping
- Trained encoders and decoders are saved separately for later use in the alignment pipeline

## Level 2: Latent-Space Alignment

### Motivation

After Level 1, each modality has its own encoder producing 32-dimensional latent vectors. However, these latent spaces are **not comparable**: the encoder for the 32x32 grid and the encoder for the 128x128 grid have no reason to produce similar latent vectors for the same PDE sample. The second level corrects this.

### Alignment Strategy

The alignment procedure works as follows:

1. **Project onto the unit hypersphere**: Each modality's latent vectors are L2-normalized, placing them on the surface of the unit sphere in R^d
2. **Define a joint latent**: For each sample, compute the normalized mean of all modality latents as the "consensus" representation
3. **Gradual target shifting**: During training, the reconstruction target smoothly transitions from each modality's own latent (self-reconstruction) to the joint latent (cross-modal agreement)

This enforces three properties simultaneously:

- **Geometric consistency**: All modalities share the same latent geometry (the hypersphere)
- **Cross-modal agreement**: Encoding any modality of sample *i* produces a similar latent vector
- **Information preservation**: Each modality can still be reconstructed from its aligned latent

### Why the Unit Hypersphere?

Normalizing latents onto the sphere prevents **magnitude collapse** (where the model shrinks latents to near-zero to minimize alignment loss) and provides a natural distance metric (cosine similarity). It also makes the alignment loss independent of the overall scale of latent activations.

## Level 3: End-to-End Fine-Tuning

After alignment, the full encode-decode chains can be fine-tuned:

- **Encoder fine-tuning** (`finetune_encoder_to_latent.py`): Trains the path from raw modality input directly to the joint latent, bypassing the intermediate Level 1 latent
- **Decoder fine-tuning** (`finetune_decoder_from_latent.py`): Trains the path from the joint latent directly to the modality output

Fine-tuning corrects accumulated approximation errors from the two-level procedure and improves:

- Numerical stability of the encode-decode chain
- Cross-modal reconstruction accuracy (encode one modality, decode another)
- Direct usability of the joint latent space for downstream tasks

## Cross-Modal Transfer

The aligned latent space enables **cross-modal transfer**: encode a sample using one modality's encoder, then decode it using a *different* modality's decoder. For example:

- Encode a 32x32 solution field
- Decode into a 256x256 solution field (super-resolution)
- Decode into streamfunction coefficients (inverse problem)

This is the central capability that the project demonstrates and evaluates.

## Evaluation Methodology

### Quantitative Metrics

**Relative Energy Error (REE)** is computed for:

- **Encoding accuracy**: REE between the encoded sample and the joint latent target
- **Decoding accuracy**: REE between the decoded reconstruction and the original modality data
- **Cross-modal accuracy**: REE when encoding one modality and decoding another

All metrics are reported separately for train, validation, and test splits.

### Qualitative Evaluation

- **Cross-modal reconstructions**: Encode one modality, decode into all others, visualize side-by-side
- **Visual inspection**: Check for systematic artifacts, resolution-dependent smoothing, or modality-specific biases

### Alignment Diagnostics

Additional analyses from `analyze_latent_alignment.py`:

- Pairwise latent distance matrices between modalities
- REE after centering (mean subtraction) to isolate rotational differences
- REE after optimal Procrustes alignment to measure best-case agreement
- Deviation of each modality from the canonical mean latent

## Experimental Philosophy

This methodology is intentionally **modular**, **explicit**, and **diagnostic**:

- Each stage can be run, inspected, and debugged independently
- All intermediate artifacts (latent vectors, metrics, plots) are saved to disk
- The emphasis is on understanding what the model learns, not on optimizing a leaderboard metric

The pipeline is designed so that researchers can modify one stage (e.g., swap the autoencoder architecture) without disrupting the rest of the pipeline.

```{keypoints}
- The ML approach is representation learning (shared latent space), not prediction (surrogate model)
- Each grid resolution is treated as a separate modality with its own autoencoder
- Relative Energy Error (REE) provides scale-invariant loss and evaluation
- A second-level alignment maps all modality latents onto a shared unit hypersphere
- End-to-end fine-tuning corrects accumulated errors from the two-level approach
- Cross-modal transfer (encode one modality, decode another) is the central demonstrated capability
- The pipeline is modular: each stage is independent and produces inspectable artifacts
```
