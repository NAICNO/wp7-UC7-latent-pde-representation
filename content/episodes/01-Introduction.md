# Introduction to PDE Representation via Autoencoders and Aligned Latent Spaces

```{objectives}
- Understand the goal and motivation of the PDE representation project
- Learn how numerical experiments are structured and connected through a layered pipeline
- See how autoencoders and latent alignment build a unified, resolution-invariant representation
- Identify the target audience and prerequisites for working with this codebase
```

```{admonition} Why latent representations of PDEs?
:class: tip

Traditional PDE solvers produce one solution at a time for one set of parameters. But what if you need to explore an entire *family* of solutions -- varying boundary conditions, coefficients, or forcing terms? Re-solving from scratch each time is expensive. By learning a compact latent representation of the **solution manifold**, we gain the ability to:

- Interpolate between solutions without re-solving
- Transfer information across different grid resolutions
- Compare and cluster solution families in a shared geometric space
- Enable downstream tasks (optimization, uncertainty quantification) on a low-dimensional representation
```

## Overview

This project is a research-oriented numerical framework for learning compact, structured representations of partial differential equation (PDE) solution manifolds. It focuses on **steady-state 2D convection-diffusion equations** on the spatial domain [-1, 1]^2 and investigates how multiple numerical resolutions and a parameter-based representation can be embedded into a shared latent space using autoencoders.

The convection-diffusion equation takes the form:

```
-epsilon * Laplacian(u) + b(x) . grad(u) = 0
```

where the advective velocity field **b(x)** is divergence-free and parameterized by streamfunction coefficients. This ensures physical consistency (mass conservation) while providing a natural low-dimensional parameterization of the PDE family.

The project is **not** intended as a production-ready PDE solver. Instead, it serves as an environment for:

- Representation learning on scientific data
- Multi-resolution consistency analysis
- Cross-modal latent alignment
- Diagnostic analysis of learned PDE manifolds

## Conceptual Structure

The numerical experiments follow a layered pipeline:

1. **Mathematics-driven data generation** -- Sample streamfunction coefficients, construct divergence-free velocity fields, solve the PDE on grids from 16x16 to 256x256
2. **Modality-specific autoencoding** -- Train independent autoencoders for each grid resolution and the coefficient representation
3. **Latent-space alignment** -- Map all modality latents onto a shared unit hypersphere through a second-level alignment model
4. **Quantitative and qualitative evaluation** -- Measure reconstruction fidelity (REE), cross-modal consistency, and alignment quality

Each stage produces artifacts (data files, trained models, latent vectors, metrics) that are used by subsequent stages. Strict **index alignment** is maintained across all datasets, models, and representations: sample *i* always refers to the same underlying PDE state.

## What Is Being Represented

Different numerical views of the same PDE state are treated as distinct but related **modalities**:

- **Solution fields** at grid resolutions 16x16, 32x32, 64x64, 128x128, and 256x256
- **Streamfunction coefficient vectors** defining the velocity field (low-dimensional)
- **Latent vectors** produced by modality-specific encoders (typically 32-dimensional)
- **Joint latent representation** capturing the underlying PDE solution in a shared geometric space

The central goal is to learn representations that are **resolution-invariant** and **semantically meaningful**: encoding a 32x32 solution and a 256x256 solution of the same PDE instance should produce nearby latent vectors.

## Why Cross-Modal Alignment Matters

The most distinctive aspect of this project is the **cross-modal alignment** step. Without it, each modality's autoencoder lives in its own latent space with arbitrary geometry. The alignment procedure maps all modality latents onto a **shared unit hypersphere**, enabling:

- **Cross-resolution transfer**: Encode a 32x32 solution and decode a 256x256 reconstruction
- **Parameter recovery**: Encode a solution field and decode the streamfunction coefficients
- **Latent arithmetic**: Interpolate between PDE solutions in latent space and decode physically meaningful intermediate states
- **Unified analysis**: Compare samples across modalities using a single distance metric

This is fundamentally different from training a single multi-input network. Each modality retains its own specialized encoder and decoder, but all encoders agree on what the latent representation should look like.

## What This Project Is Not

To set clear expectations:

- This is **not** a neural PDE solver (no time-stepping, no operator learning)
- This is **not** a production tool (no optimized inference, no deployment pipeline)
- Performance benchmarks are diagnostic, not competitive -- the emphasis is on **understanding** the learned representations

## Technology Stack

| Component | Technology |
|-----------|------------|
| Deep learning | TensorFlow / Keras |
| PDE solver | Finite elements with `pypardiso` sparse solver |
| Scientific computing | NumPy, SciPy |
| Visualization | Matplotlib |
| Interactive exploration | JupyterLab with `ipywidgets` |
| Documentation | Sphinx with MyST Markdown |

## Intended Audience

This repository is intended for:

- **Students** in scientific machine learning seeking hands-on experience with representation learning for PDEs
- **Researchers** in numerical PDEs interested in machine-learning-based dimensionality reduction
- **Practitioners** in representation learning looking for structured, mathematically grounded benchmark problems

Familiarity with basic PDE theory (diffusion, convection, finite elements) and neural network fundamentals (autoencoders, loss functions, training loops) is assumed.

## How to Use This Tutorial

The tutorial episodes are designed to be followed sequentially:

| Episode | What You Will Learn |
|---------|-------------------|
| 01 (this page) | Project goals, structure, and motivation |
| 02 | How to provision a VM on the NAIC Orchestrator |
| 03 | Environment setup, dependency installation, and verification |
| 04 | The ML methodology: autoencoders, alignment, and evaluation |
| 05 | Running the full experimental pipeline step by step |
| 06 | Interpreting results: REE metrics, latent visualization, cross-modal checks |
| 07 | Frequently asked questions and troubleshooting |

```{keypoints}
- The project learns PDE solution manifolds, not individual solutions
- Multiple grid resolutions and parameter vectors are treated as separate modalities
- Autoencoders compress each modality; a second-level model aligns them on the unit hypersphere
- The pipeline is modular: data generation, encoding, alignment, and evaluation are independent stages
- The emphasis is on research, diagnostics, and understanding -- not production performance
```
