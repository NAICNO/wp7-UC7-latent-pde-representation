# Running Experiments

```{objectives}
- Run the full experimental pipeline from data generation through evaluation
- Understand the purpose and output of each pipeline stage
- Use the Jupyter notebook for interactive exploration
- Monitor training progress and identify convergence
- Run experiments with different configurations
```

```{admonition} Before you start
:class: note

Ensure your environment is set up and activated (see [Episode 03](03-setup-environment.md)):

    cd ~/latent-representation-of-pde-solutions
    source venv/bin/activate
```

## Pipeline Overview

The experimental pipeline consists of six stages that must be run in order. Each stage reads artifacts from previous stages and writes its own outputs to disk.

```
Stage 1: Data Generation      --> data/*.npy
Stage 2: Dataset Splitting     --> data/splits/*.npy
Stage 3: Autoencoder Training  --> models/*.keras, latents/lat_1_*.npy
Stage 4: Latent Alignment      --> models/align_*, latents/lat_2_*.npy
Stage 5: Fine-Tuning           --> models/ft_*, latents/enc_3_*, dec_3_*
Stage 6: Evaluation            --> results/*.csv, results/*.png
```

## Stage 1: Data Generation

```bash
python src/cd2d_streamfunc.py
```

This script:

1. **Samples random streamfunction coefficients** from a specified distribution
2. **Constructs divergence-free velocity fields** from each streamfunction using analytical derivatives
3. **Assembles and solves the convection-diffusion PDE** on multiple grid resolutions (16x16 through 256x256) using finite elements and the `pypardiso` sparse solver
4. **Saves all outputs** as NumPy `.npy` files to `data/`

Output files include solution fields at each resolution (`u_16.npy`, `u_32.npy`, ..., `u_256.npy`) and streamfunction coefficient vectors (`streamfunc.npy`).

```{admonition} Expected runtime
:class: tip

Data generation is the most computationally intensive stage on CPU. For the default configuration (1000 samples, 5 resolutions), expect:
- **With pypardiso + MKL**: 5-15 minutes
- **With scipy.sparse.linalg.spsolve**: 30-90 minutes

The 256x256 grid is the bottleneck. If running on limited hardware, consider reducing the number of samples or the maximum resolution.
```

## Stage 2: Dataset Splitting

```bash
python src/create_splits.py
```

Creates a single global train/validation/test split:

- Generates random index arrays for each split
- Saves split indices as `.npy` files
- All downstream scripts load these same indices to ensure consistency

```{admonition} Why a global split?
:class: note

Using the same split across all modalities and experiments guarantees that:
- The same PDE instance is always in the same split (train/val/test) regardless of modality
- Evaluation metrics are directly comparable across experiments
- There is no information leakage between splits
```

## Stage 3: Autoencoder Training

### Solution Field Autoencoders

```bash
python src/train_solution_autoencoder.py
```

Trains one convolutional autoencoder per grid resolution. For each resolution:

1. Loads solution field data and split indices
2. Builds a multi-scale convolutional autoencoder (encoder + decoder)
3. Trains using REE loss with validation monitoring
4. Saves the trained model (`.keras`) and extracted latent vectors (`.npy`)

**What to look for during training:**

- **Training REE** should decrease steadily
- **Validation REE** should track training REE without diverging (no overfitting)
- **Final REE** values below 0.01 (1%) indicate good reconstruction quality

### Streamfunction Autoencoder

```bash
python src/train_streamfunction_autoencoder.py
```

Trains an MLP autoencoder for streamfunction coefficient vectors:

1. Loads streamfunction data and split indices
2. Builds a fully connected autoencoder with latent whitening
3. Trains using reconstruction loss plus whitening penalty
4. Saves model and latent vectors

**What to look for:**

- Latent whitening encourages isotropic latent distributions
- Coefficient vectors are low-dimensional to begin with, so reconstruction should be very accurate

## Stage 4: Latent Alignment

```bash
python src/align_latent_spaces.py
```

Aligns all modality-specific latent spaces into a shared representation:

1. Loads Level 1 latent vectors from all modalities
2. L2-normalizes each modality's latents onto the unit hypersphere
3. Computes the joint latent as the normalized mean across modalities
4. Trains second-level alignment networks with gradual target shifting
5. Saves aligned latent vectors (`lat_2_*.npy`)

**What to look for:**

- **Pairwise REE** between modalities should decrease as alignment progresses
- **REE to mean** should converge to similar values across modalities (no outlier modality)

### Alignment Analysis (Optional)

```bash
python src/analyze_latent_alignment.py
```

Computes detailed alignment diagnostics:

- Pairwise REE matrices (raw, centered, Procrustes-aligned)
- Per-modality deviation from the canonical mean latent
- Saves the canonical joint latent (`lat_3_ld32.npy`) and CSV metrics

## Stage 5: Fine-Tuning

```bash
# Fine-tune encoders: raw input --> joint latent
python src/finetune_encoder_to_latent.py

# Fine-tune decoders: joint latent --> modality output
python src/finetune_decoder_from_latent.py
```

End-to-end fine-tuning corrects accumulated approximation errors from the two-level procedure. Each script:

1. Loads the pre-trained encoder/decoder and alignment networks
2. Connects them into a single differentiable chain
3. Fine-tunes with a small learning rate to preserve the aligned structure
4. Saves fine-tuned models and updated latent/reconstruction vectors

## Stage 6: Evaluation

### End-to-End Decoder Evaluation

```bash
python src/evaluate_decoder_end_to_end.py
```

The primary evaluation script:

1. For each modality pair (source, target), encodes the source modality and decodes into the target
2. Computes REE between the decoded output and the original target data
3. Reports results per modality, per split (train/val/test)
4. Saves summary tables to `results/`

### Error Computation

```bash
python src/compute_errors.py
```

Computes detailed encoding and decoding errors:

- Encoding REE: how consistently do different encoders map to the joint latent?
- Decoding REE: how faithfully does each decoder reconstruct from the joint latent?

### Visualization

```bash
# Visualize raw PDE solutions at multiple resolutions
python src/plot_solutions.py

# Cross-modal reconstruction plots
python src/plot_modalities.py
```

## Interactive Exploration with Jupyter

For interactive, step-by-step exploration:

```bash
# Activate environment
cd ~/latent-representation-of-pde-solutions
source venv/bin/activate

# Start Jupyter Lab
jupyter lab --no-browser --ip=0.0.0.0 --port=8888

# Open demonstrator_pde.ipynb
```

The notebook contains numbered sections corresponding to each pipeline stage, with inline visualization and parameter widgets for interactive experimentation.

```{admonition} Notebook vs. CLI scripts
:class: tip

Both approaches run the same underlying code:
- **Notebook** (`demonstrator_pde.ipynb`): Best for learning and exploration. See results inline, adjust parameters interactively.
- **CLI scripts** (`src/*.py`): Best for reproducible runs and batch experiments. Easier to run in `tmux` or submit to a job scheduler.
```

## Running Custom Experiments

To experiment with different configurations, the key parameters to vary include:

| Parameter | Where to Change | Effect |
|-----------|----------------|--------|
| Number of PDE samples | `cd2d_streamfunc.py` | More data = better generalization, longer training |
| Grid resolutions | `cd2d_streamfunc.py` | Which resolutions to include as modalities |
| Latent dimension | `train_solution_autoencoder.py` | Higher = more capacity, harder to align |
| Autoencoder depth | `train_solution_autoencoder.py` | Deeper = more expressive, risk of overfitting |
| Alignment schedule | `align_latent_spaces.py` | How quickly targets shift from self to joint latent |

## Monitoring Long Runs

For experiments that take a long time, use `tmux` to keep the process running after SSH disconnect:

```bash
tmux new -s experiment
cd ~/latent-representation-of-pde-solutions
source venv/bin/activate
python src/train_solution_autoencoder.py 2>&1 | tee training.log
# Detach: Ctrl+B, then D

# Reattach later:
tmux attach -t experiment

# Monitor from another terminal:
tail -f training.log
```

```{keypoints}
- The pipeline has six stages that must be run in order: data, splits, autoencoders, alignment, fine-tuning, evaluation
- Each stage reads artifacts from previous stages and writes its own outputs to disk
- Training REE should decrease steadily; validation REE should track it without diverging
- The Jupyter notebook provides interactive exploration; CLI scripts provide reproducibility
- Use tmux for long-running experiments on remote VMs
- Key parameters to vary: number of samples, latent dimension, alignment schedule
```
