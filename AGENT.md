# AGENT.md — AI Agent Setup Instructions for UC7

## Quick Start

```bash
# 1. SSH into the VM
ssh <user>@<vm-host>

# 2. VM initialisation (first time only)
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-venv git

# 3. Clone the repository
git clone https://github.com/NAICNO/wp7-UC7-latent-pde-representation.git
cd latent-representation-of-pde-solutions

# 4. Create virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install tensorflow numpy scipy matplotlib jupyter
```

## Project Overview

UC7 — **Latent Representation of PDE Solutions** — trains autoencoders to
learn compact latent representations of Partial Differential Equation (PDE)
solution fields. The project uses **TensorFlow** (not PyTorch) as the deep
learning framework.

Key goals:
- Generate PDE solution data (convection-diffusion 2D problems)
- Train autoencoders on solution fields and streamfunction fields
- Align latent spaces between different representations
- Evaluate end-to-end decoder pipelines

## Entry Points

| Task | Command |
|------|---------|
| **Interactive demo** | Open `demonstrator_pde.ipynb` in Jupyter |
| **Data generation** | `python src/cd2d_streamfunc.py` |
| **Train solution AE** | `python src/train_solution_autoencoder.py` |
| **Align latent spaces** | `python src/align_latent_spaces.py` |
| **End-to-end eval** | `python src/evaluate_decoder_end_to_end.py` |

### Typical workflow

```bash
# Step 1 — Generate PDE solution data
python src/cd2d_streamfunc.py

# Step 2 — Train autoencoders
python src/train_solution_autoencoder.py

# Step 3 — Align latent spaces
python src/align_latent_spaces.py

# Step 4 — Evaluate end-to-end
python src/evaluate_decoder_end_to_end.py
```

## Directory Layout

```
data/                     # Generated PDE solution data (.npy files)
models/                   # Saved trained models (.keras files)
results/                  # Training outputs, plots, metrics
src/
  cd2d_streamfunc.py                  # Data generation: convection-diffusion 2D with streamfunction
  create_splits.py                    # Train/validation/test split creation
  train_solution_autoencoder.py       # Autoencoder for solution fields (conv-AE)
  train_streamfunction_autoencoder.py # Autoencoder for streamfunction coefficients (MLP-AE)
  align_latent_spaces.py              # Latent space alignment across modalities
  analyze_latent_alignment.py         # Alignment quality diagnostics
  finetune_encoder_to_latent.py       # Encoder fine-tuning toward joint latent
  finetune_decoder_from_latent.py     # Decoder fine-tuning from joint latent
  evaluate_decoder_end_to_end.py      # End-to-end evaluation (REE metrics)
  compute_errors.py                   # Error metrics and diagnostics
  plot_solutions.py                   # Solution field visualization
  plot_modalities.py                  # Cross-modal reconstruction visualization
demonstrator_pde.ipynb     # Main interactive notebook / demonstrator
content/                   # Sphinx tutorial content
tests/                     # Unit and integration tests
```

## Data and Models

- **Data directory:** `data/` — stores generated `.npy` arrays (solution fields,
  streamfunction fields, mesh coordinates)
- **Models directory:** `models/` — stores trained model checkpoints (`.keras`
  format)
- Data files are **not** committed to the repository; generate them with
  `python src/cd2d_streamfunc.py`

## Jupyter Tunnel

To run the notebook on a remote VM:

```bash
# On the VM
source venv/bin/activate
jupyter notebook --no-browser --port=8888

# On your local machine (separate terminal)
ssh -N -L 8888:localhost:8888 <user>@<vm-host>
```

Then open http://localhost:8888 in your browser.

## Verification Steps

Run these commands to verify the environment is correctly set up:

```bash
# Check Python version (3.8+ required)
python3 --version

# Verify TensorFlow is importable
python3 -c "import tensorflow as tf; print('TensorFlow', tf.__version__)"

# Verify NumPy and SciPy
python3 -c "import numpy; import scipy; print('NumPy', numpy.__version__, 'SciPy', scipy.__version__)"

# Check generated data exists (after running data generation)
ls data/*.npy

# Check saved models (after training)
ls models/*.keras
```

## GPU Verification

```bash
# Check if GPU is available
python3 -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

If no GPU is detected, training will fall back to CPU (slower but functional).

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `ModuleNotFoundError: No module named 'tensorflow'` | TensorFlow not installed | `pip install tensorflow` |
| `No module named 'numpy'` | Missing dependency | `pip install numpy scipy matplotlib` |
| `ls: cannot access 'data/*.npy'` | Data not generated yet | Run `python src/cd2d_streamfunc.py` first |
| `OOM (Out of Memory)` on GPU | Batch size too large | Reduce `--batch` argument or `--n-sol` |
| Jupyter kernel dies | Insufficient RAM | Reduce `n_sol` or use a larger VM |
| `Permission denied` on SSH | Key not configured | Add SSH key to VM or use password auth |
| `Could not load dynamic library 'libcudart.so'` | CUDA not installed | Install CUDA toolkit or use CPU-only TF |
| Slow training on CPU | No GPU available | Install GPU drivers + `tensorflow[and-cuda]` |
| Port 8888 already in use | Another Jupyter instance | Kill it or use `--port=8889` |

## Framework Notes

- This project uses **TensorFlow / Keras** — not PyTorch
- Models are saved in `.keras` format (Keras 3 native format)
- The main notebook `demonstrator_pde.ipynb` contains the full interactive pipeline
- All CLI scripts are in `src/` and can be run independently
