# Setting Up the Environment

```{objectives}
- Initialize a fresh VM with required system packages
- Clone the repository and set up the Python environment
- Verify TensorFlow installation and GPU detection
- Confirm pypardiso (sparse solver) is functional
- Start Jupyter Lab and create an SSH tunnel for remote access
```

```{admonition} Prerequisites
:class: note

This episode assumes you have SSH access to a VM (see [Episode 02](02-provision-vm.md)) or a local machine with Python 3.8+. The instructions are written for Ubuntu-based systems but work on most Linux distributions with minor adjustments.
```

## 1. Connect to Your VM

```bash
ssh -i /path/to/your-key.pem ubuntu@<VM_IP>
```

## 2. Initialize Fresh VM (Run Once)

On a fresh NAIC VM, install the required system packages:

```bash
sudo apt update -y
sudo apt install -y build-essential git python3-dev python3-venv python3-pip libssl-dev zlib1g-dev
```

This installs:

| Package | Purpose |
|---------|---------|
| `build-essential` | Compiler toolchain (gcc, make) for building Python extensions |
| `git` | Repository cloning |
| `python3-dev` | Python C headers for compiled extensions |
| `python3-venv` | Virtual environment support |
| `python3-pip` | Python package installer |
| `libssl-dev`, `zlib1g-dev` | Required for building certain Python packages from source |

## 3. Clone and Setup

```bash
git clone https://github.com/NAICNO/wp7-UC7-latent-pde-representation.git
cd latent-representation-of-pde-solutions
./setup.sh
source venv/bin/activate
```

The `setup.sh` script automatically:
- Detects the NAIC module system (if available) and loads an appropriate Python version
- Checks that Python 3.8+ is available
- Detects NVIDIA GPU and sets up CUDA library symlinks if needed
- Creates a Python virtual environment in `venv/`
- Installs all dependencies from `requirements.txt`

## 4. Verify TensorFlow and GPU

After setup completes, verify that TensorFlow is correctly installed and can detect available hardware:

```bash
# Check TensorFlow version and GPU availability
python3 -c "
import tensorflow as tf
print('TensorFlow version:', tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f'GPU detected: {len(gpus)} device(s)')
    for gpu in gpus:
        print(f'  - {gpu.name}')
else:
    print('No GPU detected -- training will use CPU (slower but functional)')
"
```

```{admonition} GPU is recommended but not required
:class: tip

All scripts in this project run correctly on CPU. GPU acceleration significantly speeds up autoencoder training (typically 5-10x), but the dataset sizes are modest enough that CPU training completes within reasonable time (minutes to hours depending on grid resolution).

If you see warnings about missing CUDA libraries, these can be safely ignored when running in CPU mode.
```

## 5. Verify pypardiso

The `pypardiso` package provides the sparse direct solver used for the finite element PDE assembly. It depends on Intel MKL:

```bash
python3 -c "
from pypardiso import spsolve
import scipy.sparse as sp
import numpy as np

# Quick test: solve a small sparse system
A = sp.eye(5, format='csr') * 2.0
b = np.ones(5)
x = spsolve(A, b)
print('pypardiso OK -- test solution:', x)
"
```

```{admonition} pypardiso installation issues
:class: warning

If `pypardiso` fails to install or import, the most common cause is a missing Intel MKL library. On Ubuntu, try:

    pip install mkl pypardiso

On systems where MKL is not available (e.g., ARM-based machines), you can modify the PDE solver scripts to use `scipy.sparse.linalg.spsolve` instead, though this is significantly slower for large grids.
```

## 6. Start Jupyter Lab (Optional)

For interactive exploration using the demonstrator notebook:

```bash
# Use tmux for persistence (session survives SSH disconnects)
tmux new -s jupyter
cd ~/latent-representation-of-pde-solutions
source venv/bin/activate
jupyter lab --no-browser --ip=0.0.0.0 --port=8888 --NotebookApp.token='' --NotebookApp.password=''
# Detach with Ctrl+B, then D
```

## 7. Create SSH Tunnel (On Your Local Machine)

To access Jupyter Lab from your local browser, create an SSH tunnel:

```bash
# Verbose mode (recommended -- shows connection status)
ssh -v -N -L 8888:localhost:8888 -i /path/to/your-key.pem ubuntu@<VM_IP>
```

> **Note:** The tunnel will appear to "hang" after connecting -- this is normal. It means the tunnel is active. Keep the terminal open while using Jupyter.

**If port 8888 is already in use**, use an alternative port:

```bash
ssh -v -N -L 9999:localhost:8888 -i /path/to/your-key.pem ubuntu@<VM_IP>
# Then access via http://localhost:9999
```

Then navigate to: **http://localhost:8888/lab/tree/demonstrator_pde.ipynb**

To close the tunnel, press `Ctrl+C` in the terminal.

## Project Structure

After cloning, you will have:

```
latent-representation-of-pde-solutions/
├── src/                                    # Core ML pipeline (12 modules)
│   ├── cd2d_streamfunc.py                  # Streamfunction-based data generation
│   ├── create_splits.py                    # Train/validation/test splits
│   ├── train_solution_autoencoder.py       # Autoencoder for solution fields (u)
│   ├── train_streamfunction_autoencoder.py # Autoencoder for streamfunction modality
│   ├── align_latent_spaces.py              # Latent space alignment
│   ├── analyze_latent_alignment.py         # Alignment quality diagnostics
│   ├── finetune_encoder_to_latent.py       # Encoder fine-tuning toward shared latent
│   ├── finetune_decoder_from_latent.py     # Decoder fine-tuning from shared latent
│   ├── evaluate_decoder_end_to_end.py      # End-to-end decoder evaluation (REE)
│   ├── compute_errors.py                   # Error metrics
│   ├── plot_solutions.py                   # Solution visualization
│   └── plot_modalities.py                  # Cross-modality visualization
├── demonstrator_pde.ipynb                  # Interactive Jupyter notebook
├── tests/                                  # Unit and integration tests
├── content/                                # Sphinx tutorial documentation
├── results/                                # Training outputs and plots
├── setup.sh                                # Automated environment setup
├── vm-init.sh                              # VM initialization script
├── requirements.txt                        # Python dependencies
└── README.md                               # Project overview
```

## Dependencies

The following packages are installed via `setup.sh` (from `requirements.txt`):

| Package | Purpose |
|---------|---------|
| `numpy` | Array operations and data storage (.npy format) |
| `scipy` | Scientific computing, sparse matrix operations |
| `pandas` | Data manipulation and CSV output |
| `tensorflow[and-cuda]` | Deep learning framework with optional CUDA support |
| `pypardiso` | Intel MKL-based sparse direct solver for FEM assembly |
| `matplotlib` | Visualization of solutions, latents, and metrics |
| `jupyterlab` | Interactive notebook environment |
| `ipywidgets` | Interactive widgets for the demonstrator notebook |

## Running Tests

To verify the installation is fully functional, run the test suite:

```bash
cd ~/latent-representation-of-pde-solutions
source venv/bin/activate
python -m pytest tests/ -v
```

All tests should pass. If any fail, check the error messages for missing dependencies or environment issues.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `git: command not found` | Run VM init: `sudo apt install -y git` |
| `python3: command not found` | Install Python: `sudo apt install -y python3 python3-venv` |
| Connection refused | Verify VM is running with `ping <VM_IP>` |
| Permission denied (SSH) | `chmod 600 /path/to/your-key.pem` |
| Host key error | `ssh-keygen -R <VM_IP>` (VM IP changed after reprovisioning) |
| Jupyter not accessible | Check that the SSH tunnel is running and port is correct |
| Port 8888 already in use | Use alternative port: `-L 9999:localhost:8888` |
| SSH tunnel appears to hang | This is normal -- the tunnel is active; keep the terminal open |
| Import errors | Verify venv is activated: `which python` should show `venv/bin/python` |
| `pypardiso` import error | `pip install mkl pypardiso` or check Intel MKL availability |
| TensorFlow GPU warnings | Safe to ignore in CPU mode; install CUDA toolkit for GPU support |

```{keypoints}
- Initialize fresh VMs with `sudo apt install -y build-essential git python3-dev python3-venv`
- Clone the repository and run `./setup.sh` to automatically set up the Python environment
- Verify TensorFlow GPU detection with `tf.config.list_physical_devices('GPU')`
- Verify pypardiso with a small test solve to ensure the sparse solver works
- Use tmux for persistent Jupyter Lab sessions that survive SSH disconnects
- Create an SSH tunnel to access Jupyter Lab from your local browser
- Run `pytest tests/ -v` to verify the full installation
```
