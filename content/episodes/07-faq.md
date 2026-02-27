# FAQ and Troubleshooting

```{objectives}
- Find solutions to common installation, training, and evaluation issues
- Understand the design decisions behind key project choices
- Know where to look when results are unexpected
- Get guidance on extending the project for new experiments
```

```{admonition} How to use this page
:class: tip

This episode is organized into four sections: **Installation and Setup**, **Data and PDE**, **Training and Models**, and **Results and Evaluation**. Use `Ctrl+F` to search for specific error messages or keywords.
```

## Installation and Setup

### Q: TensorFlow fails to install or import

**Symptoms:** `pip install tensorflow` fails, or `import tensorflow` raises errors about missing shared libraries.

**Solutions:**

- Ensure Python 3.8+ is installed: `python3 --version`
- On NAIC VMs, run `./setup.sh` which handles version detection
- If you see `libcudart.so` errors, these are GPU-related and can be ignored for CPU-only use
- For ARM-based machines (e.g., Apple Silicon), install `tensorflow-macos` instead

### Q: pypardiso will not install

**Symptoms:** `pip install pypardiso` fails with compiler errors or missing MKL libraries.

**Solutions:**

- Install Intel MKL first: `pip install mkl`
- On Ubuntu/Debian: `sudo apt install -y libmkl-dev`
- On systems without MKL (ARM, some HPC nodes), pypardiso cannot be used. Modify the solver to use `scipy.sparse.linalg.spsolve` as a fallback
- If using conda: `conda install -c conda-forge pypardiso`

### Q: GPU is not detected by TensorFlow

**Symptoms:** `tf.config.list_physical_devices('GPU')` returns an empty list.

**Solutions:**

1. Verify the GPU is visible to the OS: `nvidia-smi`
2. Check CUDA toolkit: `nvcc --version`
3. Run `./setup.sh` which creates CUDA library symlinks
4. Ensure `tensorflow[and-cuda]` is installed (not plain `tensorflow`)
5. GPU is recommended but not required -- all experiments run on CPU

### Q: Jupyter Lab is not accessible from my browser

**Symptoms:** Opening `http://localhost:8888` shows "connection refused" or times out.

**Solutions:**

- Verify Jupyter is running on the VM: `ps aux | grep jupyter`
- Check the SSH tunnel is active on your local machine
- Ensure port 8888 is not already in use (try port 9999 instead)
- If using a VPN, the tunnel may not route correctly -- try disabling it
- Start Jupyter with explicit IP binding: `--ip=127.0.0.1`

### Q: SSH connection is refused or times out

**Solutions:**

- Verify the VM is running at [orchestrator.naic.no](https://orchestrator.naic.no)
- Check that your current IP is whitelisted in the Orchestrator
- Set correct key permissions: `chmod 600 /path/to/key.pem`
- If the VM was reprovisioned: `ssh-keygen -R <old_ip>`

---

## Data and PDE

### Q: What PDE is being solved?

The project solves for a scalar field $u$ (like temperature or concentration) in the steady-state equation on $[-1, 1]^2$:

$$\nabla \cdot (\mathbf{v}u) - \nabla \cdot (D \nabla u) = 0$$

*   **Diffusion ($D$):** Kept constant at 1.0.
*   **Advection ($\mathbf{v}$):** The velocity field is derived from a **Streamfunction** $\psi$, in a way that makes it divergence-free.

Dirichlet boundary conditions are applied. The streamfunction is parameterized by a small number of coefficients (typically 10-20), making it a natural low-dimensional representation of the PDE family.

### Q: Why divergence-free velocity fields?

Divergence-free fields (div(b) = 0) satisfy mass conservation. Using a streamfunction parameterization guarantees this property by construction: if b = curl(psi), then div(b) = div(curl(psi)) = 0. This ensures physically meaningful velocity fields without additional constraints.

### Q: Why multiple grid resolutions?

Multiple resolutions serve two purposes:

1. **Multi-resolution consistency testing**: Does the latent representation capture the physics regardless of grid fineness?
2. **Cross-resolution transfer**: Can we encode a coarse solution and decode a fine one (a form of learned super-resolution)?

The resolutions 16x16 through 256x256 span a 256x range in degrees of freedom, from very coarse to moderately fine.

### Q: How are the PDE solutions computed?

Solutions are computed using the finite element method (FEM):

1. Assemble stiffness and convection matrices for the given grid
2. Apply boundary conditions
3. Solve the resulting sparse linear system using `pypardiso` (Intel MKL Pardiso solver)
4. Store the solution as a flat NumPy array, reshaped to the grid dimensions

### Q: I see `data/*.npy` files missing

Run the data generation step first:

```bash
python src/cd2d_streamfunc.py
```

Data files are not committed to the repository because they are large and easily regenerated.

---

## Training and Models

### Q: Why autoencoders instead of neural operators?

This project targets **representation learning**, not surrogate modeling:

- **Neural operators** (DeepONet, FNO) learn the map from parameters to solutions
- **Autoencoders** learn a compressed representation of the solution space itself

The autoencoder approach enables cross-modal transfer, interpolation in latent space, and resolution-invariant embeddings -- capabilities that neural operators do not directly provide.

### Q: What is the typical latent dimension?

The default latent dimension is **32** across all modalities. This was chosen as a balance:

- Large enough to capture the dominant variation in the solution manifold
- Small enough to enable efficient alignment and meaningful geometric analysis
- Same dimension for all modalities (required for direct latent comparison)

### Q: Why Relative Energy Error (REE) instead of MSE?

PDE solutions can vary over many orders of magnitude depending on the parameters. Standard MSE would be dominated by high-amplitude solutions and insensitive to errors in low-amplitude ones. REE normalizes by the solution energy, making the metric scale-invariant:

```
REE = || x - x_hat ||^2 / || x ||^2
```

### Q: Training loss is not decreasing

**Possible causes and solutions:**

- **Learning rate too high**: Reduce by a factor of 10
- **Latent dimension too small**: Try increasing from 32 to 64
- **Data issue**: Verify that data files loaded correctly (check shapes, NaN values)
- **Architecture mismatch**: For high-resolution grids, the convolutional autoencoder may need more layers

### Q: Training loss decreases but validation loss increases

This indicates **overfitting**. Solutions:

- Reduce model capacity (fewer layers, fewer filters)
- Add dropout or regularization
- Increase the training set size (generate more PDE samples)
- Use early stopping based on validation REE

### Q: What is latent whitening?

The streamfunction autoencoder applies **latent whitening regularization**: a penalty term that encourages the latent distribution to have zero mean and identity covariance. This prevents the latent space from collapsing to a low-dimensional subspace and makes alignment easier.

---

## Results and Evaluation

### Q: Cross-modal REE is much higher than same-modal REE

This usually means the **alignment** stage did not converge properly. Check:

1. Were the Level 1 autoencoders well-trained? (same-modal REE should be low)
2. Did the alignment loss decrease during training?
3. Try running alignment for more epochs
4. Check that all modality latent vectors have similar magnitude distributions (L2 norms)

### Q: One modality has dramatically higher error than others

Common causes:

- **Resolution too high**: The 256x256 autoencoder may need a deeper architecture
- **Insufficient training**: That modality's autoencoder did not converge
- **Data issue**: Check for NaN or inf values in that modality's data files
- **Alignment outlier**: The modality's latent space may have a fundamentally different structure

### Q: Test error is much higher than training error

This indicates overfitting. The most likely culprit:

- **Small dataset**: Generate more PDE samples
- **Model too large**: Reduce the number of parameters
- **No early stopping**: Add validation-based early stopping

### Q: Reconstructed solutions show checkerboard artifacts

This is a known issue with transposed convolutions in the decoder. Solutions:

- Replace transposed convolutions with upsampling + convolution (resize-convolution)
- Add a smoothing layer after the decoder output
- Reduce the decoder's stride to 1 and use explicit upsampling

### Q: Reconstructed solutions are too smooth

The autoencoder may be losing high-frequency information. Solutions:

- Increase the latent dimension (try 64 or 128)
- Add skip connections (U-Net style) to the autoencoder
- Use a perceptual or gradient-based loss in addition to REE

### Q: How do I visualize the latent space?

The latent space is 32-dimensional, so direct visualization requires projection:

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# lat_2_u32 and lat_2_u128 are (N, 32) arrays
pca = PCA(n_components=2)
z_all = np.vstack([lat_2_u32, lat_2_u128])
z_proj = pca.fit_transform(z_all)

plt.scatter(z_proj[:N, 0], z_proj[:N, 1], label='u_32', alpha=0.5)
plt.scatter(z_proj[N:, 0], z_proj[N:, 1], label='u_128', alpha=0.5)
plt.legend()
plt.title('PCA of aligned latent vectors')
plt.show()
```

If the modalities form separate clusters, alignment has failed. If they intermingle, alignment is working.

---

## Extending the Project

### Q: Can I add a new grid resolution?

Yes. Modify `cd2d_streamfunc.py` to include the new resolution, then retrain all downstream stages. The pipeline is designed to be modular -- adding a resolution means adding a new modality.

### Q: Can I use a different PDE?

The autoencoder and alignment pipeline is PDE-agnostic. To use a different PDE:

1. Replace the data generation script with one that solves your PDE
2. Ensure output files follow the same format (`.npy`, indexed consistently)
3. Retrain all autoencoders and alignment from scratch

### Q: Can I use PyTorch instead of TensorFlow?

The project currently uses TensorFlow/Keras. Porting to PyTorch would require rewriting the model definitions and training loops, but the data pipeline, evaluation metrics, and analysis scripts are framework-independent.

```{keypoints}
- Most installation issues stem from TensorFlow or pypardiso dependencies -- check Python version and Intel MKL availability
- Data files are not committed to the repository; generate them with `cd2d_streamfunc.py`
- High cross-modal REE usually indicates alignment issues, not autoencoder issues
- Overfitting manifests as train-test gap; address with more data, less capacity, or early stopping
- Visual artifacts (smoothing, checkerboards) point to specific architectural changes
- The pipeline is modular: new resolutions, PDEs, or frameworks can be substituted at individual stages
```
