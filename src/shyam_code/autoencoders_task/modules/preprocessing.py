"""
CIFAR-10 preprocessing pipeline for k-Sparse Autoencoder.
Implements: patch extraction, Local Contrast Normalization (LCN), ZCA Whitening.
Based on Coates et al. 2011, as used in Makhzani & Frey 2014.
"""

import os
import numpy as np
import torchvision.datasets


def load_cifar10(data_dir="./data") -> np.ndarray:
    """Load CIFAR-10 training images.

    Returns:
        (50000, 32, 32, 3) uint8 numpy array
    """
    dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True)
    return dataset.data  # shape: (50000, 32, 32, 3), dtype uint8


def extract_patches(images, n_patches=1_000_000, patch_size=8, seed=42) -> np.ndarray:
    """Extract random patches from images.

    Args:
        images: (N, H, W, C) uint8 array
        n_patches: number of patches to extract
        patch_size: height and width of each patch (square)
        seed: random seed for reproducibility

    Returns:
        (n_patches, patch_size*patch_size*C) float32 array
    """
    rng = np.random.default_rng(seed)
    n_images, h, w, c = images.shape
    max_row = h - patch_size   # 0–24 for 8×8 patches in 32×32 images
    max_col = w - patch_size

    img_indices = rng.integers(0, n_images, size=n_patches)
    row_offsets = rng.integers(0, max_row + 1, size=n_patches)
    col_offsets = rng.integers(0, max_col + 1, size=n_patches)

    patch_dim = patch_size * patch_size * c
    patches = np.empty((n_patches, patch_dim), dtype=np.float32)

    for i in range(n_patches):
        r, co, idx = row_offsets[i], col_offsets[i], img_indices[i]
        patch = images[idx, r:r + patch_size, co:co + patch_size, :]
        patches[i] = patch.ravel()

    return patches


def local_contrast_normalize(patches, epsilon=1e-8) -> np.ndarray:
    """Apply Local Contrast Normalization per patch.

    Per-patch: subtract mean, divide by std (clamped to epsilon).

    Args:
        patches: (N, D) float32 array
        epsilon: minimum std to avoid division by zero

    Returns:
        (N, D) float32 array
    """
    means = patches.mean(axis=1, keepdims=True)
    centered = patches - means
    stds = np.maximum(centered.std(axis=1, keepdims=True), epsilon)
    return (centered / stds).astype(np.float32)


def compute_zca_params(patches_lcn, epsilon=0.1) -> dict:
    """Compute ZCA whitening parameters from LCN patches.

    Args:
        patches_lcn: (N, D) float32 array of LCN patches
        epsilon: regularization for near-zero eigenvalues

    Returns:
        dict with keys: W, mean, U, s, epsilon
    """
    n = patches_lcn.shape[0]

    mean = patches_lcn.mean(axis=0)  # shape (D,)
    X = patches_lcn.astype(np.float64) - mean

    C = X.T @ X / n  # covariance (D, D), uses N not N-1

    # eigh: correct for symmetric PSD matrix, returns real eigenvalues in ascending order
    s, U = np.linalg.eigh(C)

    W = U @ np.diag((s + epsilon) ** -0.5) @ U.T  # (D, D) whitening matrix

    return {
        "W": W,
        "mean": mean,
        "U": U,
        "s": s,
        "epsilon": np.float64(epsilon),
    }


def apply_zca_whitening(patches_lcn, params, chunk_size=100_000) -> np.ndarray:
    """Apply ZCA whitening to LCN patches.

    Processes in chunks to avoid large float64 peak memory allocation.

    Args:
        patches_lcn: (N, D) float32 array
        params: dict from compute_zca_params
        chunk_size: number of patches to process at once

    Returns:
        (N, D) float32 whitened array
    """
    W    = params["W"]
    mean = params["mean"]
    n    = patches_lcn.shape[0]
    out  = np.empty_like(patches_lcn)

    for start in range(0, n, chunk_size):
        end           = min(start + chunk_size, n)
        X             = patches_lcn[start:end].astype(np.float64) - mean
        out[start:end] = (X @ W).astype(np.float32)

    return out


def save_whitening_params(params, path):
    """Save ZCA params to .npz file."""
    np.savez(path, **params)
    print(f"Saved whitening params to {path}")


def load_whitening_params(path) -> dict:
    """Load ZCA params from .npz file."""
    data = np.load(path)
    return dict(data)


def preprocess_pipeline(
    data_dir="./data",
    output_dir="./outputs",
    n_patches=1_000_000,
    patch_size=8,
    seed=42,
    lcn_epsilon=1e-8,
    zca_epsilon=0.1,
):
    """Run full preprocessing pipeline: load → extract → LCN → ZCA → save.

    Returns:
        (patches_white, params): whitened patches and ZCA parameters
    """
    os.makedirs(output_dir, exist_ok=True)

    print("Loading CIFAR-10...")
    images = load_cifar10(data_dir)
    print(f"  Loaded {images.shape[0]} images, shape {images.shape}")

    print(f"Extracting {n_patches:,} patches of size {patch_size}x{patch_size}...")
    patches = extract_patches(images, n_patches=n_patches, patch_size=patch_size, seed=seed)
    print(f"  Patches shape: {patches.shape}, dtype: {patches.dtype}")

    print("Applying Local Contrast Normalization...")
    patches_lcn = local_contrast_normalize(patches, epsilon=lcn_epsilon)
    print(f"  LCN patches shape: {patches_lcn.shape}")

    print("Computing ZCA whitening parameters...")
    params = compute_zca_params(patches_lcn, epsilon=zca_epsilon)
    print(f"  Eigenvalue range: [{params['s'].min():.4f}, {params['s'].max():.4f}]")

    print("Applying ZCA whitening...")
    patches_white = apply_zca_whitening(patches_lcn, params)
    print(f"  Whitened patches shape: {patches_white.shape}")

    params_path = os.path.join(output_dir, f"whitening_params_{patch_size}x{patch_size}.npz")
    patches_path = os.path.join(output_dir, f"patches_whitened_{patch_size}x{patch_size}.npy")

    save_whitening_params(params, params_path)
    np.save(patches_path, patches_white)
    print(f"Saved whitened patches to {patches_path}")

    print("Pipeline complete.")
    return patches_white, params
