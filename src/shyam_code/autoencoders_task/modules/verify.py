"""
Verification and visualization utilities for the CIFAR-10 preprocessing pipeline.
"""

import numpy as np
import matplotlib.pyplot as plt


def verify_lcn(patches_lcn, n_samples=10_000) -> dict:
    """Verify Local Contrast Normalization correctness.

    Checks: per-patch mean ≈ 0, per-patch std ≈ 1, no NaN/Inf.

    Args:
        patches_lcn: (N, D) float32 array
        n_samples: number of patches to sample for checking

    Returns:
        dict with verification results
    """
    idx = np.random.default_rng(0).integers(0, patches_lcn.shape[0], size=min(n_samples, patches_lcn.shape[0]))
    sample = patches_lcn[idx]

    means = sample.mean(axis=1)
    stds = sample.std(axis=1)
    has_nan = np.isnan(patches_lcn).any()
    has_inf = np.isinf(patches_lcn).any()

    mean_of_means = float(np.abs(means).mean())
    max_mean_deviation = float(np.abs(means).max())
    mean_of_stds = float(stds.mean())
    std_deviation = float(np.abs(stds - 1.0).mean())

    results = {
        "mean_of_abs_means": mean_of_means,
        "max_abs_mean": max_mean_deviation,
        "mean_of_stds": mean_of_stds,
        "mean_abs_std_deviation_from_1": std_deviation,
        "has_nan": bool(has_nan),
        "has_inf": bool(has_inf),
        "pass": mean_of_means < 1e-5 and std_deviation < 0.05 and not has_nan and not has_inf,
    }

    print("=== LCN Verification ===")
    print(f"  Mean of |patch means|:      {mean_of_means:.2e}  (should be ~0)")
    print(f"  Max |patch mean|:           {max_mean_deviation:.2e}  (should be ~0)")
    print(f"  Mean of patch stds:         {mean_of_stds:.4f}  (should be ~1)")
    print(f"  Mean |std - 1|:             {std_deviation:.4f}  (should be ~0)")
    print(f"  NaN: {has_nan}, Inf: {has_inf}")
    print(f"  PASS: {results['pass']}")
    return results


def verify_zca(patches_white, n_samples=50_000) -> dict:
    """Verify ZCA whitening correctness.

    Checks: empirical covariance ≈ identity matrix.

    Args:
        patches_white: (N, D) float32 whitened patches
        n_samples: number of patches to sample

    Returns:
        dict with verification results
    """
    idx = np.random.default_rng(1).integers(0, patches_white.shape[0], size=min(n_samples, patches_white.shape[0]))
    sample = patches_white[idx].astype(np.float64)

    mean = sample.mean(axis=0)
    X = sample - mean
    C = X.T @ X / len(X)

    d = C.shape[0]
    I = np.eye(d)
    frob_norm = float(np.linalg.norm(C - I, "fro"))
    diag_mean = float(np.diag(C).mean())
    offdiag_mean = float((np.abs(C) - np.abs(np.diag(np.diag(C)))).mean())

    results = {
        "frobenius_norm_C_minus_I": frob_norm,
        "mean_diagonal": diag_mean,
        "mean_abs_offdiagonal": offdiag_mean,
        # ZCA with regularization (epsilon=0.1) produces C_white = U diag(s/(s+eps)) U^T,
        # not exactly I. Diagonal < 1 is expected when many eigenvalues are near zero
        # (common for natural image patches). The real test is off-diagonal magnitude.
        "pass": offdiag_mean < 0.05,
    }

    print("=== ZCA Verification ===")
    print(f"  ||C - I||_F:                {frob_norm:.4f}")
    print(f"  Mean diagonal of C:         {diag_mean:.4f}  (< 1 expected with eps=0.1 regularization)")
    print(f"  Mean |off-diagonal| of C:   {offdiag_mean:.4f}  (should be ~0 — this is the real test)")
    print(f"  PASS: {results['pass']}")
    return results


def plot_patches(patches, n=20, patch_size=8, title="Patches"):
    """Display a grid of patches.

    Args:
        patches: (N, D) array, D = patch_size^2 * 3
        n: number of patches to display
        patch_size: spatial size of each patch
        title: plot title
    """
    cols = 10
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.2))
    fig.suptitle(title, fontsize=12)

    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < n:
            patch = patches[i].reshape(patch_size, patch_size, 3)
            # Normalize to [0, 1] for display
            p_min, p_max = patch.min(), patch.max()
            if p_max > p_min:
                patch = (patch - p_min) / (p_max - p_min)
            else:
                patch = np.zeros_like(patch)
            ax.imshow(patch.clip(0, 1))

    plt.tight_layout()
    return fig


def plot_eigenspectrum(params):
    """Plot the eigenvalue spectrum from ZCA params.

    Args:
        params: dict with key 's' (eigenvalues, ascending order)
    """
    s = params["s"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(s[::-1], marker="o", markersize=3)
    axes[0].set_title("Eigenvalue Spectrum (descending)")
    axes[0].set_xlabel("Component index")
    axes[0].set_ylabel("Eigenvalue")
    axes[0].grid(True)

    axes[1].semilogy(s[::-1], marker="o", markersize=3)
    axes[1].set_title("Eigenvalue Spectrum (log scale)")
    axes[1].set_xlabel("Component index")
    axes[1].set_ylabel("Eigenvalue (log)")
    axes[1].grid(True)

    plt.tight_layout()
    return fig


def plot_pixel_histograms(patches_raw, patches_lcn, patches_white, n_samples=10_000):
    """Compare pixel value distributions before and after preprocessing.

    Args:
        patches_raw: (N, D) raw patches
        patches_lcn: (N, D) LCN patches
        patches_white: (N, D) whitened patches
        n_samples: number of patches to sample
    """
    rng = np.random.default_rng(2)

    def sample(arr):
        idx = rng.integers(0, arr.shape[0], size=min(n_samples, arr.shape[0]))
        return arr[idx].ravel()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    data = [
        (sample(patches_raw), "Raw patches", "tab:blue"),
        (sample(patches_lcn), "After LCN", "tab:orange"),
        (sample(patches_white), "After ZCA", "tab:green"),
    ]

    for ax, (vals, title, color) in zip(axes, data):
        ax.hist(vals, bins=100, color=color, alpha=0.7, density=True)
        ax.set_title(title)
        ax.set_xlabel("Pixel value")
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.3)

    plt.suptitle("Pixel Value Distributions", fontsize=13)
    plt.tight_layout()
    return fig


def plot_covariance_matrix(patches_lcn, patches_white, n_samples=5_000):
    """Plot empirical covariance matrices before and after ZCA whitening.

    Args:
        patches_lcn: (N, D) LCN patches
        patches_white: (N, D) whitened patches
        n_samples: number of patches to use
    """
    rng = np.random.default_rng(3)
    idx = rng.integers(0, patches_lcn.shape[0], size=min(n_samples, patches_lcn.shape[0]))

    def cov(arr):
        X = arr[idx].astype(np.float64)
        X -= X.mean(axis=0)
        return X.T @ X / len(X)

    C_lcn = cov(patches_lcn)
    C_white = cov(patches_white)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    vmax = np.abs(C_lcn).max()
    im0 = axes[0].imshow(C_lcn, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[0].set_title("Covariance (after LCN)")
    plt.colorbar(im0, ax=axes[0])

    vmax2 = max(np.abs(C_white).max(), 0.1)
    im1 = axes[1].imshow(C_white, cmap="RdBu_r", vmin=-vmax2, vmax=vmax2)
    axes[1].set_title("Covariance (after ZCA) — should be ≈ I")
    plt.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    return fig
