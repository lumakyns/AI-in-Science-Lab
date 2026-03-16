"""
Main training script for the two-layer FC-WTA sparse autoencoder.
Usage:
    python train.py [--patch_size 8] [--hidden_dim 256] [--sparsity_rate 0.05]
                   [--n_epochs 50] [--lr 1e-3] [--batch_size 10000]
                   [--force_preprocess]
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt


sys.path.insert(0, "./modules")
from preprocessing import (
    preprocess_pipeline, load_whitening_params,
    load_cifar10, extract_patches, local_contrast_normalize,
)
from two_layer_sparse_autoencoder import make_dataloader, WTAAutoencoder, train_autoencoder
from verify import verify_lcn, verify_zca

# Argument parsing
parser = argparse.ArgumentParser(description="Train FC-WTA Sparse Autoencoder on CIFAR-10 patches")

parser.add_argument("--patch_size",      type=int,   default=8,      help="Patch size (7, 8, or 11)")
parser.add_argument("--hidden_dim",      type=int,   default=256,    help="Hidden layer size (256 or 512)")
parser.add_argument("--sparsity_rate",   type=float, default=0.05,   help="WTA sparsity rate (e.g. 0.05 = 5%%)")
parser.add_argument("--n_epochs",        type=int,   default=50,     help="Number of training epochs")
parser.add_argument("--lr",              type=float, default=1e-3,   help="Adam learning rate")
parser.add_argument("--batch_size",      type=int,   default=10_000, help="Patches per batch")
parser.add_argument("--tied",            action="store_true", default=True, help="Use tied weights")
parser.add_argument("--output_dir",      type=str,   default="./outputs", help="Directory to save all outputs")
parser.add_argument("--force_preprocess",action="store_true", default=False, help="Rerun preprocessing even if outputs exist")

args = parser.parse_args()

# Fixed config
DATA_DIR   = "./data"
OUTPUT_DIR = args.output_dir
N_PATCHES  = 1_000_000
SEED       = 42
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE}")
print(f"Config: patch_size={args.patch_size}, hidden_dim={args.hidden_dim}, "
      f"sparsity_rate={args.sparsity_rate}, n_epochs={args.n_epochs}, lr={args.lr}\n")

# Step 1–2: Preprocessing
patches_path = os.path.join(OUTPUT_DIR, f"patches_whitened_{args.patch_size}x{args.patch_size}.npy")
params_path  = os.path.join(OUTPUT_DIR, f"whitening_params_{args.patch_size}x{args.patch_size}.npz")

if not args.force_preprocess and os.path.exists(patches_path) and os.path.exists(params_path):
    print("Preprocessed files found — skipping preprocessing.")
    params = load_whitening_params(params_path)
else:
    print("Running preprocessing pipeline...")
    _, params = preprocess_pipeline(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        n_patches=N_PATCHES,
        patch_size=args.patch_size,
        seed=SEED,
    )

print(f"  W shape        : {params['W'].shape}")
print(f"  Eigenvalue range: [{params['s'].min():.4f}, {params['s'].max():.4f}]")

# Verify LCN + ZCA
print("\nVerifying LCN (fresh 50k sample)...")
images      = load_cifar10(DATA_DIR)
patches     = extract_patches(images, n_patches=50_000, patch_size=args.patch_size, seed=SEED)
patches_lcn = local_contrast_normalize(patches)
verify_lcn(patches_lcn)

print("\nVerifying ZCA (full 1M saved patches)...")
patches_white_full = np.load(patches_path)
verify_zca(patches_white_full)

# Step 3: DataLoader
print()
loader = make_dataloader(patches_path, batch_size=args.batch_size, seed=SEED)

# Step 4: Train FC-WTA Autoencoder
input_dim = args.patch_size ** 2 * 3
print(f"\n--- Step 4: Training FC-WTA Autoencoder ---")
print(f"  Architecture : {input_dim} → {args.hidden_dim} (WTA {args.sparsity_rate*100:.0f}%) → {input_dim}")
print(f"  Tied weights : {args.tied}")
print(f"  Epochs       : {args.n_epochs}")

model = WTAAutoencoder(
    input_dim=input_dim,
    hidden_dim=args.hidden_dim,
    sparsity_rate=args.sparsity_rate,
    tied=args.tied,
).to(DEVICE)

losses = train_autoencoder(model, loader, n_epochs=args.n_epochs, lr=args.lr, device=DEVICE)

# Save model
model_name = f"wta_p{args.patch_size}_h{args.hidden_dim}_s{args.sparsity_rate}_e{args.n_epochs}.pt"
model_path = os.path.join(OUTPUT_DIR, model_name)
torch.save(model.state_dict(), model_path)
print(f"\nModel saved to {model_path}")
print(f"Final loss: {losses[-1]:.6f}")

# Step 5: Visualize learned encoder filters
print("\n--- Step 5: Visualizing Encoder Filters ---")

# Wenc shape: (hidden_dim, input_dim) — each row is one filter
filters = model.encoder.weight.detach().cpu().numpy()  # (hidden_dim, input_dim)
n_filters = filters.shape[0]

cols = 16
rows = (n_filters + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.2))
fig.suptitle(
    f"Learned Encoder Filters — patch={args.patch_size}x{args.patch_size}, "
    f"hidden={args.hidden_dim}, sparsity={args.sparsity_rate}, epochs={args.n_epochs}",
    fontsize=11
)

for i, ax in enumerate(axes.flat):
    ax.axis("off")
    if i < n_filters:
        f = filters[i].reshape(args.patch_size, args.patch_size, 3)
        # normalize each filter independently to [0, 1] for display
        f_min, f_max = f.min(), f.max()
        if f_max > f_min:
            f = (f - f_min) / (f_max - f_min)
        ax.imshow(f.clip(0, 1))

plt.tight_layout()
fig_name = f"filters_p{args.patch_size}_h{args.hidden_dim}_s{args.sparsity_rate}_e{args.n_epochs}.png"
fig_path = os.path.join(OUTPUT_DIR, fig_name)
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"Filter visualization saved to {fig_path}")
plt.show()

# Step 5b: Loss curve
print("\n--- Step 5b: Loss Curve ---")

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(1, len(losses) + 1), losses, linewidth=2)
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE Loss")
ax.set_title(
    f"Training Loss — patch={args.patch_size}x{args.patch_size}, "
    f"hidden={args.hidden_dim}, sparsity={args.sparsity_rate}"
)
ax.grid(True, alpha=0.3)
plt.tight_layout()
loss_fig_name = f"loss_p{args.patch_size}_h{args.hidden_dim}_s{args.sparsity_rate}_e{args.n_epochs}.png"
loss_fig_path = os.path.join(OUTPUT_DIR, loss_fig_name)
plt.savefig(loss_fig_path, dpi=150, bbox_inches="tight")
print(f"Loss curve saved to {loss_fig_path}")
plt.show()

# Step 5c: Original vs reconstructed patches
print("\n--- Step 5c: Original vs Reconstructed Patches ---")

model.eval()
sample_batch = next(iter(loader)).to(DEVICE)[:16]  # grab 16 patches
with torch.no_grad():
    recon_batch, _ = model(sample_batch, training=False)

originals = sample_batch.cpu().numpy()    # (16, D)
recons    = recon_batch.cpu().numpy()     # (16, D)

fig, axes = plt.subplots(4, 8, figsize=(16, 8))
fig.suptitle(
    f"Original (top) vs Reconstructed (bottom) — epoch {args.n_epochs}",
    fontsize=12
)

for i in range(16):
    # original
    ax_orig = axes[i // 8 * 2, i % 8]
    orig = originals[i].reshape(args.patch_size, args.patch_size, 3)
    o_min, o_max = orig.min(), orig.max()
    if o_max > o_min:
        orig = (orig - o_min) / (o_max - o_min)
    ax_orig.imshow(orig.clip(0, 1))
    ax_orig.axis("off")
    if i % 8 == 0:
        ax_orig.set_ylabel("Original", fontsize=9)

    # reconstruction
    ax_recon = axes[i // 8 * 2 + 1, i % 8]
    recon = recons[i].reshape(args.patch_size, args.patch_size, 3)
    r_min, r_max = recon.min(), recon.max()
    if r_max > r_min:
        recon = (recon - r_min) / (r_max - r_min)
    ax_recon.imshow(recon.clip(0, 1))
    ax_recon.axis("off")
    if i % 8 == 0:
        ax_recon.set_ylabel("Recon", fontsize=9)

plt.tight_layout()
recon_fig_name = f"recon_p{args.patch_size}_h{args.hidden_dim}_s{args.sparsity_rate}_e{args.n_epochs}.png"
recon_fig_path = os.path.join(OUTPUT_DIR, recon_fig_name)
plt.savefig(recon_fig_path, dpi=150, bbox_inches="tight")
print(f"Reconstruction comparison saved to {recon_fig_path}")
plt.show()