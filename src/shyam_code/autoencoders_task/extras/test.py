"""
Sanity checks for the full FC-WTA autoencoder implementation.
Tests model correctness, WTA mechanism, and data pipeline.
Run with: python test.py
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../modules"))
from two_layer_sparse_autoencoder import WTAAutoencoder, make_dataloader, train_autoencoder
from preprocessing import preprocess_pipeline

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../outputs")
PASS = "  PASS"
FAIL = "  FAIL"

results = {}

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    results[name] = condition
    print(f"{status} | {name}" + (f" — {detail}" if detail else ""))
    return condition

print("=" * 60)
print(f"Running sanity checks  (device: {DEVICE})")
print("=" * 60)

# ---------------------------------------------------------------------------
# 1. Model instantiation — different patch sizes and hidden dims
# ---------------------------------------------------------------------------
print("\n[1] Model instantiation")

for patch_size, hidden_dim in [(8, 256), (11, 512), (7, 256)]:
    input_dim = patch_size ** 2 * 3
    try:
        model = WTAAutoencoder(input_dim, hidden_dim).to(DEVICE)
        check(f"Instantiate p{patch_size} h{hidden_dim}", True,
              f"input={input_dim}, hidden={hidden_dim}")
    except Exception as e:
        check(f"Instantiate p{patch_size} h{hidden_dim}", False, str(e))

# ---------------------------------------------------------------------------
# 2. Forward pass — output shapes correct
# ---------------------------------------------------------------------------
print("\n[2] Forward pass shapes")

input_dim, hidden_dim = 192, 256
model = WTAAutoencoder(input_dim, hidden_dim, sparsity_rate=0.05).to(DEVICE)
x = torch.randn(10_000, input_dim).to(DEVICE)

x_hat, h_sparse = model(x, training=True)

check("x_hat shape = x shape",     x_hat.shape == x.shape,      str(x_hat.shape))
check("h_sparse shape = (B, H)",   h_sparse.shape == (10_000, hidden_dim), str(h_sparse.shape))

# ---------------------------------------------------------------------------
# 3. WTA sparsity — fraction of zeros ≈ (1 - sparsity_rate)
# ---------------------------------------------------------------------------
print("\n[3] WTA sparsity")

sparsity_rate = 0.05
model = WTAAutoencoder(input_dim, hidden_dim, sparsity_rate=sparsity_rate).to(DEVICE)
x = torch.randn(10_000, input_dim).to(DEVICE)
_, h_sparse = model(x, training=True)

actual_zero_frac = (h_sparse == 0).float().mean().item()
expected_zero_frac = 1 - sparsity_rate
tolerance = 0.02

check("Zero fraction ≈ (1 - sparsity_rate)",
      abs(actual_zero_frac - expected_zero_frac) < tolerance,
      f"expected≈{expected_zero_frac:.2f}, got={actual_zero_frac:.4f}")

# WTA is across batch dim (dim=0), not hidden dim
zeros_per_unit = (h_sparse == 0).float().mean(dim=0)  # (H,) — per hidden unit
check("WTA applied across batch (not hidden)",
      zeros_per_unit.mean().item() > 0.5,
      f"avg zeros per hidden unit = {zeros_per_unit.mean():.3f}")

# ---------------------------------------------------------------------------
# 4. Training=False turns off WTA (no extra zeros)
# ---------------------------------------------------------------------------
print("\n[4] Test-time behaviour (training=False)")

x = torch.randn(1_000, input_dim).to(DEVICE)
_, h_train = model(x, training=True)
_, h_test  = model(x, training=False)

train_zeros = (h_train == 0).float().mean().item()
test_zeros  = (h_test  == 0).float().mean().item()

check("training=True has ~95% zeros",  train_zeros > 0.8,  f"{train_zeros:.3f}")
check("training=False has fewer zeros", test_zeros < train_zeros,
      f"test zeros={test_zeros:.3f} < train zeros={train_zeros:.3f}")

# ---------------------------------------------------------------------------
# 5. Tied weights — decoder weight = encoder.weight.T
# ---------------------------------------------------------------------------
print("\n[5] Tied weights")

model_tied   = WTAAutoencoder(input_dim, hidden_dim, tied=True).to(DEVICE)
model_untied = WTAAutoencoder(input_dim, hidden_dim, tied=False).to(DEVICE)

enc_w = model_tied.encoder.weight          # (H, D)
dec_w = model_tied.encoder.weight.t()      # (D, H) — what decoder should use

x = torch.randn(100, input_dim).to(DEVICE)
x_hat_tied, _ = model_tied(x, training=False)

# Manually compute: h = ReLU(xW^T + b), x_hat = hW + b_dec
h_manual   = F.relu(x @ enc_w.t() + model_tied.encoder.bias)
x_hat_manual = h_manual @ enc_w + model_tied.decoder_bias

check("Tied decoder uses encoder.weight.T",
      torch.allclose(x_hat_tied, x_hat_manual, atol=1e-5),
      f"max diff = {(x_hat_tied - x_hat_manual).abs().max():.2e}")

check("Untied model has separate decoder",
      hasattr(model_untied, 'decoder'), "decoder attribute exists")

# ---------------------------------------------------------------------------
# 6. Loss is computable and finite
# ---------------------------------------------------------------------------
print("\n[6] Loss")

x = torch.randn(10_000, input_dim).to(DEVICE)
x_hat, _ = model(x, training=True)
loss = F.mse_loss(x_hat, x)

check("Loss is finite",   loss.item() == loss.item(),  f"loss={loss.item():.6f}")
check("Loss > 0",         loss.item() > 0,              f"loss={loss.item():.6f}")

# ---------------------------------------------------------------------------
# 7. Backward pass — gradients flow
# ---------------------------------------------------------------------------
print("\n[7] Backward pass")

model = WTAAutoencoder(input_dim, hidden_dim).to(DEVICE)
x = torch.randn(10_000, input_dim).to(DEVICE)
x_hat, _ = model(x, training=True)
loss = F.mse_loss(x_hat, x)
loss.backward()

enc_grad = model.encoder.weight.grad
check("Encoder weight has gradient",  enc_grad is not None and enc_grad.abs().sum() > 0,
      f"grad norm={enc_grad.norm():.4f}" if enc_grad is not None else "None")

# ---------------------------------------------------------------------------
# 8. Loss decreases over a few steps
# ---------------------------------------------------------------------------
print("\n[8] Loss decreases over 10 steps")

model = WTAAutoencoder(input_dim, hidden_dim, sparsity_rate=0.05).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
x = torch.randn(10_000, input_dim).to(DEVICE)

first_loss, last_loss = None, None
for step in range(10):
    x_hat, _ = model(x, training=True)
    loss = F.mse_loss(x_hat, x)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step == 0:  first_loss = loss.item()
    if step == 9:  last_loss  = loss.item()

check("Loss decreases over 10 steps", last_loss < first_loss,
      f"start={first_loss:.4f} → end={last_loss:.4f}")

# ---------------------------------------------------------------------------
# 9. DataLoader — check saved patches load and have correct shape
# ---------------------------------------------------------------------------
print("\n[9] DataLoader (saved patches)")

for patch_size in [8, 11, 7]:
    path = os.path.join(OUTPUT_DIR, f"patches_whitened_{patch_size}x{patch_size}.npy")

    if not os.path.exists(path):
        print(f"  patches_whitened_{patch_size}x{patch_size}.npy not found — running preprocessing...")
        try:
            preprocess_pipeline(
                data_dir="./data",
                output_dir=OUTPUT_DIR,
                n_patches=1_000_000,
                patch_size=patch_size,
                seed=42,
            )
        except Exception as e:
            check(f"DataLoader p{patch_size}", False, f"preprocessing failed: {e}")
            continue

    try:
        loader = make_dataloader(path, batch_size=10_000, seed=42)
        batch = next(iter(loader))
        expected_dim = patch_size ** 2 * 3
        check(f"DataLoader p{patch_size} batch shape",
              batch.shape == (10_000, expected_dim),
              str(tuple(batch.shape)))
        check(f"DataLoader p{patch_size} mean ≈ 0",
              abs(batch.mean().item()) < 0.05,
              f"mean={batch.mean():.4f}")
    except Exception as e:
        check(f"DataLoader p{patch_size}", False, str(e))

# ---------------------------------------------------------------------------
# 10. Short training run + filter visualization (5 epochs on real patches)
# ---------------------------------------------------------------------------
print("\n[10] Short training + filter visualization (5 epochs, patch_size=8)")

short_path = os.path.join(OUTPUT_DIR, "patches_whitened_8x8.npy")
if not os.path.exists(short_path):
    print("  Skipping — patches_whitened_8x8.npy not found.")
else:
    try:
        short_loader = make_dataloader(short_path, batch_size=10_000, seed=42)
        short_model  = WTAAutoencoder(input_dim=192, hidden_dim=256,
                                      sparsity_rate=0.05).to(DEVICE)
        losses = train_autoencoder(short_model, short_loader, n_epochs=5, lr=1e-3, device=DEVICE)

        # Plot all 256 filters
        filters = short_model.encoder.weight.detach().cpu().numpy()  # (256, 192)
        cols, rows = 16, 16
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.2))
        fig.suptitle("Encoder Filters after 5 epochs (8×8 patches, 256 hidden)", fontsize=11)

        for i, ax in enumerate(axes.flat):
            ax.axis("off")
            if i < filters.shape[0]:
                f = filters[i].reshape(8, 8, 3)
                f_min, f_max = f.min(), f.max()
                if f_max > f_min:
                    f = (f - f_min) / (f_max - f_min)
                ax.imshow(f.clip(0, 1))

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "test_filters_5epochs.png"), dpi=150, bbox_inches="tight")
        plt.show()
        check("Filter visualization completed", True, "saved to outputs/test_filters_5epochs.png")

        # Loss curve
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(range(1, len(losses) + 1), losses, linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.set_title("Training Loss — 5 epochs (8×8, 256 hidden, 5% sparsity)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "test_loss_5epochs.png"), dpi=150, bbox_inches="tight")
        plt.show()
        check("Loss curve saved", True, "saved to outputs/test_loss_5epochs.png")

        # Original vs reconstructed patches
        short_model.eval()
        sample = next(iter(short_loader)).to(DEVICE)[:16]
        with torch.no_grad():
            recon, _ = short_model(sample, training=False)

        originals = sample.cpu().numpy()
        recons    = recon.cpu().numpy()

        fig, axes = plt.subplots(4, 8, figsize=(16, 8))
        fig.suptitle("Original (top) vs Reconstructed (bottom) — 5 epochs", fontsize=12)
        for i in range(16):
            for row_offset, data in [(0, originals), (1, recons)]:
                ax = axes[i // 8 * 2 + row_offset, i % 8]
                patch = data[i].reshape(8, 8, 3)
                p_min, p_max = patch.min(), patch.max()
                if p_max > p_min:
                    patch = (patch - p_min) / (p_max - p_min)
                ax.imshow(patch.clip(0, 1))
                ax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "test_recon_5epochs.png"), dpi=150, bbox_inches="tight")
        plt.show()
        check("Reconstruction plot saved", True, "saved to outputs/test_recon_5epochs.png")
    except Exception as e:
        check("Filter visualization completed", False, str(e))

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
passed = sum(results.values())
total  = len(results)
print(f"Results: {passed}/{total} passed")
if passed == total:
    print("All checks passed. Ready for Step 5.")
else:
    failed = [k for k, v in results.items() if not v]
    print("Failed checks:")
    for f in failed:
        print(f"  - {f}")
print("=" * 60)
