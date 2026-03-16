"""
Two-Layer Sparse Autoencoder on CIFAR-10 patches.
Step 3: Data loading — sample 10,000 random patches per batch from preprocessed pool.
Step 4: (coming soon) Winner-take-all sparse autoencoder training.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# Step 3: Patch Dataset & DataLoader
class PatchDataset(Dataset):
    """Dataset wrapping the preprocessed (LCN + ZCA whitened) patch pool."""

    def __init__(self, patches_path):
        """
        Args:
            patches_path: path to patches_whitened_{P}x{P}.npy, shape (N, D)
        """
        data = np.load(patches_path)
        self.patches = torch.from_numpy(data)  # (N, D) float32 tensor
        print(f"Loaded {len(self.patches):,} patches of dim {self.patches.shape[1]} from {patches_path}")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return self.patches[idx]


def make_dataloader(patches_path, batch_size=10_000, seed=42):
    """Create a DataLoader that randomly samples patches each epoch.

    Args:
        patches_path: path to preprocessed .npy patch file
        batch_size: number of patches per batch (default 10,000)
        seed: random seed for reproducibility

    Returns:
        DataLoader yielding (batch_size, D) tensors
    """
    dataset = PatchDataset(patches_path)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        drop_last=True,   # ensures every batch is exactly batch_size
    )
    n_batches = len(loader)
    print(f"DataLoader: {n_batches} batches/epoch (batch_size={batch_size:,}, pool={len(dataset):,})")
    return loader


# Step 4: FC-WTA Autoencoder
import torch.nn as nn
import torch.nn.functional as F


class WTAAutoencoder(nn.Module):
    """Two-layer Fully-Connected Winner-Take-All Autoencoder.

    Architecture (Makhzani & Frey 2015, Section 2):
        Encoder: Linear(input_dim, hidden_dim) + ReLU
        WTA:     per hidden unit, keep top k% activations across batch
        Decoder: Linear(hidden_dim, input_dim), no activation (linear)

    Tied weights: decoder weight = encoder.weight.T (Appendix A.2).
    """

    def __init__(self, input_dim, hidden_dim, sparsity_rate=0.05, tied=True):
        """
        Args:
            input_dim:     patch dimensionality (PATCH_SIZE^2 * 3)
            hidden_dim:    number of hidden units (e.g. 256 or 512)
            sparsity_rate: fraction of batch to keep per hidden unit (e.g. 0.05 = 5%)
            tied:          if True, decoder weight = encoder.weight.T
        """
        super().__init__()
        self.sparsity_rate = sparsity_rate
        self.tied = tied

        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder_bias = nn.Parameter(torch.zeros(input_dim))

        if not tied:
            self.decoder = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        """x: (B, D) → h: (B, H)"""
        return F.relu(self.encoder(x))

    def apply_wta(self, h):
        """Lifetime sparsity: for each hidden unit, keep top k% across the batch.

        h: (B, H) → h_sparse: (B, H) with ~(1 - sparsity_rate) fraction zeroed.
        Gradients are blocked at zeroed positions.
        """
        k = max(1, int(self.sparsity_rate * h.shape[0]))
        topk_vals, _ = torch.topk(h, k, dim=0)   # (k, H) — top-k across batch
        threshold = topk_vals[-1:, :]              # (1, H) — k-th largest per unit
        mask = (h >= threshold).float()
        return h * mask

    def decode(self, h):
        """h: (B, H) → x_hat: (B, D)"""
        if self.tied:
            return F.linear(h, self.encoder.weight.t(), self.decoder_bias)
        return self.decoder(h)

    def forward(self, x, training=True):
        """
        Args:
            x:        (B, D) input patches
            training: if True, apply WTA sparsity; if False, use raw ReLU (test time)

        Returns:
            x_hat:    (B, D) reconstruction
            h_sparse: (B, H) sparse hidden representation
        """
        h = self.encode(x)
        h_sparse = self.apply_wta(h) if training else h
        x_hat = self.decode(h_sparse)
        return x_hat, h_sparse


def train_autoencoder(model, loader, n_epochs, lr, device):
    """Train the WTA autoencoder with MSE reconstruction loss.

    Args:
        model:    WTAAutoencoder instance
        loader:   DataLoader yielding (batch_size, D) patch batches
        n_epochs: number of training epochs
        lr:       Adam learning rate
        device:   'cuda' or 'cpu'

    Returns:
        losses: list of per-epoch average MSE losses
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_sparsity = 0.0

        for batch in loader:
            x = batch.to(device)

            x_hat, h_sparse = model(x, training=True)
            loss = F.mse_loss(x_hat, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_sparsity += (h_sparse == 0).float().mean().item()

        avg_loss = epoch_loss / len(loader)
        avg_sparsity = epoch_sparsity / len(loader)
        losses.append(avg_loss)

        print(f"  Epoch {epoch:3d}/{n_epochs}  loss={avg_loss:.6f}  sparsity={avg_sparsity:.3f}")

    return losses
