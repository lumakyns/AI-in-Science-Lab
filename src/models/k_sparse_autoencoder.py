import torch
import torch.nn as nn
import torch.nn.functional as F


class K_Sparse_Autoencoder(nn.Module):
    def __init__(
        self,
        dim: tuple,
        k: int,
        total_epochs: int,
    ) -> None:
        super().__init__()

        self.input_dim, self.bottleneck_dim = dim
        self.k = k  # number of activations to keep per sample
        self.total_epochs = total_epochs

        self.encoder = nn.Linear(self.input_dim, self.bottleneck_dim)
        self.identity = nn.Identity()
        self.decoder_bias = nn.Parameter(torch.zeros(self.input_dim))

    def forward(
        self,
        x: torch.Tensor,
        epoch: int = 0,
    ) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)

        z1 = self.encoder(x)
        a1 = self.identity(z1)

        if self.training:
            anneal_epochs = self.total_epochs // 2
            if epoch < anneal_epochs:
                progress = epoch / anneal_epochs
                current_k = self.bottleneck_dim + progress * (self.k - self.bottleneck_dim)
            else:
                current_k = self.k
        else:
            current_k = self.k

        k_count = min(max(1, int(current_k)), self.bottleneck_dim)
        _, a1_topk_idx = torch.topk(a1, k_count, dim=1)
        a1_wta_mask = torch.zeros_like(a1)
        a1_wta_mask.scatter_(1, a1_topk_idx, 1)
        a1 = a1 * a1_wta_mask

        z2 = F.linear(a1, self.encoder.weight.t(), self.decoder_bias)

        return z2

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)

        a1 = self.identity(self.encoder(x))
        k_count = min(max(1, self.k), self.bottleneck_dim)
        _, topk_idx = torch.topk(a1, k_count, dim=1)
        mask = torch.zeros_like(a1).scatter_(1, topk_idx, 1)
        return a1 * mask
