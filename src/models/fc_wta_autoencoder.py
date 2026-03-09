import torch
import torch.nn as nn
import torch.nn.functional as F


class FC_WTA_Autoencoder(nn.Module):
    def __init__(
        self,
        dim: tuple,
        k: int,
    ) -> None:
        super().__init__()

        self.input_dim, self.bottleneck_dim = dim
        self.k = k  # number of winners (activations to keep) per hidden unit

        self.encoder      = nn.Linear(self.input_dim, self.bottleneck_dim)
        self.relu         = nn.ReLU()
        self.decoder_bias = nn.Parameter(torch.zeros(self.input_dim))

    def forward(
        self,
        x: torch.Tensor,
        epoch: int = 0,
    ) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)

        z1 = self.encoder(x)
        a1 = self.relu(z1)

        if self.training:
            k_count = min(max(1, self.k), x.shape[0])
            _, a1_topk_idx = torch.topk(a1, k_count, dim=0)
            a1_wta_mask = torch.zeros_like(a1)
            a1_wta_mask.scatter_(0, a1_topk_idx, 1)
            a1 = a1 * a1_wta_mask

        z2 = F.linear(a1, self.encoder.weight.t(), self.decoder_bias)

        return z2

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return self.relu(self.encoder(x))
