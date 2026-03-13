import torch
import torch.nn as nn
import torch.nn.functional as F


class WTA_CONV_AE(nn.Module):
    def __init__(
        self,
        dim: tuple,
        k_spatial: int,
        k_lifetime: int,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.in_ch, self.in_h, self.in_w, self.out_ch = dim
        self.k_spatial = k_spatial
        self.k_lifetime = k_lifetime
        padding = kernel_size // 2

        self.padding      = padding
        self.encoder      = nn.Conv2d(self.in_ch, self.out_ch, kernel_size, padding=padding)
        self.relu         = nn.ReLU()
        self.decoder_bias = nn.Parameter(torch.zeros(self.in_ch, self.in_h, self.in_w))

    def _apply_spatial_sparsity(self, activations: torch.Tensor) -> torch.Tensor:
        B, C, H, W = activations.shape
        k = min(max(1, self.k_spatial), H * W)
        flat = activations.view(B, C, -1)
        _, topk_idx = torch.topk(flat, k, dim=2)
        mask = torch.zeros_like(flat)
        mask.scatter_(2, topk_idx, 1)
        return (activations * mask.view(B, C, H, W))

    def _apply_lifetime_sparsity(self, activations: torch.Tensor) -> torch.Tensor:
        B, C, H, W = activations.shape
        k = min(max(1, self.k_lifetime), B)
        scores = activations.sum(dim=(2, 3))
        _, topk_idx = torch.topk(scores, k, dim=0)
        mask = torch.zeros_like(scores)
        mask.scatter_(0, topk_idx, 1)
        return activations * mask.view(B, C, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.view(x.shape[0], self.in_ch, self.in_h, self.in_w)

        z1 = self.encoder(x)
        a1 = self.relu(z1)

        if self.training:
            a1 = self._apply_spatial_sparsity(a1)
            a1 = self._apply_lifetime_sparsity(a1)

        z2 = F.conv_transpose2d(a1, self.encoder.weight, padding=self.padding) + self.decoder_bias.unsqueeze(0)
        return z2.view(z2.shape[0], -1)
