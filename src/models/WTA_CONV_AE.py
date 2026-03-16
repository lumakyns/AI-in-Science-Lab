import torch
import torch.nn as nn


class WTA_CONV_AE(nn.Module):
    def __init__(
        self,
        dim: tuple,
        k_spatial: float,
        k_lifetime: float,
    ) -> None:
        super().__init__()
        self.in_ch, self.in_h, self.in_w, self.hidden_ch = dim
        self.k_spatial  = k_spatial
        self.k_lifetime = k_lifetime

        self.encoder   = nn.Conv2d(self.in_ch, self.hidden_ch, kernel_size=5, padding=2, bias=False)
        self.decoder   = nn.ConvTranspose2d(self.hidden_ch, self.in_ch, kernel_size=11, padding=5, bias=False)
        self.relu      = nn.ReLU()

    def _apply_spatial_sparsity(self, activations: torch.Tensor) -> torch.Tensor:
        B, C, H, W = activations.shape
        k = max(1, int(self.k_spatial * H * W))
        k = min(k, H * W)
        flat = activations.view(B, C, -1)
        _, topk_idx = torch.topk(flat, k, dim=2)
        mask = torch.zeros_like(flat)
        mask.scatter_(2, topk_idx, 1)
        return activations * mask.view(B, C, H, W)

    def _apply_lifetime_sparsity(self, activations: torch.Tensor) -> torch.Tensor:
        B, C, H, W = activations.shape
        k = max(1, int(self.k_lifetime * C * H * W))
        k = min(k, C * H * W)
        flat = activations.view(B, -1)
        _, topk_idx = torch.topk(flat, k, dim=1)
        mask = torch.zeros_like(flat)
        mask.scatter_(1, topk_idx, 1)
        return activations * mask.view(B, C, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.view(x.shape[0], self.in_ch, self.in_h, self.in_w)

        z1 = self.encoder(x)
        a1 = self.relu(z1)

        if self.training:
            a1 = self._apply_spatial_sparsity(a1)
            a1 = self._apply_lifetime_sparsity(a1)
        else:
            self.last_latent = a1.detach()

        z2 = self.decoder(a1)
        return z2.view(z2.shape[0], -1)
