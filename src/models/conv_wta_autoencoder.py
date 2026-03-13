import torch
import torch.nn as nn
import torch.nn.functional as F


class CONV_WTA_Autoencoder(nn.Module):
    def __init__(
        self,
        dim: tuple,
        k: int,
        total_epochs: int = 5000,
        dataset_size: int = 1000000,
        a: int = 1,
    ) -> None:
        super().__init__()
        in_ch, in_h, in_w, bottleneck_dim = dim
        self.in_ch = in_ch
        self.in_h = in_h
        self.in_w = in_w
        self.bottleneck_dim = bottleneck_dim
        self.k = k
        self.a = a
        self.total_epochs = total_epochs
        self.dataset_size = dataset_size

        self.encoder = nn.Conv2d(in_ch, bottleneck_dim, kernel_size=in_h, stride=1, padding=0)
        self.decoder_bias = nn.Parameter(torch.zeros(in_ch * in_h * in_w))

    def _apply_topk_mask(self, activations: torch.Tensor, k_count: int) -> torch.Tensor:
        k_count = min(max(1, int(k_count)), activations.shape[1])
        _, topk_idx = torch.topk(activations, k_count, dim=1)
        mask = torch.zeros_like(activations)
        mask.scatter_(1, topk_idx, 1)
        return activations * mask

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.view(x.shape[0], self.in_ch, self.in_h, self.in_w)
        z = self.encoder(x)
        return z.squeeze(-1).squeeze(-1)

    def forward(
        self,
        x: torch.Tensor,
        epoch: int = 0,
        inputs_processed_in_epoch: int = 0,
    ) -> torch.Tensor:
        if x.ndim == 2:
            x = x.view(x.shape[0], self.in_ch, self.in_h, self.in_w)

        z1 = self.encoder(x)
        a1 = z1.squeeze(-1).squeeze(-1)

        if self.training:
            current_samples = epoch * self.dataset_size + inputs_processed_in_epoch
            anneal_samples = (self.total_epochs // 2) * self.dataset_size
            if anneal_samples > 0:
                progress = min(current_samples / anneal_samples, 1.0)
            else:
                progress = 1.0
            current_k = self.bottleneck_dim + progress * (self.k - self.bottleneck_dim)
        else:
            current_k = self.a * self.k

        k_count = min(max(1, int(current_k)), self.bottleneck_dim)
        a1 = self._apply_topk_mask(a1, k_count)

        a1 = a1.unsqueeze(-1).unsqueeze(-1)
        w = self.encoder.weight.permute(1, 0, 2, 3)
        z2 = F.conv_transpose2d(a1, w)
        z2 = z2 + self.decoder_bias.view(1, self.in_ch, self.in_h, self.in_w)
        return z2.view(z2.shape[0], -1)
