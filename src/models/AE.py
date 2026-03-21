import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(
        self,
        dim: tuple
    ) -> None:
        super().__init__()

        self.input_dim, self.bottleneck_dim = dim

        self.encoder = nn.Linear(self.input_dim, self.bottleneck_dim, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.decoder = nn.Linear(self.bottleneck_dim, self.input_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        epoch: int = 0,
    ) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)

        z1 = self.encoder(x)
        a1 = self.sigmoid(z1)
        if not self.training:
            self.last_latent = a1.detach()
            self.last_filter_mask = torch.ones(
                a1.shape[0], self.bottleneck_dim, device=a1.device, dtype=a1.dtype
            )
        z2 = self.decoder(a1)

        return z2
