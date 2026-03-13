import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(
        self,
        dim: tuple
    ) -> None:
        super().__init__()

        self.input_dim, self.bottleneck_dim = dim

        self.encoder = nn.Linear(self.input_dim, self.bottleneck_dim)
        self.sigmoid = nn.Sigmoid()
        self.decoder = nn.Linear(self.bottleneck_dim, self.input_dim)

    def forward(
        self,
        x: torch.Tensor,
        epoch: int = 0,
    ) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)

        z1 = self.encoder(x)
        a1 = self.sigmoid(z1)
        z2 = self.decoder(a1)

        return z2
