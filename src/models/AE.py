import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(
        self,
        dim: tuple
    ) -> None:
        super().__init__()

        self.input_dim, self.bottleneck_dim = dim
        self.uses_k_population = False
        self.is_convolutional  = False

        self.encoder = nn.Linear(self.input_dim, self.bottleneck_dim, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.decoder = nn.Linear(self.bottleneck_dim, self.input_dim, bias=False)

        self.last_filter_mask = torch.ones(1, self.bottleneck_dim)

    @property
    def detached_encoder_weights(self) -> torch.Tensor:
        return self.encoder.weight.detach()

    @property
    def detached_decoder_weights(self) -> torch.Tensor:
        return self.decoder.weight.detach()

    def forward(
        self,
        x: torch.Tensor,
        epoch: int = 0,
    ) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)

        z1 = self.encoder(x)
        h1 = self.sigmoid(z1)
        if not self.training:
            self.last_latent = h1.detach()
            
        z2 = self.decoder(h1)

        return z2, h1 # output, hidden
