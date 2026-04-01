import torch
import torch.nn as nn


class WTA_FC_AE(nn.Module):
    def __init__(
        self,
        dim: tuple,
        k_lifetime: float,
    ) -> None:
        super().__init__()

        self.input_dim, self.bottleneck_dim = dim
        self.k_lifetime        = k_lifetime
        self.uses_k_population = False
        self.is_convolutional  = False

        self.encoder = nn.Linear(self.input_dim, self.bottleneck_dim, bias=False)
        self.relu    = nn.ReLU()
        self.decoder = nn.Linear(self.bottleneck_dim, self.input_dim, bias=False)

        self.last_filter_mask = torch.ones(1, self.bottleneck_dim)

    @property
    def detached_encoder_weights(self) -> torch.Tensor:
        return self.encoder.weight.detach()

    @property
    def detached_decoder_weights(self) -> torch.Tensor:
        return self.decoder.weight.detach()

    def _apply_lifetime_sparsity(self, activations: torch.Tensor) -> torch.Tensor:
        batch_size = activations.shape[0]
        k_count = max(1, int(self.k_lifetime * batch_size))
        k_count = min(k_count, batch_size)
        _, topk_idx = torch.topk(activations, k_count, dim=0)
        mask = torch.zeros_like(activations)
        mask.scatter_(0, topk_idx, 1)
        return activations * mask

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)

        z1 = self.encoder(x)
        a1 = self.relu(z1)

        if self.training:
            a1 = self._apply_lifetime_sparsity(a1)
        else:
            self.last_latent = a1.detach()

        z2 = self.decoder(a1)

        return z2
