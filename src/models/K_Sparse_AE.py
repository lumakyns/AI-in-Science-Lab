import torch
import torch.nn as nn
import torch.nn.functional as F


class K_Sparse_AE(nn.Module):
    def __init__(
        self,
        dim: tuple,
        k_population: float,
        total_epochs: int,
        dataset_size: int,
        a: int = 1,
    ) -> None:
        super().__init__()

        self.input_dim, self.bottleneck_dim = dim
        self.a            = a
        self.k_population = k_population
        self.total_epochs = total_epochs
        self.dataset_size = dataset_size
        self.uses_k_population = True
        self.is_convolutional  = False

        self.encoder  = nn.Linear(self.input_dim, self.bottleneck_dim, bias=False)
        self.identity = nn.Identity()
        self.decoder  = nn.Linear(self.bottleneck_dim, self.input_dim, bias=False)

    @property
    def detached_encoder_weights(self) -> torch.Tensor:
        return self.encoder.weight.detach()

    @property
    def detached_decoder_weights(self) -> torch.Tensor:
        return self.decoder.weight.detach()

    def _compute_annealed_k(
        self,
        epoch: int,
        inputs_processed_in_epoch: int,
        target_k: float,
        training: bool,
    ) -> float:
        if training:
            current_samples = epoch * self.dataset_size + inputs_processed_in_epoch
            anneal_samples  = (self.total_epochs // 2) * self.dataset_size
            progress        = min(current_samples / anneal_samples, 1.0) if anneal_samples > 0 else 1.0
            start_k         = float(self.bottleneck_dim)
            return start_k + progress * (target_k - start_k)
        else:
            return self.a * target_k

    def _apply_population_sparsity(self, activations: torch.Tensor, k: float) -> tuple[torch.Tensor, int]:
        k_count = min(max(1, int(k)), activations.shape[1])
        _, topk_idx = torch.topk(activations, k_count, dim=1)
        mask = torch.zeros_like(activations)
        mask.scatter_(1, topk_idx, 1)
        sparse = activations * mask
        if not self.training:
            self.last_filter_mask = mask.detach()
        return sparse

    def forward(
        self,
        x: torch.Tensor,
        epoch: int = 0,
        inputs_processed_in_epoch: int = 0,
    ) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)

        z1 = self.encoder(x)
        h1 = self.identity(z1)

        target_k = self.k_population * self.bottleneck_dim
        current_k = self._compute_annealed_k(
            epoch=epoch,
            inputs_processed_in_epoch=inputs_processed_in_epoch,
            target_k=float(target_k),
            training=self.training,
        )
        self.last_k = current_k

        h1 = self._apply_population_sparsity(h1, current_k)
        
        if not self.training:
            self.last_latent = h1.detach()
            
        z2 = self.decoder(h1)
        
        return z2, h1 # output, hidden
