import torch
import torch.nn as nn


class WTA_CONV_AE(nn.Module):
    def __init__(
        self,
        dim: tuple,
        k_spatial: float,
        k_lifetime: float | None = None,
        k_population: float | None = None,
        total_epochs: int = 1,
        dataset_size: int = 1,
        a: float = 1.0,
    ) -> None:
        super().__init__()
        self.in_ch, self.in_h, self.in_w, self.hidden_ch = dim
        self.k_spatial = k_spatial
        self.total_epochs = total_epochs
        self.dataset_size = dataset_size
        self.a = a
        self.uses_k_population = True

        # population sparsity is offered as an option in WTA_CONV so we keep things biologically plausible
        if k_lifetime is not None and k_population is not None:
            raise ValueError("Specify either k_lifetime or k_population, not both.")
        if k_lifetime is None and k_population is None:
            raise ValueError("Specify one of k_lifetime or k_population.")

        self.use_population_sparsity = k_population is not None
        self.k_lifetime   = k_lifetime
        self.k_population = k_population

        self.encoder = nn.Conv2d(self.in_ch, self.hidden_ch, kernel_size=5, padding=2, bias=False)
        self.decoder = nn.ConvTranspose2d(self.hidden_ch, self.in_ch, kernel_size=11, padding=5, bias=False)
        self.relu    = nn.ReLU()

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
            start_k         = float(self.hidden_ch)
            return start_k + progress * (target_k - start_k)
        else:
            return self.a * target_k

    def _apply_population_sparsity(self, activations: torch.Tensor, k: float) -> torch.Tensor:
        B, C, H, W = activations.shape
        k_count = min(max(1, int(k)), activations.shape[1])
        scores  = activations.view(B, C, -1).sum(dim=2)
        _, topk_idx = torch.topk(scores, k_count, dim=1)
        mask = torch.zeros_like(scores)
        mask.scatter_(1, topk_idx, 1)
        sparse = activations * mask.view(B, C, 1, 1)
        if not self.training:
            self.last_filter_mask = mask.detach()
        return sparse

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
        sparse = activations * mask.view(B, C, H, W)
        if not self.training:
            self.last_filter_mask = (sparse.abs().sum(dim=(2, 3)) > 0).to(sparse.dtype).detach()
        return sparse

    def forward(
        self,
        x: torch.Tensor,
        epoch: int = 0,
        inputs_processed_in_epoch: int = 0,
    ) -> torch.Tensor:
        if x.ndim == 2:
            x = x.view(x.shape[0], self.in_ch, self.in_h, self.in_w)

        z1 = self.encoder(x)
        a1 = self.relu(z1)

        a1 = self._apply_spatial_sparsity(a1)
        if self.use_population_sparsity:
            target_k = self.k_population * self.hidden_ch
            current_k = self._compute_annealed_k(
                epoch=epoch,
                inputs_processed_in_epoch=inputs_processed_in_epoch,
                target_k=float(target_k),
                training=self.training,
            )
            self.last_k = min(max(1, int(current_k)), self.hidden_ch)
            a1 = self._apply_population_sparsity(a1, current_k)
        else:
            a1 = self._apply_lifetime_sparsity(a1)

        if not self.training:
            self.last_latent = a1.detach()

        z2 = self.decoder(a1)
        return z2.view(z2.shape[0], -1)
