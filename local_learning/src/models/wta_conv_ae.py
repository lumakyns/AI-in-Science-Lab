import torch
import torch.nn as nn

from .common import LayerCaptureMixin


class WTA_CONV_AE(LayerCaptureMixin, nn.Module):
    def __init__(
        self,
        dim: tuple[int, int, int] = (3, 32, 32),
        hidden_ch: int = 64,
        k_spatial: float = 0.2,
        k_lifetime: float | None = 0.2,
        k_population: float | None = None,
        total_epochs: int = 1,
        dataset_size: int = 1,
        a: float = 1.0,
        inference_mode: bool = False,
    ) -> None:
        super().__init__()

        if k_lifetime is not None and k_population is not None:
            raise ValueError("Specify either k_lifetime or k_population, not both.")
        if k_lifetime is None and k_population is None:
            raise ValueError("Specify one of k_lifetime or k_population.")

        self.inference_mode = bool(inference_mode)
        self.layer_outputs = None
        self.in_ch, self.in_h, self.in_w = dim
        self.hidden_ch = int(hidden_ch)
        self.k_spatial = float(k_spatial)
        self.k_lifetime = k_lifetime
        self.k_population = k_population
        self.total_epochs = int(total_epochs)
        self.dataset_size = int(dataset_size)
        self.a = float(a)
        self.uses_k_population = k_population is not None
        self.use_population_sparsity = k_population is not None
        self.is_convolutional = True

        self.encoder = nn.Conv2d(self.in_ch, self.hidden_ch, kernel_size=5, padding=2, bias=False)
        self.decoder = nn.ConvTranspose2d(self.hidden_ch, self.in_ch, kernel_size=11, padding=5, bias=False)
        self.relu = nn.ReLU()

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
        if not training:
            return self.a * target_k

        current_samples = epoch * self.dataset_size + inputs_processed_in_epoch
        anneal_samples = (self.total_epochs // 2) * self.dataset_size
        progress = min(current_samples / anneal_samples, 1.0) if anneal_samples > 0 else 1.0
        start_k = float(self.hidden_ch)
        return start_k + progress * (target_k - start_k)

    def _apply_population_sparsity(self, activations: torch.Tensor, k: float) -> torch.Tensor:
        batch_size, channels, _, _ = activations.shape
        k_count = min(max(1, int(k)), channels)
        scores = activations.view(batch_size, channels, -1).sum(dim=2)
        _, topk_idx = torch.topk(scores, k_count, dim=1)
        mask = torch.zeros_like(scores)
        mask.scatter_(1, topk_idx, 1)
        sparse = activations * mask.view(batch_size, channels, 1, 1)
        if not self.training:
            self.last_filter_mask = mask.detach()
        return sparse

    def _apply_spatial_sparsity(self, activations: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = activations.shape
        k_count = max(1, int(self.k_spatial * height * width))
        k_count = min(k_count, height * width)
        flat = activations.view(batch_size, channels, -1)
        _, topk_idx = torch.topk(flat, k_count, dim=2)
        mask = torch.zeros_like(flat)
        mask.scatter_(2, topk_idx, 1)
        return activations * mask.view(batch_size, channels, height, width)

    def _apply_lifetime_sparsity(self, activations: torch.Tensor) -> torch.Tensor:
        if self.k_lifetime is None:
            raise RuntimeError("k_lifetime is required when population sparsity is disabled.")

        batch_size, channels, height, width = activations.shape
        k_count = max(1, int(self.k_lifetime * channels * height * width))
        k_count = min(k_count, channels * height * width)
        flat = activations.view(batch_size, -1)
        _, topk_idx = torch.topk(flat, k_count, dim=1)
        mask = torch.zeros_like(flat)
        mask.scatter_(1, topk_idx, 1)
        sparse = activations * mask.view(batch_size, channels, height, width)
        if not self.training:
            self.last_filter_mask = (sparse.abs().sum(dim=(2, 3)) > 0).to(sparse.dtype).detach()
        return sparse

    def forward(
        self,
        x: torch.Tensor,
        epoch: int = 0,
        inputs_processed_in_epoch: int = 0,
    ) -> tuple[torch.Tensor, list[tuple[str, torch.Tensor]]]:
        self._reset_layer_outputs()
        if x.ndim == 2:
            x = x.view(x.shape[0], self.in_ch, self.in_h, self.in_w)

        z1 = self.encoder(x)
        self._save_layer_output("wta_conv_ae.encoder", z1)
        h1 = self.relu(z1)
        self._save_layer_output("wta_conv_ae.relu", h1)

        h1 = self._apply_spatial_sparsity(h1)
        self._save_layer_output("wta_conv_ae.spatial_wta", h1)
        if self.use_population_sparsity:
            if self.k_population is None:
                raise RuntimeError("k_population is required when population sparsity is enabled.")
            target_k = self.k_population * self.hidden_ch
            current_k = self._compute_annealed_k(
                epoch=epoch,
                inputs_processed_in_epoch=inputs_processed_in_epoch,
                target_k=float(target_k),
                training=self.training,
            )
            self.last_k = min(max(1, int(current_k)), self.hidden_ch)
            h1 = self._apply_population_sparsity(h1, current_k)
        else:
            h1 = self._apply_lifetime_sparsity(h1)
        self._save_layer_output("wta_conv_ae.hidden", h1)

        reconstruction = self.decoder(h1)
        self._save_layer_output("wta_conv_ae.decoder", reconstruction)
        return reconstruction, [("wta_conv_ae.hidden", h1)]

