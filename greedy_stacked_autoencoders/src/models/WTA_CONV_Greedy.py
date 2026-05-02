from __future__ import annotations

import torch
import torch.nn as nn


class WTA_CONV_Greedy(nn.Module):
    """Two-layer convolutional WTA autoencoder for CIFAR-10 color patches."""

    def __init__(
        self,
        dim: tuple[int, int, int],
        hidden_channels: tuple[int, int] = (64, 128),
        k_spatial: float = 0.2,
        k_lifetime: float | None = None,
        k_population: float | None = 0.1,
        total_epochs: int = 1,
        dataset_size: int = 1,
        a: float = 1.0,
        feature_map_mode: str = "post_wta",
    ) -> None:
        super().__init__()

        if k_lifetime is not None and k_population is not None:
            raise ValueError("Specify either k_lifetime or k_population, not both.")
        if k_lifetime is None and k_population is None:
            raise ValueError("Specify one of k_lifetime or k_population.")
        if feature_map_mode not in {"pre_wta", "post_wta", "both"}:
            raise ValueError("feature_map_mode must be 'pre_wta', 'post_wta', or 'both'.")

        self.in_ch, self.in_h, self.in_w = dim
        self.hidden_ch1, self.hidden_ch2 = hidden_channels
        self.k_spatial = k_spatial
        self.k_lifetime = k_lifetime
        self.k_population = k_population
        self.total_epochs = total_epochs
        self.dataset_size = dataset_size
        self.a = a
        self.feature_map_mode = feature_map_mode

        self.uses_k_population = k_population is not None
        self.use_population_sparsity = k_population is not None
        self.is_convolutional = True

        self.encoder1 = nn.Conv2d(
            self.in_ch,
            self.hidden_ch1,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.encoder2 = nn.Conv2d(
            self.hidden_ch1,
            self.hidden_ch2,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.decoder1 = nn.ConvTranspose2d(
            self.hidden_ch2,
            self.hidden_ch1,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )
        self.decoder2 = nn.ConvTranspose2d(
            self.hidden_ch1,
            self.in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.relu = nn.ReLU()
        self.last_filter_mask = torch.ones(1, self.hidden_ch2)

    @property
    def detached_encoder_weights(self) -> torch.Tensor:
        return self.encoder1.weight.detach()

    @property
    def detached_decoder_weights(self) -> torch.Tensor:
        return self.decoder2.weight.detach()

    def _compute_annealed_k(
        self,
        *,
        epoch: int,
        inputs_processed_in_epoch: int,
        target_k: float,
        channels: int,
        training: bool,
    ) -> float:
        if not training:
            return self.a * target_k

        current_samples = epoch * self.dataset_size + inputs_processed_in_epoch
        anneal_samples = (self.total_epochs // 2) * self.dataset_size
        progress = min(current_samples / anneal_samples, 1.0) if anneal_samples > 0 else 1.0
        start_k = float(channels)
        return start_k + progress * (target_k - start_k)

    def _apply_spatial_sparsity(self, activations: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = activations.shape
        del batch_size, channels

        k_count = max(1, int(self.k_spatial * height * width))
        k_count = min(k_count, height * width)
        flat = activations.view(activations.shape[0], activations.shape[1], -1)
        _, topk_idx = torch.topk(flat, k_count, dim=2)
        mask = torch.zeros_like(flat)
        mask.scatter_(2, topk_idx, 1)
        return activations * mask.view_as(activations)

    def _apply_population_sparsity(
        self,
        activations: torch.Tensor,
        *,
        k: float,
        save_mask: bool,
    ) -> torch.Tensor:
        batch_size, channels, height, width = activations.shape
        del height, width

        k_count = min(max(1, int(k)), channels)
        scores = activations.view(batch_size, channels, -1).sum(dim=2)
        _, topk_idx = torch.topk(scores, k_count, dim=1)
        mask = torch.zeros_like(scores)
        mask.scatter_(1, topk_idx, 1)
        sparse = activations * mask.view(batch_size, channels, 1, 1)
        if save_mask:
            self.last_filter_mask = mask.detach()
        return sparse

    def _apply_lifetime_sparsity(
        self,
        activations: torch.Tensor,
        *,
        save_mask: bool,
    ) -> torch.Tensor:
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
        if save_mask:
            self.last_filter_mask = (sparse.abs().sum(dim=(2, 3)) > 0).to(sparse.dtype).detach()
        return sparse

    def _apply_wta(
        self,
        activations: torch.Tensor,
        *,
        epoch: int,
        inputs_processed_in_epoch: int,
        save_mask: bool,
    ) -> torch.Tensor:
        sparse = self._apply_spatial_sparsity(activations)

        if self.use_population_sparsity:
            if self.k_population is None:
                raise RuntimeError("k_population is required when population sparsity is enabled.")

            target_k = self.k_population * activations.shape[1]
            current_k = self._compute_annealed_k(
                epoch=epoch,
                inputs_processed_in_epoch=inputs_processed_in_epoch,
                target_k=target_k,
                channels=activations.shape[1],
                training=self.training,
            )
            self.last_k = min(max(1, int(current_k)), activations.shape[1])
            return self._apply_population_sparsity(
                sparse,
                k=current_k,
                save_mask=save_mask,
            )

        return self._apply_lifetime_sparsity(sparse, save_mask=save_mask)

    def _feature_maps(
        self,
        *,
        z1: torch.Tensor,
        h1: torch.Tensor,
        z2: torch.Tensor,
        h2: torch.Tensor,
    ) -> list[tuple[str, torch.Tensor]]:
        maps: list[tuple[str, torch.Tensor]] = []
        if self.feature_map_mode in {"pre_wta", "both"}:
            maps.extend([
                ("encoder1.pre_wta", z1),
                ("encoder2.pre_wta", z2),
            ])
        if self.feature_map_mode in {"post_wta", "both"}:
            maps.extend([
                ("encoder1.post_wta", h1),
                ("encoder2.post_wta", h2),
            ])
        return maps

    def forward(
        self,
        x: torch.Tensor,
        epoch: int = 0,
        inputs_processed_in_epoch: int = 0,
    ) -> tuple[torch.Tensor, list[tuple[str, torch.Tensor]]]:
        if x.ndim == 2:
            x = x.view(x.shape[0], self.in_ch, self.in_h, self.in_w)

        z1 = self.encoder1(x)
        a1 = self.relu(z1)
        h1 = self._apply_wta(
            a1,
            epoch=epoch,
            inputs_processed_in_epoch=inputs_processed_in_epoch,
            save_mask=False,
        )

        z2 = self.encoder2(h1)
        a2 = self.relu(z2)
        h2 = self._apply_wta(
            a2,
            epoch=epoch,
            inputs_processed_in_epoch=inputs_processed_in_epoch,
            save_mask=not self.training,
        )

        recon = self.relu(self.decoder1(h2))
        recon = self.decoder2(recon)
        recon = recon.view(recon.shape[0], -1)

        return recon, self._feature_maps(z1=z1, h1=h1, z2=z2, h2=h2)
