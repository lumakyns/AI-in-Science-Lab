import torch
import torch.nn as nn

from .common import LayerCaptureMixin
from .wta_conv_ae import WTA_CONV_AE


class GreedyStackedAutoencoder(LayerCaptureMixin, nn.Module):
    def __init__(
        self,
        dim: tuple[int, int, int] = (3, 32, 32),
        hidden_channels: int | list[int] | tuple[int, ...] = 64,
        num_classes: int | None = None,
        num_layers: int = 2,
        k_spatial: float = 0.2,
        k_lifetime: float | None = 0.2,
        k_population: float | None = None,
        total_epochs: int = 1,
        dataset_size: int = 1,
        a: float = 1.0,
        local_training: bool = False,
    ) -> None:
        """Create a chain of WTA autoencoders that pass each sparse code to the next layer."""
        super().__init__()

        if isinstance(hidden_channels, int):
            if num_layers <= 0:
                raise ValueError(f"num_layers must be positive, got {num_layers}")
            layer_channels = [int(hidden_channels)] * int(num_layers)
        else:
            layer_channels = [int(ch) for ch in hidden_channels]
            if not layer_channels:
                raise ValueError("hidden_channels must contain at least one layer width.")

        self.layer_outputs = None
        self.in_ch, self.in_h, self.in_w = dim
        self.hidden_channels = layer_channels
        self.num_layers = len(layer_channels)
        self.local_training = bool(local_training)
        self.is_convolutional = True
        self.last_reconstructions: list[torch.Tensor] = []
        self.last_reconstruction_targets: list[torch.Tensor] = []
        self.classifier = (
            nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(layer_channels[-1], int(num_classes)),
            )
            if num_classes is not None
            else None
        )

        layers: list[WTA_CONV_AE] = []
        in_ch = self.in_ch
        for hidden_width in layer_channels:
            layers.append(
                WTA_CONV_AE(
                    dim=(in_ch, self.in_h, self.in_w),
                    hidden_channels=hidden_width,
                    k_spatial=k_spatial,
                    k_lifetime=k_lifetime,
                    k_population=k_population,
                    total_epochs=total_epochs,
                    dataset_size=dataset_size,
                    a=a,
                )
            )
            in_ch = hidden_width
        self.layers = nn.ModuleList(layers)

    @property
    def detached_encoder_weights(self) -> list[torch.Tensor]:
        """Return all encoder weights detached from the gradient graph."""
        return [layer.detached_encoder_weights for layer in self.layers]

    @property
    def detached_decoder_weights(self) -> list[torch.Tensor]:
        """Return all decoder weights detached from the gradient graph."""
        return [layer.detached_decoder_weights for layer in self.layers]

    def forward(
        self,
        x: torch.Tensor,
        epoch: int = 0,
        inputs_processed_in_epoch: int = 0,
    ) -> tuple[torch.Tensor | list[tuple[torch.Tensor, torch.Tensor]], list]:
        """Reconstruct each layer input and feed each sparse hidden code into the next block."""
        self._reset_layer_outputs()
        if x.ndim == 2:
            x = x.view(x.shape[0], self.in_ch, self.in_h, self.in_w)

        current = x
        feature_maps: list = []
        reconstructions: list[torch.Tensor] = []
        reconstruction_targets: list[torch.Tensor] = []
        last_hidden = current

        for idx, layer in enumerate(self.layers):
            target = current
            reconstruction, layer_features = layer(
                current,
                epoch=epoch,
                inputs_processed_in_epoch=inputs_processed_in_epoch,
            )
            hidden = layer_features[0][1]
            last_hidden = hidden

            reconstructions.append(reconstruction)
            reconstruction_targets.append(target)
            feature_maps.append((f"greedy_stacked_autoencoder.layer{idx}.hidden", hidden))

            self._save_layer_output(f"greedy_stacked_autoencoder.layer{idx}.input", current)
            self._save_layer_output(f"greedy_stacked_autoencoder.layer{idx}.reconstruction", reconstruction)
            self._save_layer_output(f"greedy_stacked_autoencoder.layer{idx}.hidden", hidden)

            current = hidden.detach() if self.local_training else hidden

        self.last_reconstructions = reconstructions
        self.last_reconstruction_targets = reconstruction_targets
        if self.local_training:
            for idx, (reconstruction, target) in enumerate(zip(reconstructions, reconstruction_targets, strict=True)):
                feature_maps.append((f"greedy_stacked_autoencoder.layer{idx}.reconstruction", reconstruction, target))
        if self.classifier is not None:
            logits = self.classifier(last_hidden)
            self._save_layer_output("greedy_stacked_autoencoder.classifier", logits)
            return logits, feature_maps
        if self.local_training:
            return list(zip(reconstructions, reconstruction_targets, strict=True)), feature_maps
        return reconstructions[0], feature_maps
