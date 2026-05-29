import torch
import torch.nn as nn

from .common import LayerCaptureMixin
from .wta_conv_ae import WTA_CONV_AE

GSA_LAYER_SPEC = int | str


class GreedyStackedAutoencoder(LayerCaptureMixin, nn.Module):
    def __init__(
        self,
        dim: tuple[int, int, int] = (3, 32, 32),
        hidden_channels: int | list[GSA_LAYER_SPEC] | tuple[GSA_LAYER_SPEC, ...] = 64,
        num_classes: int | None = None,
        num_layers: int = 2,
        k_spatial: float | None = None,
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
            layer_specs: list[GSA_LAYER_SPEC] = [int(hidden_channels)] * int(num_layers)
        else:
            layer_specs = list(hidden_channels)
            if not layer_specs:
                raise ValueError("hidden_channels must contain at least one layer width.")

        self.layer_outputs = None
        self.in_ch, self.in_h, self.in_w = dim
        self.local_training = bool(local_training)
        self.is_convolutional = True
        self.last_reconstructions: list[torch.Tensor] = []
        self.last_reconstruction_targets: list[torch.Tensor] = []

        layers: list[nn.Module] = []
        layer_channels: list[int] = []
        in_ch = self.in_ch
        in_h = self.in_h
        in_w = self.in_w

        for spec in layer_specs:
            if isinstance(spec, str):
                token = spec.upper()
                if token == "BN":
                    if not layer_channels:
                        raise ValueError("'BN' must follow a numeric hidden channel width.")
                    layers.append(nn.BatchNorm2d(in_ch))
                elif token == "M":
                    if in_h < 2 or in_w < 2:
                        raise ValueError("'M' cannot downsample feature maps smaller than 2x2.")
                    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                    in_h //= 2
                    in_w //= 2
                else:
                    raise ValueError(f"Unknown GSA hidden_channels token {spec!r}; expected 'BN' or 'M'.")
                continue

            hidden_width = int(spec)
            if hidden_width <= 0:
                raise ValueError(f"hidden channel widths must be positive, got {hidden_width}")
            layers.append(
                WTA_CONV_AE(
                    dim=(in_ch, in_h, in_w),
                    hidden_channels=hidden_width,
                    k_spatial=k_spatial,
                    k_lifetime=k_lifetime,
                    k_population=k_population,
                    total_epochs=total_epochs,
                    dataset_size=dataset_size,
                    a=a,
                )
            )
            layer_channels.append(hidden_width)
            in_ch = hidden_width

        if not layer_channels:
            raise ValueError("hidden_channels must include at least one numeric layer width.")

        self.hidden_channels = layer_channels
        self.layer_specs = layer_specs
        self.num_layers = len(layer_channels)
        self.classifier = (
            nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(in_ch, int(num_classes)),
            )
            if num_classes is not None
            else None
        )
        self.layers = nn.ModuleList(layers)

    @property
    def detached_encoder_weights(self) -> list[torch.Tensor]:
        """Return all encoder weights detached from the gradient graph."""
        return [layer.detached_encoder_weights for layer in self.layers if isinstance(layer, WTA_CONV_AE)]

    @property
    def detached_decoder_weights(self) -> list[torch.Tensor]:
        """Return all decoder weights detached from the gradient graph."""
        return [layer.detached_decoder_weights for layer in self.layers if isinstance(layer, WTA_CONV_AE)]

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
        reconstruction_names: list[str] = []
        last_hidden = current

        for idx, layer in enumerate(self.layers):
            layer_name = f"greedy_stacked_autoencoder.layer{idx}"
            if isinstance(layer, nn.BatchNorm2d):
                current = layer(current)
                last_hidden = current
                self._save_layer_output(f"{layer_name}.bn", current)
                continue
            if isinstance(layer, nn.MaxPool2d):
                current = layer(current)
                last_hidden = current
                self._save_layer_output(f"{layer_name}.maxpool", current)
                continue

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
            reconstruction_names.append(layer_name)
            feature_maps.append((f"{layer_name}.hidden", hidden))

            self._save_layer_output(f"{layer_name}.input", current)
            self._save_layer_output(f"{layer_name}.reconstruction", reconstruction)
            self._save_layer_output(f"{layer_name}.hidden", hidden)

            current = hidden.detach() if self.local_training else hidden

        self.last_reconstructions = reconstructions
        self.last_reconstruction_targets = reconstruction_targets
        if self.local_training:
            for layer_name, reconstruction, target in zip(
                reconstruction_names,
                reconstructions,
                reconstruction_targets,
                strict=True,
            ):
                feature_maps.append((f"{layer_name}.reconstruction", reconstruction, target))
        if self.classifier is not None:
            logits = self.classifier(last_hidden)
            self._save_layer_output("greedy_stacked_autoencoder.classifier", logits)
            return logits, feature_maps
        if self.local_training:
            return list(zip(reconstructions, reconstruction_targets, strict=True)), feature_maps
        return reconstructions[0], feature_maps
