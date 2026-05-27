from typing import Any, NamedTuple

import torch
import torch.nn as nn


class FeatureMapEntry(NamedTuple):
    name: str
    feature_map: torch.Tensor
    conv: nn.Conv2d | None
    conv_input: torch.Tensor | None


class LayerCaptureMixin:
    layer_outputs: dict[str, torch.Tensor] | None

    def _reset_layer_outputs(self) -> None:
        """Enable layer capture only while PyTorch inference mode is active."""
        self.layer_outputs = {} if torch.is_inference_mode_enabled() else None

    def _save_layer_output(self, name: str, output: torch.Tensor) -> None:
        """Store a detached CPU copy of a layer output when capture is enabled."""
        if self.layer_outputs is not None:
            self.layer_outputs[name] = output.detach().cpu()


def get_num_classes(dataset: str) -> int:
    """Map supported dataset names to their classifier output widths."""
    match dataset:
        case "cifar10":
            return 10
        case "cifar100":
            return 100
        case _:
            raise ValueError("dataset must be 'cifar10' or 'cifar100'.")


def _first_hidden_channel(cfg: dict[str, Any]) -> int:
    hidden_channels = cfg["hidden_channels"]
    if isinstance(hidden_channels, int):
        return int(hidden_channels)
    if not hidden_channels:
        raise ValueError("hidden_channels must contain at least one layer width.")
    return int(hidden_channels[0])


def get_model(cfg: dict[str, Any]) -> nn.Module:
    """Build the requested model from the experiment config dictionary."""
    from .basic_cnn import BasicCNN1, BasicCNN2
    from .greedy_stacked_autoencoder import GreedyStackedAutoencoder
    from .resnet import TorchvisionResNet18
    from .wta_conv_ae import WTA_CONV_AE

    num_classes = get_num_classes(cfg["dataset"])
    architecture_type = cfg["architecture_type"]

    match architecture_type:
        case "cnn1":
            return BasicCNN1(num_classes=num_classes)
        case "cnn2":
            return BasicCNN2(num_classes=num_classes)
        case "resnet18":
            return TorchvisionResNet18(
                num_classes=num_classes,
                pretrained=False,
                freeze_backbone=False,
            )
        case "pretrained_resnet18":
            return TorchvisionResNet18(
                num_classes=num_classes,
                pretrained=True,
                freeze_backbone=True,
            )
        case "wta_conv_ae":
            return WTA_CONV_AE(
                dim=(3, 32, 32),
                hidden_channels=_first_hidden_channel(cfg),
                k_spatial=cfg.get("k_spatial"),
                k_population=cfg.get("k_population"),
                k_lifetime=cfg.get("k_lifetime"),
                total_epochs=int(cfg["epochs"]),
                dataset_size=int(cfg.get("dataset_size", 1)),
                a=float(cfg.get("wta_eval_multiplier", 1.0)),
            )
        case "greedy_stacked_autoencoder":
            return GreedyStackedAutoencoder(
                dim=(3, 32, 32),
                hidden_channels=cfg["hidden_channels"],
                num_classes=num_classes if cfg["training_mode"] == "classification" else None,
                num_layers=int(cfg["num_layers"]),
                k_spatial=cfg.get("k_spatial"),
                k_population=cfg.get("k_population"),
                k_lifetime=cfg.get("k_lifetime"),
                total_epochs=int(cfg["epochs"]),
                dataset_size=int(cfg.get("dataset_size", 1)),
                a=float(cfg.get("wta_eval_multiplier", 1.0)),
                local_training=bool(cfg.get("local_training", False)),
            )
        case _:
            raise ValueError(f"Unknown architecture_type={architecture_type!r}.")
