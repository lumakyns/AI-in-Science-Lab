from typing import Any, NamedTuple

import torch
import torch.nn as nn


class FeatureMapEntry(NamedTuple):
    name: str
    feature_map: torch.Tensor
    conv: nn.Conv2d | None
    conv_input: torch.Tensor | None


class LayerCaptureMixin:
    inference_mode: bool
    layer_outputs: dict[str, torch.Tensor] | None

    def _reset_layer_outputs(self) -> None:
        self.layer_outputs = {} if self.inference_mode else None

    def _save_layer_output(self, name: str, output: torch.Tensor) -> None:
        if self.layer_outputs is not None:
            self.layer_outputs[name] = output.detach().cpu()


def get_num_classes(dataset: str) -> int:
    match dataset:
        case "cifar10":
            return 10
        case "cifar100":
            return 100
        case _:
            raise ValueError("dataset must be 'cifar10' or 'cifar100'.")


def get_model(cfg: dict[str, Any]) -> nn.Module:
    from .basic_cnn import BasicCNN1, BasicCNN2
    from .resnet import TorchvisionResNet18
    from .wta_conv_ae import WTA_CONV_AE

    num_classes = get_num_classes(cfg["dataset"])
    inference_mode = bool(cfg.get("inference_mode", False))
    architecture_type = cfg["architecture_type"]

    match architecture_type:
        case "cnn1":
            return BasicCNN1(num_classes=num_classes, inference_mode=inference_mode)
        case "cnn2":
            return BasicCNN2(num_classes=num_classes, inference_mode=inference_mode)
        case "resnet18":
            return TorchvisionResNet18(
                num_classes=num_classes,
                pretrained=False,
                freeze_backbone=False,
                inference_mode=inference_mode,
            )
        case "pretrained_resnet18":
            return TorchvisionResNet18(
                num_classes=num_classes,
                pretrained=True,
                freeze_backbone=True,
                inference_mode=inference_mode,
            )
        case "wta_conv_ae":
            return WTA_CONV_AE(
                dim=(3, 32, 32),
                hidden_ch=int(cfg.get("hidden_ch", 64)),
                k_spatial=float(cfg["k_spatial"]),
                k_population=cfg.get("k_population"),
                k_lifetime=cfg.get("k_lifetime"),
                total_epochs=int(cfg["epochs"]),
                dataset_size=int(cfg.get("dataset_size", 1)),
                a=float(cfg.get("wta_eval_multiplier", 1.0)),
                inference_mode=inference_mode,
            )
        case _:
            raise ValueError(f"Unknown architecture_type={architecture_type!r}.")
