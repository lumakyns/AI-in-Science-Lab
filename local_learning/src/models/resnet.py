import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from .common import FeatureMapEntry, LayerCaptureMixin


class TorchvisionResNet18(LayerCaptureMixin, nn.Module):
    def __init__(
        self,
        num_classes: int,
        *,
        pretrained: bool = False,
        freeze_backbone: bool = False,
        inference_mode: bool = False,
    ) -> None:
        super().__init__()
        self.inference_mode = bool(inference_mode)
        self.layer_outputs = None

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.model = resnet18(weights=weights)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

        if freeze_backbone:
            for name, param in self.model.named_parameters():
                if not name.startswith("fc."):
                    param.requires_grad = False

        self._hook_names = {
            "conv1",
            "bn1",
            "relu",
            "maxpool",
            "layer1.0",
            "layer1.1",
            "layer2.0",
            "layer2.1",
            "layer3.0",
            "layer3.1",
            "layer4.0",
            "layer4.1",
            "avgpool",
            "fc",
        }
        self._hook_handles = []
        self._forward_feature_maps: list[tuple[str, torch.Tensor]] = []
        self._register_capture_hooks()

    def _register_capture_hooks(self) -> None:
        for name, module in self.model.named_modules():
            if name in self._hook_names:
                self._hook_handles.append(module.register_forward_hook(self._make_capture_hook(name)))

    def _make_capture_hook(self, name: str):
        def hook(_module, _inputs, output) -> None:
            if isinstance(output, torch.Tensor):
                self._save_layer_output(f"resnet18.{name}", output)
                if output.ndim == 4 and name not in {"maxpool"}:
                    self._forward_feature_maps.append((f"resnet18.{name}", output))

        return hook

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[FeatureMapEntry | tuple[str, torch.Tensor]]]:
        self._reset_layer_outputs()
        self._forward_feature_maps = []
        logits = self.model(x)
        return logits, self._forward_feature_maps
