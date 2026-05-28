import torch
import torch.nn as nn
from torchvision.models import VGG16_Weights, vgg16

from .common import FeatureMapEntry, LayerCaptureMixin


class TorchvisionVGG16(LayerCaptureMixin, nn.Module):
    def __init__(
        self,
        num_classes: int,
        *,
        pretrained: bool = False,
        freeze_backbone: bool = False,
    ) -> None:
        """Wrap torchvision VGG-16 with optional pretrained frozen features and capture hooks."""
        super().__init__()
        self.layer_outputs = None

        weights = VGG16_Weights.DEFAULT if pretrained else None
        self.model = vgg16(weights=weights)
        in_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(in_features, num_classes)

        if freeze_backbone:
            for name, param in self.model.named_parameters():
                if not name.startswith("classifier.6."):
                    param.requires_grad = False

        self._hook_names = {
            "features.0",
            "features.2",
            "features.5",
            "features.7",
            "features.10",
            "features.12",
            "features.14",
            "features.17",
            "features.19",
            "features.21",
            "features.24",
            "features.26",
            "features.28",
            "avgpool",
            "classifier.6",
        }
        self._hook_handles = []
        self._forward_feature_maps: list[tuple[str, torch.Tensor]] = []
        self._register_capture_hooks()

    def _register_capture_hooks(self) -> None:
        """Attach forward hooks to selected VGG modules for logging feature maps."""
        for name, module in self.model.named_modules():
            if name in self._hook_names:
                self._hook_handles.append(module.register_forward_hook(self._make_capture_hook(name)))

    def _make_capture_hook(self, name: str):
        """Create a hook that records tensor outputs and exposes 4D maps for losses."""
        def hook(_module, _inputs, output) -> None:
            """Save the hooked module output when it is a tensor."""
            if isinstance(output, torch.Tensor):
                self._save_layer_output(f"vgg16.{name}", output)
                if output.ndim == 4:
                    self._forward_feature_maps.append((f"vgg16.{name}", output))

        return hook

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[FeatureMapEntry | tuple[str, torch.Tensor]]]:
        """Run VGG-16 and return logits plus the hook-collected feature maps."""
        self._reset_layer_outputs()
        self._forward_feature_maps = []
        logits = self.model(x)
        return logits, self._forward_feature_maps
