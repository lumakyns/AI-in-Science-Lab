import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import DenseNet121_Weights, densenet121
from torchvision.models.densenet import _DenseBlock, _Transition

from .common import FeatureMapEntry, LayerCaptureMixin, load_torchvision_state_dict


class TorchvisionDenseNet121(LayerCaptureMixin, nn.Module):
    def __init__(
        self,
        num_classes: int,
        *,
        pretrained: bool = False,
        freeze_backbone: bool = False,
    ) -> None:
        """Local DenseNet-121 layout with optional torchvision weight transfer and capture hooks."""
        super().__init__()
        self.layer_outputs = None

        self.conv0 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm0 = nn.BatchNorm2d(64)
        self.relu0 = nn.ReLU(inplace=True)
        self.pool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.denseblock1 = _DenseBlock(
            num_layers=6,
            num_input_features=64,
            bn_size=4,
            growth_rate=32,
            drop_rate=0.0,
        )
        self.transition1 = _Transition(num_input_features=256, num_output_features=128)
        self.denseblock2 = _DenseBlock(
            num_layers=12,
            num_input_features=128,
            bn_size=4,
            growth_rate=32,
            drop_rate=0.0,
        )
        self.transition2 = _Transition(num_input_features=512, num_output_features=256)
        self.denseblock3 = _DenseBlock(
            num_layers=24,
            num_input_features=256,
            bn_size=4,
            growth_rate=32,
            drop_rate=0.0,
        )
        self.transition3 = _Transition(num_input_features=1024, num_output_features=512)
        self.denseblock4 = _DenseBlock(
            num_layers=16,
            num_input_features=512,
            bn_size=4,
            growth_rate=32,
            drop_rate=0.0,
        )
        self.norm5 = nn.BatchNorm2d(1024)
        self.classifier = nn.Linear(1024, num_classes)

        self._initialize_weights()
        if pretrained:
            self._load_torchvision_weights()

        if freeze_backbone:
            for name, param in self.named_parameters():
                if not name.startswith("classifier."):
                    param.requires_grad = False

        self._hook_modules = (
            ("features.conv0", self.conv0),
            ("features.norm0", self.norm0),
            ("features.relu0", self.relu0),
            ("features.pool0", self.pool0),
            ("features.denseblock1", self.denseblock1),
            ("features.transition1", self.transition1),
            ("features.denseblock2", self.denseblock2),
            ("features.transition2", self.transition2),
            ("features.denseblock3", self.denseblock3),
            ("features.transition3", self.transition3),
            ("features.denseblock4", self.denseblock4),
            ("features.norm5", self.norm5),
            ("classifier", self.classifier),
        )
        self._hook_handles = []
        self._forward_feature_maps: list[tuple[str, torch.Tensor]] = []
        self._register_capture_hooks()

    def _initialize_weights(self) -> None:
        """Match torchvision DenseNet initialization before any optional weight transfer."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.constant_(module.bias, 0)

    def _load_torchvision_weights(self) -> None:
        """Copy DenseNet-121 weights from torchvision, keeping this model's final layer shape."""
        source = densenet121(weights=None)
        source.load_state_dict(
            load_torchvision_state_dict("densenet121", DenseNet121_Weights.DEFAULT)
        )
        source_to_local = (
            (source.features.conv0, self.conv0),
            (source.features.norm0, self.norm0),
            (source.features.denseblock1, self.denseblock1),
            (source.features.transition1, self.transition1),
            (source.features.denseblock2, self.denseblock2),
            (source.features.transition2, self.transition2),
            (source.features.denseblock3, self.denseblock3),
            (source.features.transition3, self.transition3),
            (source.features.denseblock4, self.denseblock4),
            (source.features.norm5, self.norm5),
            (source.classifier, self.classifier),
        )
        for source_module, local_module in source_to_local:
            self._copy_matching_state(source_module, local_module)

    def _copy_matching_state(self, source_module: nn.Module, local_module: nn.Module) -> None:
        local_state = local_module.state_dict()
        source_state = {
            name: value
            for name, value in source_module.state_dict().items()
            if name in local_state and local_state[name].shape == value.shape
        }
        local_module.load_state_dict(source_state, strict=False)

    def _register_capture_hooks(self) -> None:
        """Attach forward hooks to selected DenseNet modules for logging feature maps."""
        for name, module in self._hook_modules:
            self._hook_handles.append(module.register_forward_hook(self._make_capture_hook(name)))

    def _make_capture_hook(self, name: str):
        """Create a hook that records tensor outputs and exposes 4D maps for losses."""
        def hook(_module, _inputs, output) -> None:
            """Save the hooked module output when it is a tensor."""
            if isinstance(output, torch.Tensor):
                self._save_layer_output(f"densenet121.{name}", output)
                if output.ndim == 4 and name != "features.pool0":
                    self._forward_feature_maps.append((f"densenet121.{name}", output))

        return hook

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[FeatureMapEntry | tuple[str, torch.Tensor]]]:
        """Run DenseNet-121 and return logits plus the hook-collected feature maps."""
        self._reset_layer_outputs()
        self._forward_feature_maps = []

        x = self.conv0(x)
        x = self.norm0(x)
        x = self.relu0(x)
        x = self.pool0(x)
        x = self.denseblock1(x)
        x = self.transition1(x)
        x = self.denseblock2(x)
        x = self.transition2(x)
        x = self.denseblock3(x)
        x = self.transition3(x)
        x = self.denseblock4(x)
        x = self.norm5(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits, self._forward_feature_maps
