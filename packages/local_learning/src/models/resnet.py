import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from .common import FeatureMapEntry, LayerCaptureMixin, load_torchvision_state_dict


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample_conv: nn.Conv2d | None = None,
        downsample_bn: nn.BatchNorm2d | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample_conv = downsample_conv
        self.downsample_bn = downsample_bn
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample_conv is not None and self.downsample_bn is not None:
            identity = self.downsample_conv(x)
            identity = self.downsample_bn(identity)

        out += identity
        return self.relu(out)


class TorchvisionResNet18(LayerCaptureMixin, nn.Module):
    def __init__(
        self,
        num_classes: int,
        *,
        pretrained: bool = False,
        freeze_backbone: bool = False,
        zero_init_residual: bool = False,
    ) -> None:
        """Local ResNet-18 layout with optional torchvision weight transfer and capture hooks."""
        super().__init__()
        self.layer_outputs = None

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1_0 = BasicBlock(64, 64)
        self.layer1_1 = BasicBlock(64, 64)

        self.layer2_0 = BasicBlock(
            64,
            128,
            stride=2,
            downsample_conv=nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
            downsample_bn=nn.BatchNorm2d(128),
        )
        self.layer2_1 = BasicBlock(128, 128)

        self.layer3_0 = BasicBlock(
            128,
            256,
            stride=2,
            downsample_conv=nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False),
            downsample_bn=nn.BatchNorm2d(256),
        )
        self.layer3_1 = BasicBlock(256, 256)

        self.layer4_0 = BasicBlock(
            256,
            512,
            stride=2,
            downsample_conv=nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False),
            downsample_bn=nn.BatchNorm2d(512),
        )
        self.layer4_1 = BasicBlock(512, 512)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

        self._initialize_weights(zero_init_residual)
        if pretrained:
            self._load_torchvision_weights()

        if freeze_backbone:
            for name, param in self.named_parameters():
                if not name.startswith("fc."):
                    param.requires_grad = False

        self._hook_modules = (
            ("conv1", self.conv1),
            ("bn1", self.bn1),
            ("relu", self.relu),
            ("maxpool", self.maxpool),
            ("layer1.0", self.layer1_0),
            ("layer1.1", self.layer1_1),
            ("layer2.0", self.layer2_0),
            ("layer2.1", self.layer2_1),
            ("layer3.0", self.layer3_0),
            ("layer3.1", self.layer3_1),
            ("layer4.0", self.layer4_0),
            ("layer4.1", self.layer4_1),
            ("avgpool", self.avgpool),
            ("fc", self.fc),
        )
        self._hook_handles = []
        self._forward_feature_maps: list[tuple[str, torch.Tensor]] = []
        self._register_capture_hooks()

    def _initialize_weights(self, zero_init_residual: bool) -> None:
        """Match torchvision ResNet initialization before any optional weight transfer."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

        if zero_init_residual:
            for module in self.modules():
                if isinstance(module, BasicBlock) and module.bn2.weight is not None:
                    nn.init.constant_(module.bn2.weight, 0)

    def _load_torchvision_weights(self) -> None:
        """Copy ResNet-18 weights from torchvision, keeping this model's final layer shape."""
        source = resnet18(weights=None)
        source.load_state_dict(
            load_torchvision_state_dict("resnet18", ResNet18_Weights.DEFAULT)
        )
        source_to_local = (
            (source.conv1, self.conv1),
            (source.bn1, self.bn1),
            (source.layer1[0], self.layer1_0),
            (source.layer1[1], self.layer1_1),
            (source.layer2[0], self.layer2_0),
            (source.layer2[1], self.layer2_1),
            (source.layer3[0], self.layer3_0),
            (source.layer3[1], self.layer3_1),
            (source.layer4[0], self.layer4_0),
            (source.layer4[1], self.layer4_1),
            (source.fc, self.fc),
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
        if not isinstance(local_module, BasicBlock) or source_module.downsample is None:
            return

        local_module.downsample_conv.weight.data.copy_(source_module.downsample[0].weight.data)
        local_module.downsample_bn.weight.data.copy_(source_module.downsample[1].weight.data)
        local_module.downsample_bn.bias.data.copy_(source_module.downsample[1].bias.data)
        local_module.downsample_bn.running_mean.data.copy_(source_module.downsample[1].running_mean.data)
        local_module.downsample_bn.running_var.data.copy_(source_module.downsample[1].running_var.data)
        local_module.downsample_bn.num_batches_tracked.data.copy_(
            source_module.downsample[1].num_batches_tracked.data
        )

    def _register_capture_hooks(self) -> None:
        """Attach forward hooks to selected ResNet modules for logging feature maps."""
        for name, module in self._hook_modules:
            self._hook_handles.append(module.register_forward_hook(self._make_capture_hook(name)))

    def _make_capture_hook(self, name: str):
        """Create a hook that records tensor outputs and exposes 4D maps for losses."""
        def hook(_module, _inputs, output) -> None:
            """Save the hooked module output when it is a tensor."""
            if isinstance(output, torch.Tensor):
                self._save_layer_output(f"resnet18.{name}", output)
                if output.ndim == 4 and name not in {"maxpool"}:
                    self._forward_feature_maps.append((f"resnet18.{name}", output))

        return hook

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[FeatureMapEntry | tuple[str, torch.Tensor]]]:
        """Run ResNet-18 and return logits plus the hook-collected feature maps."""
        self._reset_layer_outputs()
        self._forward_feature_maps = []

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1_0(x)
        x = self.layer1_1(x)
        x = self.layer2_0(x)
        x = self.layer2_1(x)
        x = self.layer3_0(x)
        x = self.layer3_1(x)
        x = self.layer4_0(x)
        x = self.layer4_1(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.fc(x)
        return logits, self._forward_feature_maps
