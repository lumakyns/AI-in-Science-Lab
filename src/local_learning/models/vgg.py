import torch
import torch.nn as nn
from torchvision.models import VGG16_Weights, vgg16

from .common import FeatureMapEntry, LayerCaptureMixin


class TorchvisionVGG16(LayerCaptureMixin, nn.Module):
    def __init__(
        self,
        num_classes: int,
        *,
        dataset: str = "cifar10",
        pretrained: bool = False,
        freeze_backbone: bool = False,
        deconv_training: bool = False,
        local_training: bool = False,
        small: bool = False,
    ) -> None:
        """Local VGG-16 layout with optional torchvision weight transfer and capture hooks."""
        super().__init__()
        self.layer_outputs = None
        self.dataset = dataset
        self.deconv_training = bool(deconv_training)
        self.local_training = bool(local_training)
        self.small = bool(small)

        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.deconv1_1 = nn.ConvTranspose2d(64, 3, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.deconv1_2 = nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.deconv2_1 = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.deconv2_2 = nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.deconv3_1 = nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.deconv3_2 = nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.deconv3_3 = nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.deconv4_1 = nn.ConvTranspose2d(512, 256, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.deconv4_2 = nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.deconv4_3 = nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.deconv5_1 = nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.deconv5_2 = nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.deconv5_3 = nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        classifier_in_features = (256 if self.small else 512) * 7 * 7
        self.classifier_fc1 = nn.Linear(classifier_in_features, 4096)
        self.classifier_relu1 = nn.ReLU(True)
        self.classifier_dropout1 = nn.Dropout()
        self.classifier_fc2 = nn.Linear(4096, 4096)
        self.classifier_relu2 = nn.ReLU(True)
        self.classifier_dropout2 = nn.Dropout()
        self.classifier_fc3 = nn.Linear(4096, num_classes)

        self._initialize_weights()
        if pretrained and self.small:
            raise ValueError("pretrained_vgg16 does not support vgg16_small=True.")
        if pretrained:
            self._load_torchvision_weights()

        if freeze_backbone:
            for name, param in self.named_parameters():
                if not (name.startswith("classifier_fc3.") or name.startswith("deconv")):
                    param.requires_grad = False

        self._hook_modules = (
            ("features.0", self.conv1_1),
            ("features.2", self.conv1_2),
            ("features.5", self.conv2_1),
            ("features.7", self.conv2_2),
            ("features.10", self.conv3_1),
            ("features.12", self.conv3_2),
            ("features.14", self.conv3_3),
            ("features.17", self.conv4_1),
            ("features.19", self.conv4_2),
            ("features.21", self.conv4_3),
            ("features.24", self.conv5_1),
            ("features.26", self.conv5_2),
            ("features.28", self.conv5_3),
            ("avgpool", self.avgpool),
            ("classifier.6", self.classifier_fc3),
        )
        self._hook_handles = []
        self._forward_feature_maps: list[tuple[str, torch.Tensor]] = []
        self._register_capture_hooks()

    def _initialize_weights(self) -> None:
        """Match torchvision VGG initialization before any optional weight transfer."""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)

    def _load_torchvision_weights(self) -> None:
        """Copy dataset-selected VGG-16 weights, keeping this model's final layer shape."""
        source = self._source_vgg16_for_dataset()
        source_to_local = (
            (source.features[0], self.conv1_1),
            (source.features[2], self.conv1_2),
            (source.features[5], self.conv2_1),
            (source.features[7], self.conv2_2),
            (source.features[10], self.conv3_1),
            (source.features[12], self.conv3_2),
            (source.features[14], self.conv3_3),
            (source.features[17], self.conv4_1),
            (source.features[19], self.conv4_2),
            (source.features[21], self.conv4_3),
            (source.features[24], self.conv5_1),
            (source.features[26], self.conv5_2),
            (source.features[28], self.conv5_3),
            (source.classifier[0], self.classifier_fc1),
            (source.classifier[3], self.classifier_fc2),
            (source.classifier[6], self.classifier_fc3),
        )
        for source_module, local_module in source_to_local:
            if source_module.weight.shape == local_module.weight.shape:
                local_module.weight.data.copy_(source_module.weight.data)
            if source_module.bias.shape == local_module.bias.shape:
                local_module.bias.data.copy_(source_module.bias.data)

    def _source_vgg16_for_dataset(self) -> nn.Module:
        """Select the pretrained VGG-16 source implied by the configured dataset."""
        base_dataset = self.dataset.removesuffix("_patches")
        match base_dataset:
            case "imagenet":
                return vgg16(weights=VGG16_Weights.DEFAULT)
            case "cifar10" | "smallcifar10" | "cifar100":
                raise ValueError(
                    f"pretrained_vgg16 requested for {base_dataset!r}, but no built-in "
                    "CIFAR VGG-16 weight source is configured yet."
                )
            case _:
                raise ValueError(f"Unsupported VGG-16 pretrained dataset source: {self.dataset!r}.")

    def _register_capture_hooks(self) -> None:
        """Attach forward hooks to selected VGG modules for logging feature maps."""
        for name, module in self._hook_modules:
            self._hook_handles.append(module.register_forward_hook(self._make_capture_hook(name)))

    def set_deconv_training(self, enabled: bool) -> None:
        """Dynamically enable or disable local deconv reconstruction outputs."""
        self.deconv_training = bool(enabled)

    def set_local_training(self, enabled: bool) -> None:
        """Dynamically enable or disable layer-local gradient stopping."""
        self.local_training = bool(enabled)

    def _make_capture_hook(self, name: str):
        """Create a hook that records tensor outputs and exposes 4D maps for losses."""
        def hook(_module, _inputs, output) -> None:
            """Save the hooked module output when it is a tensor."""
            if isinstance(output, torch.Tensor):
                self._save_layer_output(f"vgg16.{name}", output)
                if output.ndim == 4:
                    self._forward_feature_maps.append((f"vgg16.{name}", output))

        return hook

    def _classifier_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier_fc1(x)
        x = self.classifier_relu1(x)
        x = self.classifier_dropout1(x)
        x = self.classifier_fc2(x)
        x = self.classifier_relu2(x)
        x = self.classifier_dropout2(x)
        return self.classifier_fc3(x)

    def forward(
        self,
        x: torch.Tensor,
        *,
        deconv_training: bool | None = None,
        local_training: bool | None = None,
    ) -> tuple[torch.Tensor, list[FeatureMapEntry | tuple[str, torch.Tensor]]]:
        """Run VGG-16 and return logits plus the hook-collected feature maps."""
        self._reset_layer_outputs()
        self._forward_feature_maps = []
        use_deconvs = self.deconv_training if deconv_training is None else bool(deconv_training)
        use_local = self.local_training if local_training is None else bool(local_training)

        if self.small:
            if use_local:
                x = x.detach()
            conv_input = x
            x = self.conv1_1(x)
            if use_deconvs:
                self._forward_feature_maps.append(
                    ("vgg16.conv1_1.reconstruction", self.deconv1_1(x.clone()), conv_input.detach())
                )
            x = self.relu1_1(x)
            x = self.pool1(x)

            if use_local:
                x = x.detach()
            conv_input = x
            x = self.conv2_1(x)
            if use_deconvs:
                self._forward_feature_maps.append(
                    ("vgg16.conv2_1.reconstruction", self.deconv2_1(x.clone()), conv_input.detach())
                )
            x = self.relu2_1(x)
            x = self.pool2(x)

            if use_local:
                x = x.detach()
            conv_input = x
            x = self.conv3_1(x)
            if use_deconvs:
                self._forward_feature_maps.append(
                    ("vgg16.conv3_1.reconstruction", self.deconv3_1(x.clone()), conv_input.detach())
                )
            x = self.relu3_1(x)
            x = self.pool3(x)

            logits = self._classifier_forward(x)
            return logits, self._forward_feature_maps

        if use_local:
            x = x.detach()
        conv_input = x
        x = self.conv1_1(x)
        if use_deconvs:
            self._forward_feature_maps.append(
                ("vgg16.conv1_1.reconstruction", self.deconv1_1(x.clone()), conv_input.detach())
            )
        x = self.relu1_1(x)
        if use_local:
            x = x.detach()
        conv_input = x
        x = self.conv1_2(x)
        if use_deconvs:
            self._forward_feature_maps.append(
                ("vgg16.conv1_2.reconstruction", self.deconv1_2(x.clone()), conv_input.detach())
            )
        x = self.relu1_2(x)
        x = self.pool1(x)

        if use_local:
            x = x.detach()
        conv_input = x
        x = self.conv2_1(x)
        if use_deconvs:
            self._forward_feature_maps.append(
                ("vgg16.conv2_1.reconstruction", self.deconv2_1(x.clone()), conv_input.detach())
            )
        x = self.relu2_1(x)
        if use_local:
            x = x.detach()
        conv_input = x
        x = self.conv2_2(x)
        if use_deconvs:
            self._forward_feature_maps.append(
                ("vgg16.conv2_2.reconstruction", self.deconv2_2(x.clone()), conv_input.detach())
            )
        x = self.relu2_2(x)
        x = self.pool2(x)

        if use_local:
            x = x.detach()
        conv_input = x
        x = self.conv3_1(x)
        if use_deconvs:
            self._forward_feature_maps.append(
                ("vgg16.conv3_1.reconstruction", self.deconv3_1(x.clone()), conv_input.detach())
            )
        x = self.relu3_1(x)
        if use_local:
            x = x.detach()
        conv_input = x
        x = self.conv3_2(x)
        if use_deconvs:
            self._forward_feature_maps.append(
                ("vgg16.conv3_2.reconstruction", self.deconv3_2(x.clone()), conv_input.detach())
            )
        x = self.relu3_2(x)
        if use_local:
            x = x.detach()
        conv_input = x
        x = self.conv3_3(x)
        if use_deconvs:
            self._forward_feature_maps.append(
                ("vgg16.conv3_3.reconstruction", self.deconv3_3(x.clone()), conv_input.detach())
            )
        x = self.relu3_3(x)
        x = self.pool3(x)

        if use_local:
            x = x.detach()
        conv_input = x
        x = self.conv4_1(x)
        if use_deconvs:
            self._forward_feature_maps.append(
                ("vgg16.conv4_1.reconstruction", self.deconv4_1(x.clone()), conv_input.detach())
            )
        x = self.relu4_1(x)
        if use_local:
            x = x.detach()
        conv_input = x
        x = self.conv4_2(x)
        if use_deconvs:
            self._forward_feature_maps.append(
                ("vgg16.conv4_2.reconstruction", self.deconv4_2(x.clone()), conv_input.detach())
            )
        x = self.relu4_2(x)
        if use_local:
            x = x.detach()
        conv_input = x
        x = self.conv4_3(x)
        if use_deconvs:
            self._forward_feature_maps.append(
                ("vgg16.conv4_3.reconstruction", self.deconv4_3(x.clone()), conv_input.detach())
            )
        x = self.relu4_3(x)
        x = self.pool4(x)

        if use_local:
            x = x.detach()
        conv_input = x
        x = self.conv5_1(x)
        if use_deconvs:
            self._forward_feature_maps.append(
                ("vgg16.conv5_1.reconstruction", self.deconv5_1(x.clone()), conv_input.detach())
            )
        x = self.relu5_1(x)
        if use_local:
            x = x.detach()
        conv_input = x
        x = self.conv5_2(x)
        if use_deconvs:
            self._forward_feature_maps.append(
                ("vgg16.conv5_2.reconstruction", self.deconv5_2(x.clone()), conv_input.detach())
            )
        x = self.relu5_2(x)
        if use_local:
            x = x.detach()
        conv_input = x
        x = self.conv5_3(x)
        if use_deconvs:
            self._forward_feature_maps.append(
                ("vgg16.conv5_3.reconstruction", self.deconv5_3(x.clone()), conv_input.detach())
            )
        x = self.relu5_3(x)
        x = self.pool5(x)

        logits = self._classifier_forward(x)
        return logits, self._forward_feature_maps
