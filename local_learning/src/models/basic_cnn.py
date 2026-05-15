import torch
import torch.nn as nn

from .common import FeatureMapEntry, LayerCaptureMixin


class BasicCNN1(LayerCaptureMixin, nn.Module):
    """Small 3-conv CNN with max pooling."""

    def __init__(
        self,
        in_ch: int = 3,
        num_classes: int = 10,
        width: int = 32,
        inference_mode: bool = False,
    ) -> None:
        super().__init__()
        self.inference_mode = bool(inference_mode)
        self.layer_outputs = None

        self.conv1 = nn.Conv2d(in_ch, width, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, width * 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width * 2)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.act = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(width * 2, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[FeatureMapEntry]]:
        self._reset_layer_outputs()
        convs: list[FeatureMapEntry] = []

        c1_input = x
        c1 = self.conv1(c1_input)
        self._save_layer_output("cnn1.conv1", c1)
        convs.append(FeatureMapEntry("cnn1.conv1", c1, self.conv1, c1_input))
        x = self.act(self.bn1(c1))
        self._save_layer_output("cnn1.block1", x)
        x = self.pool(x)

        c2_input = x
        c2 = self.conv2(c2_input)
        self._save_layer_output("cnn1.conv2", c2)
        convs.append(FeatureMapEntry("cnn1.conv2", c2, self.conv2, c2_input))
        x = self.act(self.bn2(c2))
        self._save_layer_output("cnn1.block2", x)
        x = self.pool(x)

        c3_input = x
        c3 = self.conv3(c3_input)
        self._save_layer_output("cnn1.conv3", c3)
        convs.append(FeatureMapEntry("cnn1.conv3", c3, self.conv3, c3_input))
        x = self.act(self.bn3(c3))
        self._save_layer_output("cnn1.block3", x)

        x = self.avgpool(x)
        self._save_layer_output("cnn1.avgpool", x)
        x = torch.flatten(x, 1)
        logits = self.fc(x)
        self._save_layer_output("cnn1.fc", logits)
        return logits, convs


class BasicCNN2(LayerCaptureMixin, nn.Module):
    """Small 3-conv CNN without pooling."""

    def __init__(
        self,
        in_ch: int = 3,
        num_classes: int = 10,
        width: int = 32,
        inference_mode: bool = False,
    ) -> None:
        super().__init__()
        self.inference_mode = bool(inference_mode)
        self.layer_outputs = None

        self.conv1 = nn.Conv2d(in_ch, width, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, width * 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width * 2)

        self.act = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(width * 2, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[FeatureMapEntry]]:
        self._reset_layer_outputs()
        convs: list[FeatureMapEntry] = []

        c1_input = x
        c1 = self.conv1(c1_input)
        self._save_layer_output("cnn2.conv1", c1)
        convs.append(FeatureMapEntry("cnn2.conv1", c1, self.conv1, c1_input))
        x = self.act(self.bn1(c1))
        self._save_layer_output("cnn2.block1", x)

        c2_input = x
        c2 = self.conv2(c2_input)
        self._save_layer_output("cnn2.conv2", c2)
        convs.append(FeatureMapEntry("cnn2.conv2", c2, self.conv2, c2_input))
        x = self.act(self.bn2(c2))
        self._save_layer_output("cnn2.block2", x)

        c3_input = x
        c3 = self.conv3(c3_input)
        self._save_layer_output("cnn2.conv3", c3)
        convs.append(FeatureMapEntry("cnn2.conv3", c3, self.conv3, c3_input))
        x = self.act(self.bn3(c3))
        self._save_layer_output("cnn2.block3", x)

        x = self.avgpool(x)
        self._save_layer_output("cnn2.avgpool", x)
        x = torch.flatten(x, 1)
        logits = self.fc(x)
        self._save_layer_output("cnn2.fc", logits)
        return logits, convs

