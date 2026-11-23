"""Three convolutions for inspecting the spatial organization of weights."""
from torch import nn
from ..datasets import Dataset, specification


class CNN(nn.Module):
    def __init__(self, *, dataset: Dataset, channels=(32, 32, 32)):
        super().__init__()
        self.shape, classes, _, _ = specification(dataset)
        if len(channels) != 3 or any(type(c) is not int or c < 1 for c in channels):
            raise ValueError('channels must contain three positive integers')
        if len(set(channels)) != 1:
            raise ValueError('cnn2 requires constant channels across all three convolutions')
        self.features = nn.Sequential(
            nn.Conv2d(self.shape[0], channels[0], 3, padding=1), nn.ReLU(),
            # Even kernel: explicit asymmetric padding preserves 32x32 resolution.
            nn.ZeroPad2d((9, 10, 9, 10)),
            nn.Conv2d(channels[0], channels[1], 20), nn.ReLU(),
            nn.Conv2d(channels[1], channels[2], 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.classifier = nn.Linear(channels[2], classes)

    def forward(self, images):
        if images.ndim != 4 or tuple(images.shape[1:]) != self.shape:
            raise ValueError(f'Expected [N, {self.shape}], got {tuple(images.shape)}')
        return self.classifier(self.features(images))

    def weight_maps(self):
        return {name: layer.weight.detach().float().mean(dim=(0, 1)).cpu()
                for name, layer in self.named_modules()
                if isinstance(layer, nn.Conv2d) and layer.kernel_size == (20, 20)}
