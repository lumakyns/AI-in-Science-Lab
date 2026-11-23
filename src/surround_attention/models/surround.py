"""Channelwise surround modulation following Hasani et al., NeurIPS 2019, eqs. 1–3."""
import math

import torch
from torch import nn
from torch.nn import functional as F


VARIANTS = ('baseline', 'dog_fixed', 'random_fixed', 'dog_trainable', 'random_trainable')


def dog_kernel(size=5, sigma_e=1.2, sigma_i=1.4):
    if type(size) is not int or size < 3 or size % 2 != 1:
        raise ValueError('DoG kernel size must be odd and at least 3')
    if not 0 < sigma_e < sigma_i or not math.isfinite(sigma_i):
        raise ValueError('Require finite 0 < sigma_e < sigma_i')
    axis = torch.arange(size, dtype=torch.float64) - size // 2
    radius = axis[:, None].square() + axis[None, :].square()
    dog = (torch.exp(-radius / (2 * sigma_e**2)) / (2 * math.pi * sigma_e**2)
           - torch.exp(-radius / (2 * sigma_i**2)) / (2 * math.pi * sigma_i**2))
    return (dog / dog[size // 2, size // 2]).float()


class SurroundModulation(nn.Module):
    """Bias-free depthwise convolution on a fixed channel subset; no extra residual."""
    def __init__(self, channels, *, variant, fraction, kernel_size, sigma_e, sigma_i, random_seed):
        super().__init__()
        if variant not in VARIANTS or not 0 < fraction <= 1:
            raise ValueError('Invalid modulation variant or channel fraction')
        self.channels = channels
        self.variant = variant
        self.count = max(1, int(channels * fraction))
        self.padding = kernel_size // 2
        dog = dog_kernel(kernel_size, sigma_e, sigma_i)
        weights = dog.expand(self.count, 1, kernel_size, kernel_size).clone()
        if variant.startswith('random'):
            # Exact same empirical mean, variance, and norm as DoG, without its geometry.
            generator = torch.Generator().manual_seed(random_seed)
            weights = torch.stack([dog.flatten()[torch.randperm(dog.numel(), generator=generator)]
                                   .reshape(1, kernel_size, kernel_size) for _ in range(self.count)])
        if variant.endswith('trainable'):
            self.weight = nn.Parameter(weights)
        else:
            self.register_buffer('weight', weights)

    def forward(self, spatial):
        if spatial.ndim != 4 or spatial.shape[-1] != self.channels:
            raise ValueError('SurroundModulation expects [batch, height, width, channels]')
        if self.variant == 'baseline':
            return spatial
        features = spatial.permute(0, 3, 1, 2)
        modulated = F.conv2d(features[:, :self.count], self.weight,
                             padding=self.padding, groups=self.count)
        return torch.cat((modulated, features[:, self.count:]), dim=1).permute(0, 2, 3, 1)
