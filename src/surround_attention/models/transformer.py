"""CIFAR-scale hierarchical NAT and Vision Longformer adaptations."""
import torch
from torch import nn

from ..datasets import Dataset, specification
from .attention import LocalAttention
from .surround import SurroundModulation, VARIANTS


class DropPath(nn.Module):
    def __init__(self, probability):
        super().__init__()
        self.probability = probability

    def forward(self, x):
        if not self.training or not self.probability:
            return x
        keep = 1 - self.probability
        mask = torch.empty((len(x),) + (1,) * (x.ndim - 1), device=x.device).bernoulli_(keep)
        return x * mask / keep


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, size, window, family, global_tokens, mlp_ratio, drop_path):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attention = LocalAttention(dim, heads, size, window, family, global_tokens)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(),
                                 nn.Linear(int(dim * mlp_ratio), dim))
        self.drop_path = DropPath(drop_path)

    def forward(self, tokens):
        tokens = tokens + self.drop_path(self.attention(self.norm1(tokens)))
        return tokens + self.drop_path(self.mlp(self.norm2(tokens)))


class SurroundTransformer(nn.Module):
    def __init__(self, *, dataset: Dataset, family, dims, depths, heads, window,
                 global_tokens, mlp_ratio, drop_path, modulation):
        super().__init__()
        self.shape, classes, _, _ = specification(dataset)
        if family not in ('nat', 'vil'):
            raise ValueError('family must be nat or vil')
        if not (len(dims) == len(depths) == len(heads) == 3):
            raise ValueError('Require exactly three stages')
        if any(type(v) is not int or v < 1 for v in [*dims, *depths, *heads]):
            raise ValueError('dims, depths, and heads must be positive integers')
        if any(d % h for d, h in zip(dims, heads)) or dims[0] % 2:
            raise ValueError('Widths must divide evenly into heads; first width must be even')
        if not 0 <= drop_path < 1 or mlp_ratio <= 0 or type(global_tokens) is not int or global_tokens < 1:
            raise ValueError('Invalid drop_path, mlp_ratio, or global_tokens')
        stage, block = modulation['stage'], modulation['block']
        if stage not in range(3) or block not in range(depths[stage]) or modulation['variant'] not in VARIANTS:
            raise ValueError('Invalid modulation stage/block/variant')
        self.family = family
        self.global_count = global_tokens if family == 'vil' else 0
        self.stem = nn.Sequential(nn.Conv2d(self.shape[0], dims[0] // 2, 3, padding=1),
                                   nn.Conv2d(dims[0] // 2, dims[0], 3, stride=2, padding=1))
        self.stem_norm = nn.LayerNorm(dims[0])
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.down_norms = nn.ModuleList()
        self.global_embeddings = nn.ParameterList()
        rates = torch.linspace(0, drop_path, sum(depths)).tolist()
        offset = 0
        for s, (dim, depth, nheads) in enumerate(zip(dims, depths, heads)):
            size = self.shape[1] // (2 ** (s + 1))
            self.stages.append(nn.ModuleList([TransformerBlock(dim, nheads, size, window, family,
                                self.global_count, mlp_ratio, rates[offset + b]) for b in range(depth)]))
            offset += depth
            if self.global_count:
                self.global_embeddings.append(nn.Parameter(torch.zeros(1, self.global_count, dim)))
            if s < 2:
                self.downsamples.append(nn.Conv2d(dim, dims[s + 1], 3, stride=2, padding=1, bias=False))
                self.down_norms.append(nn.LayerNorm(dims[s + 1]))
        self.modulation_stage, self.modulation_block = stage, block
        # Construct all backbone parameters before the modulation, keeping paired initializations identical.
        self.norm = nn.LayerNorm(dims[-1])
        self.classifier = nn.Linear(dims[-1], classes)
        self.apply(self._initialize)
        for embedding in self.global_embeddings:
            nn.init.trunc_normal_(embedding, std=.02)
        self.surround = SurroundModulation(dims[stage], **{k: v for k, v in modulation.items() if k not in ('stage', 'block')})

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, images):
        if images.ndim != 4 or tuple(images.shape[1:]) != self.shape:
            raise ValueError(f'Expected [N, {self.shape}], got {tuple(images.shape)}')
        spatial = self.stem_norm(self.stem(images).permute(0, 2, 3, 1))
        g = self.global_count
        for stage, blocks in enumerate(self.stages):
            batch, height, width, channels = spatial.shape
            tokens = spatial.reshape(batch, height * width, channels)
            if g:
                tokens = torch.cat((self.global_embeddings[stage].expand(batch, -1, -1), tokens), 1)
            for block, layer in enumerate(blocks):
                tokens = layer(tokens)
                if (stage, block) == (self.modulation_stage, self.modulation_block):
                    local = self.surround(tokens[:, g:].reshape(batch, height, width, channels))
                    tokens = torch.cat((tokens[:, :g], local.reshape(batch, height * width, channels)), 1)
            spatial = tokens[:, g:].reshape(batch, height, width, channels)
            if stage < len(self.downsamples):
                spatial = self.down_norms[stage](self.downsamples[stage](spatial.permute(0, 3, 1, 2)).permute(0, 2, 3, 1))
        features = self.norm(tokens)
        return self.classifier(features[:, :g].mean(1) if g else features.mean(1))
