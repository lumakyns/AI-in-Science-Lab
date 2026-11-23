"""CIFAR-scale hierarchical NAT and Vision Longformer adaptations."""
import torch
from torch import nn

from ..datasets import Dataset, specification
from .routing import PatchBlock


class GhostTransformer(nn.Module):
    def __init__(self, *, dataset: Dataset, family, dims, depths, heads, window,
                 global_tokens, mlp_ratio, drop_path, routing):
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
        if routing['policy'] not in ('dense','random','predictive','lifetime'):
            raise ValueError('Invalid routing policy')
        if not 0 < routing['keep_ratio'] <= 1 or not 0 <= routing['threshold_decay'] < 1:
            raise ValueError('Invalid keep ratio/threshold decay')
        if type(window) is not int or window < 1 or window % 2 != 1 or window > 4:
            raise ValueError('window must be odd and fit the smallest grid')
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
            self.stages.append(nn.ModuleList([PatchBlock(dim, nheads, size, window, family,
                                self.global_count, mlp_ratio, rates[offset + b], routing) for b in range(depth)]))
            offset += depth
            if self.global_count:
                self.global_embeddings.append(nn.Parameter(torch.zeros(1, self.global_count, dim)))
            if s < 2:
                self.downsamples.append(nn.Conv2d(dim, dims[s + 1], 3, stride=2, padding=1, bias=False))
                self.down_norms.append(nn.LayerNorm(dims[s + 1]))
        # All routing policies construct identical parameters for paired initialization.
        self.norm = nn.LayerNorm(dims[-1])
        self.classifier = nn.Linear(dims[-1], classes)
        self.apply(self._initialize)
        for embedding in self.global_embeddings:
            nn.init.trunc_normal_(embedding, std=.02)
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
            spatial = tokens[:, g:].reshape(batch, height, width, channels)
            if stage < len(self.downsamples):
                spatial = self.down_norms[stage](self.downsamples[stage](spatial.permute(0, 3, 1, 2)).permute(0, 2, 3, 1))
        features = self.norm(tokens)
        return self.classifier(features[:, :g].mean(1) if g else features.mean(1))

    def routing_blocks(self):
        return [block for blocks in self.stages for block in blocks]

    def auxiliary(self):
        return torch.stack([block.auxiliary_loss for block in self.routing_blocks()]).mean()

    def diagnostics(self):
        keys = ('attended_fraction','prediction_mse','image_fraction_std','lifetime_frequency_std','inactive_fraction','selected_error','skipped_error')
        stats = torch.stack([block.stats for block in self.routing_blocks()]).mean(0).cpu().tolist()
        result = dict(zip(keys,stats))
        # Token-weighted fraction captures total spatial query reduction across stages.
        blocks = self.routing_blocks()
        result['query_fraction'] = sum(float(block.stats[0])*block.size**2 for block in blocks)/sum(block.size**2 for block in blocks)
        for index,block in enumerate(blocks):
            result[f'block_{index}_fraction'] = float(block.stats[0])
        return result
