"""ViT with balanced bipartite matching, weighted merging and proportional attention.

Algorithm reference: https://github.com/facebookresearch/ToMe (Bolya et al.).
This implementation keeps the class token protected and optionally traces every
original patch through every block, without keeping traces during training.
"""
import torch
from torch import nn
from torch.nn import functional as F

from ..datasets import Dataset, specification


def matching(metric, reduction):
    """Return a differentiable sum/amax reducer; matching itself has no gradient."""
    count = min(reduction, (metric.shape[1] - 1) // 2)
    if count == 0:
        return lambda values, mode='sum': values
    with torch.no_grad():
        metric = F.normalize(metric.float(), dim=-1)
        scores = metric[:, ::2] @ metric[:, 1::2].transpose(-1, -2)
        scores[:, 0] = -torch.inf  # CLS never merges into a spatial token.
        similarity, destination = scores.max(dim=-1)
        order = similarity.argsort(dim=-1, descending=True)
        selected = order[:, :count, None]
        remaining = order[:, count:, None].sort(dim=1).values  # CLS stays first.
        destinations = destination[..., None].gather(1, selected)

    def reduce(values, mode='sum'):
        source, target = values[:, ::2], values[:, 1::2]
        channels = values.shape[-1]
        untouched = source.gather(1, remaining.expand(-1, -1, channels))
        selected_values = source.gather(1, selected.expand(-1, -1, channels))
        target = target.scatter_reduce(1, destinations.expand(-1, -1, channels),
                                       selected_values, reduce=mode, include_self=True)
        return torch.cat((untouched, target), dim=1)
    return reduce


class DropPath(nn.Module):
    def __init__(self, probability):
        super().__init__()
        self.probability = probability

    def forward(self, x):
        if not self.training or self.probability == 0:
            return x
        keep = 1 - self.probability
        mask = torch.empty((len(x), 1, 1), device=x.device, dtype=x.dtype).bernoulli_(keep)
        return x * mask / keep


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio, dropout, drop_path, reduction, proportional_attention):
        super().__init__()
        self.heads, self.reduction = heads, reduction
        self.proportional_attention = proportional_attention
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv, self.projection = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(int(dim * mlp_ratio), dim), nn.Dropout(dropout))
        self.dropout = dropout
        self.path = DropPath(drop_path)

    def forward(self, x, size, source=None, *, merge_tokens=True):
        batch, tokens, dim = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(batch, tokens, 3, self.heads, dim // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        bias = size.log()[:, None, :, 0].unsqueeze(2) if self.proportional_attention else None
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=bias,
                    dropout_p=self.dropout if self.training else 0.0)
        x = x + self.path(self.projection(attended.transpose(1, 2).reshape(batch, tokens, dim)))
        merge = matching(k.mean(1), self.reduction if merge_tokens else 0)
        if source is not None:
            source = merge(source, mode='amax')
        weighted = merge(x * size)
        size = merge(size)
        x = weighted / size
        return x + self.path(self.mlp(self.norm2(x))), size, source


class ToMeViT(nn.Module):
    def __init__(self, *, dataset: Dataset, patch_size=4, dim=192, depth=9, heads=3,
                 mlp_ratio=4.0, dropout=0.0, drop_path=0.1, reduction=4,
                 proportional_attention=True, merge_during_training=True):
        super().__init__()
        (channels, height, width), classes, _, _ = specification(dataset)
        for key, value in dict(patch_size=patch_size, dim=dim, depth=depth, heads=heads).items():
            if type(value) is not int or value < 1:
                raise ValueError(f'{key} must be a positive integer')
        if height % patch_size or width % patch_size or dim % heads:
            raise ValueError('Patch size must divide image dimensions and heads must divide dim')
        if not 0 <= dropout < 1 or not 0 <= drop_path < 1 or mlp_ratio <= 0:
            raise ValueError('Invalid dropout, drop_path, or mlp_ratio')
        if type(proportional_attention) is not bool:
            raise ValueError('proportional_attention must be a boolean')
        if type(merge_during_training) is not bool:
            raise ValueError('merge_during_training must be a boolean')
        self.merge_during_training = merge_during_training
        reductions = [reduction] * depth if type(reduction) is int else reduction
        if not isinstance(reductions, list) or len(reductions) != depth or any(type(r) is not int or r < 0 for r in reductions):
            raise ValueError('reduction must be a nonnegative integer or one integer per block')
        self.input_shape = (channels, height, width)
        self.grid = (height // patch_size, width // patch_size)
        tokens = self.grid[0] * self.grid[1] + 1
        self.patch_embed = nn.Conv2d(channels, dim, patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.position = nn.Parameter(torch.zeros(1, tokens, dim))
        self.blocks = nn.ModuleList(Block(dim, heads, mlp_ratio, dropout, drop_path * i / max(1, depth - 1),
                                         reductions[i], proportional_attention) for i in range(depth))
        self.norm, self.head = nn.LayerNorm(dim), nn.Linear(dim, classes)
        self.token_counts = []
        for r in reductions:
            tokens -= min(r, (tokens - 1) // 2)
            self.token_counts.append(tokens)
        nn.init.trunc_normal_(self.position, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            nn.init.zeros_(module.bias)

    def forward(self, images, *, trace=False):
        if images.ndim != 4 or tuple(images.shape[1:]) != self.input_shape:
            raise ValueError(f'Expected NCHW images with shape {self.input_shape}')
        x = self.patch_embed(images).flatten(2).transpose(1, 2)
        x = torch.cat((self.cls_token.expand(len(x), -1, -1), x), dim=1) + self.position
        size = torch.ones((*x.shape[:2], 1), device=x.device, dtype=x.dtype)
        source = torch.eye(x.shape[1], device=x.device)[None].expand(len(x), -1, -1) if trace else None
        traces = []
        for block in self.blocks:
            x, size, source = block(x, size, source,
                                   merge_tokens=not self.training or self.merge_during_training)
            if trace:
                traces.append(source.detach().cpu())
        logits = self.head(self.norm(x)[:, 0])
        return (logits, traces) if trace else logits
