"""Exact local attention graphs, evaluated densely for small CIFAR token grids.

NAT shifts boundary neighborhoods inward (constant neighbor count).
ViL uses clipped centered windows and bidirectional global-token attention.
These reference implementations do not claim sparse/linear runtime complexity.
"""
import torch
from torch import nn
from torch.nn import functional as F


def spatial_graph(size, window, family):
    if family not in ('nat', 'vil') or window < 1 or window % 2 != 1 or window > size:
        raise ValueError('Invalid local attention family/window/grid')
    coords = torch.stack(torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij'), -1).reshape(-1, 2)
    if family == 'nat':
        starts = (coords - window // 2).clamp(0, size - window)
        allowed = ((coords[None] >= starts[:, None]) &
                   (coords[None] < starts[:, None] + window)).all(-1)
    else:
        allowed = (coords[:, None] - coords[None]).abs().amax(-1) <= window // 2
    relative = coords[:, None] - coords[None]
    # Unused out-of-window entries are masked; clamp just keeps lookup in range.
    relative = relative.clamp(1 - window, window - 1) + window - 1
    index = relative[..., 0] * (2 * window - 1) + relative[..., 1]
    return allowed, index


class LocalAttention(nn.Module):
    def __init__(self, dim, heads, size, window, family, global_tokens):
        super().__init__()
        if dim % heads:
            raise ValueError('Embedding width must divide evenly into heads')
        self.heads, self.dim = heads, dim
        self.globals = global_tokens if family == 'vil' else 0
        allowed, index = spatial_graph(size, window, family)
        self.register_buffer('allowed', allowed, persistent=False)
        self.register_buffer('relative_index', index, persistent=False)
        self.relative_bias = nn.Parameter(torch.zeros(heads, (2 * window - 1)**2))
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        if self.globals:
            self.global_qkv = nn.Linear(dim, dim * 3)
            self.global_proj = nn.Linear(dim, dim)
            self.global_bias = nn.Parameter(torch.zeros(heads, 3))

    def forward(self, tokens):
        batch, length, dim = tokens.shape
        g = self.globals
        local = tokens[:, g:]
        qkv = self.qkv(tokens).reshape(batch, length, 3, self.heads, dim // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        spatial_bias = self.relative_bias[:, self.relative_index].masked_fill(~self.allowed, float('-inf'))
        if not g:
            result = F.scaled_dot_product_attention(q, k, v, attn_mask=spatial_bias[None])
            return self.proj(result.transpose(1, 2).reshape(batch, length, dim))
        # ViL local queries see local K/V projections, including global tokens.
        # Global queries use a separate Q/K/V projection on the entire sequence.
        local_bias = torch.cat((self.global_bias[:, 1, None, None].expand(-1, len(self.allowed), g), spatial_bias), -1)
        local_result = F.scaled_dot_product_attention(q[:, :, g:], k, v, attn_mask=local_bias[None])
        global_qkv = self.global_qkv(tokens).reshape(batch, length, 3, self.heads, dim // self.heads).permute(2, 0, 3, 1, 4)
        gq, gk, gv = global_qkv.unbind(0)
        global_bias = torch.cat((self.global_bias[:, 2, None, None].expand(-1, g, g),
                                 self.global_bias[:, 0, None, None].expand(-1, g, local.shape[1])), -1)
        global_result = F.scaled_dot_product_attention(gq[:, :, :g], gk, gv, attn_mask=global_bias[None])
        local_result = self.proj(local_result.transpose(1, 2).reshape(batch, local.shape[1], dim))
        global_result = self.global_proj(global_result.transpose(1, 2).reshape(batch, g, dim))
        return torch.cat((global_result, local_result), 1)
