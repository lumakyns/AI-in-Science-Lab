"""Gathered neighborhood attention: only requested spatial queries are evaluated."""
import torch
from torch import nn
from torch.nn import functional as F


class SelectedAttention(nn.Module):
    def __init__(self, dim, heads, size, window, family, globals):
        super().__init__()
        self.dim, self.heads, self.globals = dim, heads, globals
        coords = torch.stack(torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij'), -1).reshape(-1, 2)
        if family == 'nat':
            start = (coords-window//2).clamp(0, size-window)
            offsets = torch.stack(torch.meshgrid(torch.arange(window), torch.arange(window), indexing='ij'), -1).reshape(-1, 2)
            neighbors = start[:, None] + offsets[None]
            valid = torch.ones(size*size, window*window, dtype=torch.bool)
        else:
            offsets = torch.stack(torch.meshgrid(torch.arange(window)-window//2, torch.arange(window)-window//2, indexing='ij'), -1).reshape(-1, 2)
            neighbors = coords[:, None]+offsets[None]
            valid = ((neighbors >= 0) & (neighbors < size)).all(-1)
        relative = coords[:, None]-neighbors
        relative = relative + window-1
        self.register_buffer('relative_index', relative[...,0]*(2*window-1)+relative[...,1], persistent=False)
        neighbors = neighbors.clamp(0, size-1)
        indices = neighbors[...,0]*size+neighbors[...,1]+globals
        if globals:
            indices = torch.cat((torch.arange(globals).expand(size*size,-1), indices), -1)
            valid = torch.cat((torch.ones(size*size,globals,dtype=torch.bool), valid), -1)
        self.register_buffer('indices', indices, persistent=False)
        self.register_buffer('valid', valid, persistent=False)
        self.relative_bias = nn.Parameter(torch.zeros(heads,(2*window-1)**2))
        self.qkv = nn.Linear(dim,dim*3)
        self.proj = nn.Linear(dim,dim)
        if globals:
            self.global_qkv = nn.Linear(dim,dim*3)
            self.global_proj = nn.Linear(dim,dim)
            self.global_bias = nn.Parameter(torch.zeros(heads,3))

    def forward(self,tokens,mask):
        b,n,d = tokens.shape
        g,h,dh = self.globals,self.heads,d//self.heads
        bi,qi = mask.nonzero(as_tuple=True)
        # Project Q only at selected positions; K/V remain available at every position.
        kv = F.linear(tokens,self.qkv.weight[d:],self.qkv.bias[d:]).reshape(b,n,2,h,dh)
        q = F.linear(tokens[bi,qi+g],self.qkv.weight[:d],self.qkv.bias[:d]).reshape(-1,h,dh)
        ids = self.indices[qi]
        k,v = kv[bi[:,None],ids].unbind(2)
        bias = self.relative_bias[:,self.relative_index[qi]].permute(1,0,2)
        if g:
            bias = torch.cat((self.global_bias[:,1][None,:,None].expand(len(q),-1,g),bias),-1)
        bias = bias.masked_fill(~self.valid[qi,None],float('-inf'))
        weights = ((q[:,:,None]*k.permute(0,2,1,3)).sum(-1)*dh**-.5+bias).softmax(-1)
        out = self.proj((weights[:,:,:,None]*v.permute(0,2,1,3)).sum(2).reshape(-1,d))
        spatial = tokens.new_zeros((b,n-g,d)).index_put((bi,qi),out)
        global_out = tokens[:,:0]
        if g:
            gq,gk,gv = self.global_qkv(tokens).reshape(b,n,3,h,dh).permute(2,0,3,1,4).unbind(0)
            gb = torch.cat((self.global_bias[:,2,None,None].expand(-1,g,g),self.global_bias[:,0,None,None].expand(-1,g,n-g)),-1)
            global_out = self.global_proj(F.scaled_dot_product_attention(gq[:,:,:g],gk,gv,attn_mask=gb[None]).transpose(1,2).reshape(b,g,d))
        return global_out,spatial
