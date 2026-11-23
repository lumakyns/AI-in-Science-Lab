"""Prediction-error routing, across-example lifetime sparsity, and ghost synthesis."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .attention import SelectedAttention


class PatchBlock(nn.Module):
    def __init__(self, dim, heads, size, window, family, global_tokens, mlp_ratio, drop_path, routing):
        super().__init__()
        self.size,self.g,self.policy = size,global_tokens,routing['policy']
        self.keep,self.decay = routing['keep_ratio'],routing['threshold_decay']
        self.norm1,self.norm2 = nn.LayerNorm(dim),nn.LayerNorm(dim)
        self.attention = SelectedAttention(dim,heads,size,window,family,global_tokens)
        self.mlp = nn.Sequential(nn.Linear(dim,int(dim*mlp_ratio)),nn.GELU(),nn.Linear(int(dim*mlp_ratio),dim))
        self.predictor = nn.Conv2d(dim,dim,1)
        self.ghost = nn.Conv2d(dim,dim,5,padding=2,groups=dim,bias=False)
        self.ghost_scale = nn.Parameter(torch.ones(dim)*.1)
        self.drop_probability = drop_path
        self.register_buffer('threshold',torch.zeros(size*size))
        self.register_buffer('calibrated',torch.tensor(False))
        self.register_buffer('frequency',torch.zeros(size*size))
        kernel = torch.ones(1,1,3,3); kernel[:,:,1,1]=0
        self.register_buffer('context_kernel',kernel,persistent=False)

    def select(self,error):
        b,n = error.shape
        if self.policy == 'dense':
            return torch.ones_like(error,dtype=torch.bool)
        if self.policy == 'lifetime':
            if self.training:
                # Each spatial unit wins for only ceil(rho*B) examples, not top patches per image.
                k = max(1,math.ceil(self.keep*b))
                values,indices = error.topk(k,dim=0)
                mask = torch.zeros_like(error,dtype=torch.bool).scatter_(0,indices,True)
                with torch.no_grad():
                    cutoff = values[-1]
                    if self.calibrated:
                        self.threshold.lerp_(cutoff,1-self.decay)
                    else:
                        self.threshold.copy_(cutoff); self.calibrated.fill_(True)
                return mask
            if not self.calibrated:
                raise RuntimeError('Lifetime evaluation requires training-calibrated thresholds')
            return error >= self.threshold
        score = torch.rand_like(error) if self.policy == 'random' else error
        return torch.zeros_like(error,dtype=torch.bool).scatter_(1,score.topk(max(1,math.ceil(self.keep*n)),dim=1).indices,True)

    def drop(self,x):
        if not self.training or not self.drop_probability:
            return x
        keep = 1-self.drop_probability
        return x*torch.empty((len(x),1,1),device=x.device).bernoulli_(keep)/keep

    def forward(self,tokens):
        z = self.norm1(tokens)
        b,_,d = z.shape
        x = z[:,self.g:].transpose(1,2).reshape(b,d,self.size,self.size)
        # Exclude the center so the predictor cannot copy its own target.
        counts = F.conv2d(torch.ones_like(x[:,:1]),self.context_kernel,padding=1)
        context = F.conv2d(x,self.context_kernel.expand(d,1,3,3),padding=1,groups=d)/counts
        prediction = self.predictor(context)
        error = (prediction.float()-x.detach().float()).square().mean(1).flatten(1)
        mask = self.select(error.detach())
        global_update,anchors = self.attention(z,mask)
        anchor_map = anchors.transpose(1,2).reshape_as(x)
        m = mask.reshape(b,1,self.size,self.size).to(x.dtype)
        density = F.avg_pool2d(m,5,stride=1,padding=2,count_include_pad=False).clamp_min(1/25)
        ghost = (prediction + self.ghost(anchor_map)/density)*self.ghost_scale[None,:,None,None]
        generated = ghost.flatten(2).transpose(1,2)
        local_update = torch.where(mask[:,:,None],anchors,generated)
        tokens = tokens + self.drop(torch.cat((global_update,local_update),1))
        tokens = tokens + self.drop(self.mlp(self.norm2(tokens)))
        self.auxiliary_loss = error.mean()
        with torch.no_grad():
            if self.training:
                self.frequency.lerp_(mask.float().mean(0),.05)
            per_image = mask.float().mean(1)
            self.last_mask = mask[0].detach().reshape(self.size,self.size)
            self.last_error = error[0].detach().reshape(self.size,self.size)
            self.stats = torch.stack((mask.float().mean(),error.mean(),per_image.std(unbiased=False),
                self.frequency.std(unbiased=False), (self.frequency < .001).float().mean(),
                (error*mask).sum()/mask.sum().clamp_min(1),
                (error*~mask).sum()/(~mask).sum().clamp_min(1)))
        return tokens
