"""Check attention graphs, ablation controls, CUDA training, outputs, and tracking."""
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .datasets import Dataset, loaders
from .models import SurroundTransformer
from .models.attention import LocalAttention, spatial_graph
from .models.surround import SurroundModulation, VARIANTS, dog_kernel
from .training import TIMESTAMP, load_config, run_experiment, seed_all


def check_attention():
    for family in ('nat', 'vil'):
        allowed, _ = spatial_graph(4, 3, family)
        assert allowed.diagonal().all()
        assert allowed[0].sum() == (9 if family == 'nat' else 4)
        if family == 'nat':
            assert (allowed.sum(1) == 9).all()
        # Compare vectorized attention with explicit per-query neighbor selection.
        attention = LocalAttention(8, 2, 4, 3, family, 1).double()
        torch.nn.init.normal_(attention.relative_bias, std=.1)
        if family == 'vil':
            torch.nn.init.normal_(attention.global_bias, std=.1)
        x = torch.randn(2, 16 + attention.globals, 8, dtype=torch.float64, requires_grad=True)
        actual = attention(x)
        g, heads, dh = attention.globals, attention.heads, 4
        qkv = attention.qkv(x).reshape(2, len(x[0]), 3, heads, dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        expected = []
        if g:
            gq, gk, gv = attention.global_qkv(x).reshape(2, 17, 3, heads, dh).permute(2, 0, 3, 1, 4).unbind(0)
            bias = torch.cat((attention.global_bias[:, 2:3], attention.global_bias[:, 0:1].expand(-1, 16)), -1)
            weights = ((gq[:, :, 0, None] * gk).sum(-1) / dh**.5 + bias).softmax(-1)
            expected.append(attention.global_proj((weights[..., None] * gv).sum(2).reshape(2, 8)))
        for query in range(16):
            neighbors = torch.where(allowed[query])[0]
            indices = torch.cat((torch.arange(g), neighbors + g))
            local_bias = attention.relative_bias[:, attention.relative_index[query, neighbors]]
            bias = torch.cat((attention.global_bias[:, 1:2], local_bias), -1) if g else local_bias
            weights = ((q[:, :, query + g, None] * k[:, :, indices]).sum(-1) / dh**.5 + bias).softmax(-1)
            expected.append(attention.proj((weights[..., None] * v[:, :, indices]).sum(2).reshape(2, 8)))
        expected = torch.stack(expected, 1)
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-8)
        actual_grad = torch.autograd.grad(actual.square().sum(), x, retain_graph=True)[0]
        expected_grad = torch.autograd.grad(expected.square().sum(), x)[0]
        torch.testing.assert_close(actual_grad, expected_grad, atol=1e-9, rtol=1e-8)
    print('Exact NAT/ViL attention outputs and input gradients agree with per-query reference.', flush=True)


def main():
    print('torch', torch.__version__, 'CUDA', torch.version.cuda, 'available', torch.cuda.is_available(), flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for actual-device verification')
    torch.set_num_threads(4)
    check_attention()
    config = load_config('src/surround_attention/config.yaml')
    data = loaders(config, Dataset.CIFAR100, 0)
    assert {s: len(loader.dataset) for s, loader in data.items()} == dict(train=45000, validation=5000, test=10000)
    # Validation/test preprocessing is deterministic and distinct from training augmentation.
    for split in ('validation', 'test'):
        torch.testing.assert_close(data[split].dataset[0][0], data[split].dataset[0][0])
    dog = dog_kernel()
    assert dog[2, 2] == 1 and dog.min() < 0 and dog[2, 1] > 0
    for family in ('nat', 'vil'):
        reference = None
        for variant in VARIANTS:
            c = deepcopy(config)
            c['model']['family'] = family
            c['model']['modulation']['variant'] = variant
            seed_all(0)
            model = SurroundTransformer(dataset=Dataset.CIFAR100, **c['model']).cuda()
            backbone = {name: tensor.cpu().clone() for name, tensor in model.state_dict().items()
                        if not name.startswith('surround.')}
            if reference is None:
                reference = backbone
            else:
                for name in backbone:
                    torch.testing.assert_close(backbone[name], reference[name], rtol=0, atol=0)
            before = model.surround.weight.detach().clone()
            if variant.startswith('random'):
                torch.testing.assert_close(before.mean((1, 2, 3)).cpu(), dog.mean().expand(len(before)))
                torch.testing.assert_close(before.std((1, 2, 3)).cpu(), dog.std().expand(len(before)))
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
            logits = model(torch.randn(2, 3, 32, 32, device='cuda'))
            assert logits.shape == (2, 100)
            loss = F.cross_entropy(logits, torch.tensor([1, 2], device='cuda'))
            assert torch.isfinite(loss)
            loss.backward()
            if variant.endswith('trainable'):
                assert model.surround.weight.grad is not None
                assert model.surround.weight.grad.abs().sum() > 0
            else:
                assert model.surround.weight.grad is None
            optimizer.step()
            if variant.endswith('trainable'):
                assert not torch.equal(before, model.surround.weight)
            else:
                torch.testing.assert_close(before, model.surround.weight, rtol=0, atol=0)
            # Only the first half of spatial channels may be modified.
            spatial = torch.randn(2, 8, 8, 128, device='cuda')
            modulated = model.surround(spatial)
            torch.testing.assert_close(spatial[..., 64:], modulated[..., 64:], rtol=0, atol=0)
            if variant == 'baseline':
                torch.testing.assert_close(spatial, modulated, rtol=0, atol=0)
            torch.cuda.synchronize()
            print(f'{family}/{variant}: forward/backward, frozen/trainable weights, paired backbone passed; '
                  f'{sum(p.numel() for p in model.parameters()):,} parameters', flush=True)
        # Actual configured batch size with mixed precision, to check memory and timing.
        del model, optimizer
        torch.cuda.empty_cache()
        c['model']['modulation']['variant'] = 'dog_trainable'
        model = SurroundTransformer(dataset=Dataset.CIFAR100, **c['model']).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        scaler = torch.amp.GradScaler('cuda')
        x = torch.randn(config['dataloader']['batch_size'], 3, 32, 32, device='cuda')
        labels = torch.arange(len(x), device='cuda') % 100
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16):
                loss = F.cross_entropy(model(x), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        torch.cuda.synchronize()
        print(f'{family}: configured-batch AMP check, {(time.monotonic()-started)/3:.3f} s/step, '
              f'{torch.cuda.max_memory_allocated()/2**20:.0f} MiB peak', flush=True)
        del model, optimizer, x, labels
        torch.cuda.empty_cache()
    # Exercise the real runner, including explicit offline tracking and all output paths.
    root = Path('checkpoints/surround_attention/smoke') / TIMESTAMP
    for family in ('nat', 'vil'):
        c = deepcopy(config)
        c['model']['family'] = family
        c['model']['modulation']['variant'] = 'dog_trainable'
        c['training'].update(epochs=1, log_every_batches=1)
        c['scheduler']['warmup_epochs'] = 0
        c['wandb']['mode'] = 'offline'
        synthetic = {split: DataLoader(TensorDataset(torch.randn(4, 3, 32, 32), torch.arange(4)), batch_size=2)
                     for split in ('train', 'validation', 'test')}
        directory = root / family
        run_experiment(c, directory, data=synthetic)
        state = torch.load(directory / 'checkpoints/last.pt', map_location='cpu', weights_only=True)
        restored = SurroundTransformer(dataset=Dataset.CIFAR100, **c['model'])
        restored.load_state_dict(state['model'], strict=True)
        assert state['epoch'] == 1
        assert (directory / 'COMPLETE').is_file()
        assert (directory / 'images/epoch_001.png').is_file()
        assert np.load(directory / 'images/surround_epoch_001.npy').shape == (64, 1, 5, 5)
        final = json.loads((directory / 'result.json').read_text())
        assert final['test/examples'] == 4 and final['validation/examples'] == 4
    print('All attention, ablation, real GPU, synthetic training, tracking and artifact checks passed.', flush=True)


if __name__ == '__main__':
    main()
