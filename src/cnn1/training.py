"""Train CNN weight-pattern experiments from a YAML configuration."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import wandb
import yaml

from .datasets import Dataset, loaders, specification
from .models import CNN

TIMESTAMP = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')


def load_config(path, overrides=()):
    c = yaml.safe_load(Path(path).read_text())
    for override in overrides:
        key, separator, value = override.partition('=')
        if not separator:
            raise ValueError('Overrides must be dotted.key=YAML-value')
        target = c
        parts = key.split('.')
        for part in parts[:-1]:
            target = target[part]
        if parts[-1] not in target:
            raise ValueError(f'Unknown config key: {key}')
        target[parts[-1]] = yaml.safe_load(value)
    for key in ('dataset', 'seed', 'device', 'data_root', 'augmentation', 'model',
                'optimizer', 'scheduler', 'training', 'dataloader', 'checkpoint', 'statistics', 'wandb'):
        if key not in c:
            raise ValueError(f'Missing config key: {key}')
    specification(Dataset(c['dataset']))
    if type(c['seed']) is not int or not 0 <= c['seed'] < 2**32:
        raise ValueError('Require integer seed in [0, 2**32)')
    if not isinstance(c['wandb']['project'], str) or not c['wandb']['project'].strip():
        raise ValueError('wandb.project must be a nonempty string')
    if c['wandb']['mode'] not in ('online', 'offline'):
        raise ValueError('Explicit online/offline W&B mode required')
    if Path(c['checkpoint']['dir']) != Path('checkpoints/cnn1'):
        raise ValueError('Outputs must use checkpoints/cnn1')
    for k in ('epochs', 'cpu_threads', 'log_every_batches'):
        if type(c['training'][k]) is not int or c['training'][k] < 1:
            raise ValueError(f'training.{k} must be a positive integer')
    if c['training']['gradient_clip_norm'] <= 0:
        raise ValueError('gradient_clip_norm must be positive')
    if c['training']['max_batches'] is not None and c['training']['max_batches'] < 1:
        raise ValueError('max_batches must be null or positive')
    if c['optimizer']['learning_rate'] <= 0 or c['optimizer']['weight_decay'] < 0:
        raise ValueError('Invalid optimizer values')
    if not 0 <= c['scheduler']['min_lr_ratio'] <= 1 or c['statistics']['color_limit'] <= 0:
        raise ValueError('Invalid scheduler or image scale')
    for k in ('batch_size', 'eval_batch_size'):
        if c['dataloader'][k] < 1:
            raise ValueError(f'{k} must be positive')
    if c['dataloader']['num_workers'] < 0:
        raise ValueError('num_workers must be nonnegative')
    CNN(dataset=Dataset(c['dataset']), **c['model'])
    return c


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_maps(model, directory, epoch, limit):
    metrics = {}
    for name, tensor in model.weight_maps().items():
        values = tensor.numpy()
        stem = directory / f'{name}_epoch_{epoch:03d}'
        np.save(Path(str(stem) + '.npy'), values)
        path = Path(str(stem) + '.png')
        fig, ax = plt.subplots(figsize=(5, 4))
        plot = ax.imshow(values, cmap='RdBu_r', vmin=-limit, vmax=limit)
        ax.set(title=f'{name}: mean weights, epoch {epoch}', xlabel='Kernel x', ylabel='Kernel y')
        fig.colorbar(plot, ax=ax, label='Mean signed weight')
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        metrics[f'weights/{name}'] = wandb.Image(str(path))
        metrics.update({f'weights/{name}_{k}': float(v) for k, v in
                        [('min', values.min()), ('max', values.max()), ('std', values.std()),
                         ('clipped_fraction', (np.abs(values) > limit).mean())]})
    return metrics


def pass_epoch(model, loader, device, *, optimizer=None, max_batches=None, report=None, gradient_clip_norm=10.0):
    model.train(optimizer is not None)
    loss_sum = correct = count = 0
    with torch.set_grad_enabled(optimizer is not None):
        for batch, (images, labels) in enumerate(loader, 1):
            if max_batches is not None and batch > max_batches:
                break
            images, labels = images.to(device), labels.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss')
            if optimizer is not None:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm, error_if_nonfinite=True)
                optimizer.step()
            loss_sum += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            count += len(labels)
            if report is not None:
                report(batch, loss_sum / count, correct / count)
    if not count:
        raise ValueError('Empty loader')
    return {'loss': loss_sum / count, 'accuracy': correct / count, 'examples': count}


def run_seed(c, root, dataset, seed):
    seed_all(seed)
    directory = root / f'{dataset.value}_seed{seed}'
    for folder in ('images', 'logs', 'checkpoints'):
        (directory / folder).mkdir(parents=True, exist_ok=True)
    device = torch.device(c['device'])
    model = CNN(dataset=dataset, **c['model']).to(device)
    data = loaders(c, dataset, seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=c['optimizer']['learning_rate'],
                                  weight_decay=c['optimizer']['weight_decay'])
    resolved = {**c, 'dataset': dataset.value, 'seed': seed, 'timestamp': TIMESTAMP,
                'parameter_count': sum(p.numel() for p in model.parameters()),
                'train_examples': len(data['train'].dataset), 'test_examples': len(data['test'].dataset),
                'weight_map_reduction': 'signed weight mean over output and input channels',
                'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu'}
    (directory / 'config.yaml').write_text(yaml.safe_dump(resolved))
    run = wandb.init(project=c['wandb']['project'], entity=c['wandb']['entity'],
                     name=f'{dataset.value}-seed-{seed}', config=resolved,
                     mode=c['wandb']['mode'], tags=c['wandb']['tags'], dir=str(directory / 'logs'))
    if run is None:
        raise RuntimeError('W&B initialization failed')
    (directory / 'wandb_url.txt').write_text(str(run.url))
    print(f'{dataset.value}: W&B {run.url}', flush=True)
    try:
        run.log({'epoch': 0, **save_maps(model, directory / 'images', 0, c['statistics']['color_limit'])})
        for epoch in range(1, c['training']['epochs'] + 1):
            ratio = c['scheduler']['min_lr_ratio'] + (1 - c['scheduler']['min_lr_ratio']) * (
                1 + math.cos(math.pi * (epoch - 1) / max(1, c['training']['epochs'] - 1))) / 2
            for group in optimizer.param_groups:
                group['lr'] = c['optimizer']['learning_rate'] * ratio
            def report(batch, loss, accuracy):
                if batch % c['training']['log_every_batches'] == 0:
                    row = {'epoch': epoch, 'batch': batch, 'train/running_loss': loss,
                           'train/running_accuracy': accuracy}
                    run.log(row)
                    with (directory / 'batch_metrics.jsonl').open('a') as handle:
                        handle.write(json.dumps(row) + '\n')
                    print(f'{dataset.value} epoch={epoch} batch={batch} loss={loss:.4f} train_acc={accuracy:.4f}', flush=True)
            metrics = {'epoch': epoch, 'optimizer/learning_rate': optimizer.param_groups[0]['lr']}
            for split in ('train', 'test'):
                values = pass_epoch(model, data[split], device,
                    optimizer=optimizer if split == 'train' else None,
                    max_batches=c['training']['max_batches'], report=report if split == 'train' else None,
                    gradient_clip_norm=c['training']['gradient_clip_norm'])
                metrics.update({f'{split}/{key}': value for key, value in values.items()})
            maps = save_maps(model, directory / 'images', epoch, c['statistics']['color_limit'])
            metrics.update({k: v for k, v in maps.items() if isinstance(v, float)})
            with (directory / 'metrics.jsonl').open('a') as handle:
                handle.write(json.dumps(metrics) + '\n')
            if c['checkpoint']['save_last']:
                torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                            'epoch': epoch, 'config': resolved}, directory / 'checkpoints' / 'last.pt')
            run.log({**metrics, **maps})
            print(f'{dataset.value} epoch={epoch} train={metrics["train/accuracy"]:.4f} test={metrics["test/accuracy"]:.4f}', flush=True)
        (directory / 'result.json').write_text(json.dumps({**metrics, 'wandb_url': run.url}, indent=2))
    except BaseException:
        run.finish(exit_code=1)
        raise
    else:
        run.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='src/cnn1/config.yaml')
    parser.add_argument('--set', action='append', default=[], metavar='KEY=VALUE')
    parser.add_argument('overrides', nargs='*', metavar='KEY=VALUE')
    args = parser.parse_args()
    c = load_config(args.config, [*args.set, *args.overrides])
    torch.set_num_threads(c['training']['cpu_threads'])
    device = torch.device(c['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    base = Path(c['checkpoint']['dir'])
    root = base / 'runs' / f"{TIMESTAMP}_{c['dataset']}_seed{c['seed']}"
    root.mkdir(parents=True, exist_ok=False)
    latest = base / 'latest'
    if latest.is_symlink():
        latest.unlink()
    latest.symlink_to(root.relative_to(base), target_is_directory=True)
    (root / 'config.yaml').write_text(yaml.safe_dump(c))
    print(f'Run directory: {root}', flush=True)
    run_seed(c, root, Dataset(c['dataset']), c['seed'])
    (root / 'COMPLETE').write_text('Run finished.\n')


if __name__ == '__main__':
    main()
