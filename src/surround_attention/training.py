"""One configured CIFAR-100 NAT/ViL surround-modulation run per sweep worker."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import wandb
import yaml

from .datasets import Dataset, loaders, specification
from .models import SurroundTransformer

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
    for key in ('dataset', 'data_root', 'seed', 'device', 'augmentation', 'model',
                'optimizer', 'scheduler', 'training', 'dataloader', 'checkpoint', 'wandb'):
        if key not in c:
            raise ValueError(f'Missing config key: {key}')
    specification(Dataset(c['dataset']))
    if type(c['seed']) is not int or not 0 <= c['seed'] < 2**32:
        raise ValueError('seed must be an integer in [0, 2**32)')
    if Path(c['checkpoint']['dir']) != Path('checkpoints/surround_attention'):
        raise ValueError('checkpoint.dir must be checkpoints/surround_attention')
    if c['wandb']['project'] != 'surround_attention' or c['wandb']['mode'] not in ('online', 'offline'):
        raise ValueError('Require stable W&B project and explicit online/offline mode')
    if c['device'] not in ('cuda', 'cpu'):
        raise ValueError('device must be cuda or cpu')
    for key in ('epochs', 'cpu_threads', 'log_every_batches'):
        if type(c['training'][key]) is not int or c['training'][key] < 1:
            raise ValueError(f'training.{key} must be a positive integer')
    if c['training']['max_batches'] is not None and (type(c['training']['max_batches']) is not int or c['training']['max_batches'] < 1):
        raise ValueError('max_batches must be null or a positive integer')
    if not 0 <= c['training']['label_smoothing'] < 1 or c['training']['gradient_clip_norm'] <= 0:
        raise ValueError('Invalid label smoothing or gradient clipping')
    if c['optimizer']['learning_rate'] <= 0 or c['optimizer']['weight_decay'] < 0:
        raise ValueError('Invalid optimizer settings')
    if len(c['optimizer']['betas']) != 2 or any(not 0 <= beta < 1 for beta in c['optimizer']['betas']):
        raise ValueError('Invalid AdamW betas')
    warmup = c['scheduler']['warmup_epochs']
    if type(warmup) is not int or not 0 <= warmup < c['training']['epochs'] or not 0 <= c['scheduler']['min_lr_ratio'] <= 1:
        raise ValueError('Invalid warmup or cosine minimum')
    for key in ('batch_size', 'eval_batch_size'):
        if type(c['dataloader'][key]) is not int or c['dataloader'][key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if type(c['dataloader']['num_workers']) is not int or c['dataloader']['num_workers'] < 0:
        raise ValueError('num_workers must be a nonnegative integer')
    for section, key in ((c, 'augmentation'), (c['training'], 'amp'), (c['checkpoint'], 'save_last'),
                         (c['dataloader'], 'pin_memory')):
        if type(section[key]) is not bool:
            raise ValueError(f'{key} must be a boolean')
    # Validate all model settings without changing subsequent run initialization.
    with torch.random.fork_rng(devices=[]):
        SurroundTransformer(dataset=Dataset(c['dataset']), **c['model'])
    return c


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def learning_rate(config, epoch):
    warmup = config['scheduler']['warmup_epochs']
    if epoch <= warmup:
        ratio = epoch / warmup
    else:
        progress = (epoch - warmup - 1) / max(1, config['training']['epochs'] - warmup - 1)
        minimum = config['scheduler']['min_lr_ratio']
        ratio = minimum + (1 - minimum) * (1 + math.cos(math.pi * progress)) / 2
    return config['optimizer']['learning_rate'] * ratio


def pass_epoch(model, loader, device, config, *, optimizer=None, scaler=None, report=None):
    model.train(optimizer is not None)
    total_loss = total_correct = total_top5 = count = 0
    amp = config['training']['amp'] and device.type == 'cuda'
    maximum = config['training']['max_batches']
    started = time.monotonic()
    with torch.set_grad_enabled(optimizer is not None):
        for batch, (images, labels) in enumerate(loader, 1):
            if maximum is not None and batch > maximum:
                break
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(images)
                loss = nn.functional.cross_entropy(logits, labels, label_smoothing=(
                    config['training']['label_smoothing'] if optimizer is not None else 0.0))
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss')
            if optimizer is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                # GradScaler skips overflowed steps; full precision must fail on nonfinite gradients.
                nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip_norm'],
                                         error_if_nonfinite=not amp)
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item() * len(labels)
            predictions = logits.detach().topk(5, dim=1).indices
            total_correct += (predictions[:, 0] == labels).sum().item()
            total_top5 += (predictions == labels[:, None]).any(1).sum().item()
            count += len(labels)
            if report is not None:
                report(batch, total_loss / count, total_correct / count)
    if not count:
        raise ValueError('Empty loader')
    return dict(loss=total_loss / count, accuracy=total_correct / count,
                top5_accuracy=total_top5 / count, examples=count, seconds=time.monotonic() - started)


def save_figures(model, directory, epoch, history):
    kernel = model.surround.weight.detach().float().cpu().numpy()
    np.save(directory / f'surround_epoch_{epoch:03d}.npy', kernel)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), layout='constrained')
    if model.surround.variant == 'baseline':
        axes[0].text(.5, .5, 'Baseline: no surround convolution', ha='center', va='center', transform=axes[0].transAxes)
        axes[0].set_axis_off()
    else:
        mean = kernel.mean((0, 1))
        limit = max(float(np.abs(mean).max()), 1e-8)
        plot = axes[0].imshow(mean, cmap='RdBu_r', vmin=-limit, vmax=limit, interpolation='nearest')
        axes[0].set_title(f'Mean surround kernel, epoch {epoch}')
        fig.colorbar(plot, ax=axes[0])
    if history:
        epochs = [row['epoch'] for row in history]
        for split in ('train', 'validation'):
            axes[1].plot(epochs, [row[f'{split}/loss'] for row in history], label=split)
            axes[2].plot(epochs, [row[f'{split}/accuracy'] for row in history], label=split)
        axes[1].legend()
        axes[2].legend()
    axes[1].set(xlabel='Epoch', ylabel='Loss', title='Training uses label smoothing')
    axes[2].set(xlabel='Epoch', ylabel='Top-1 accuracy', ylim=(0, 1))
    path = directory / f'epoch_{epoch:03d}.png'
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def run_experiment(c, directory, data=None):
    seed_all(c['seed'])
    torch.set_num_threads(c['training']['cpu_threads'])
    device = torch.device(c['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    for folder in ('images', 'logs', 'checkpoints'):
        (directory / folder).mkdir(parents=True, exist_ok=True)
    model = SurroundTransformer(dataset=Dataset(c['dataset']), **c['model']).to(device)
    data = loaders(c, Dataset(c['dataset']), c['seed']) if data is None else data
    # No decay on biases/norms; apply the same rule to every ablation.
    parameters = [p for p in model.parameters() if p.requires_grad]
    groups = [dict(params=[p for p in parameters if p.ndim > 1], weight_decay=c['optimizer']['weight_decay']),
              dict(params=[p for p in parameters if p.ndim <= 1], weight_decay=0.0)]
    optimizer = torch.optim.AdamW(groups, lr=c['optimizer']['learning_rate'], betas=c['optimizer']['betas'])
    scaler = torch.amp.GradScaler('cuda', enabled=c['training']['amp'] and device.type == 'cuda')
    resolved = {**c, 'timestamp': TIMESTAMP,
                'parameter_count': sum(p.numel() for p in model.parameters()),
                'trainable_parameters': sum(p.numel() for p in parameters),
                'split_examples': {split: len(loader.dataset) for split, loader in data.items()},
                'modulated_channels': 0 if model.surround.variant == 'baseline' else model.surround.count,
                'attention_backend': 'dense scaled_dot_product_attention with exact local masks',
                'token_grids': [16, 8, 4], 'torch_version': str(torch.__version__),
                'cuda_runtime': torch.version.cuda,
                'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
                'trainable_dog': 'DoG initialized; unconstrained weights after initialization'}
    (directory / 'config.yaml').write_text(yaml.safe_dump(resolved))
    family, variant = c['model']['family'], c['model']['modulation']['variant']
    run = wandb.init(project=c['wandb']['project'], entity=c['wandb']['entity'],
                     name=f'{family}-{variant}-seed{c["seed"]}', group=TIMESTAMP, config=resolved,
                     mode=c['wandb']['mode'], tags=c['wandb']['tags'], dir=str(directory / 'logs'))
    if run is None:
        raise RuntimeError('W&B initialization failed')
    (directory / 'wandb_url.txt').write_text(str(run.url) + '\n')
    print(f'Run {family}/{variant}, parameters={resolved["parameter_count"]}, W&B {run.url}', flush=True)
    history = []
    try:
        run.define_metric('epoch')
        for split in ('train', 'validation', 'optimizer', 'surround'):
            run.define_metric(f'{split}/*', step_metric='epoch')
        path = save_figures(model, directory / 'images', 0, history)
        run.log({'epoch': 0, 'figures/epoch': wandb.Image(str(path))})
        for epoch in range(1, c['training']['epochs'] + 1):
            for group in optimizer.param_groups:
                group['lr'] = learning_rate(c, epoch)
            def report(batch, loss, accuracy):
                if batch % c['training']['log_every_batches'] == 0:
                    row = dict(epoch=epoch, batch=batch, **{'train/running_loss': loss, 'train/running_accuracy': accuracy})
                    run.log(row)
                    with (directory / 'batch_metrics.jsonl').open('a') as handle:
                        handle.write(json.dumps(row) + '\n')
                    print(f'{family}/{variant} epoch={epoch} batch={batch} loss={loss:.4f} accuracy={accuracy:.4f}', flush=True)
            metrics = {'epoch': epoch, 'optimizer/learning_rate': optimizer.param_groups[0]['lr'],
                       'optimizer/amp_scale': scaler.get_scale()}
            for split in ('train', 'validation'):
                values = pass_epoch(model, data[split], device, c,
                                    optimizer=optimizer if split == 'train' else None,
                                    scaler=scaler, report=report if split == 'train' else None)
                metrics.update({f'{split}/{key}': value for key, value in values.items()})
            weights = model.surround.weight.detach().float()
            metrics.update({'surround/weight_mean': weights.mean().item(), 'surround/weight_std': weights.std().item()})
            history.append(metrics)
            with (directory / 'metrics.jsonl').open('a') as handle:
                handle.write(json.dumps(metrics) + '\n')
            path = save_figures(model, directory / 'images', epoch, history)
            if c['checkpoint']['save_last']:
                checkpoint = directory / 'checkpoints/last.pt'
                temporary = checkpoint.with_suffix('.tmp')
                torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                            'scaler': scaler.state_dict(), 'epoch': epoch, 'config': resolved}, temporary)
                temporary.replace(checkpoint)
            run.log({**metrics, 'figures/epoch': wandb.Image(str(path))})
            print(f'{family}/{variant} epoch={epoch}/{c["training"]["epochs"]} '
                  f'train={metrics["train/accuracy"]:.4f} validation={metrics["validation/accuracy"]:.4f} '
                  f'seconds={metrics["train/seconds"]:.1f}', flush=True)
        test = pass_epoch(model, data['test'], device, c)
        final = {**metrics, **{f'test/{key}': value for key, value in test.items()}, 'wandb_url': run.url}
        run.log({key: value for key, value in final.items() if key.startswith('test/')})
        run.summary.update({key: value for key, value in final.items() if isinstance(value, (int, float))})
        (directory / 'result.json').write_text(json.dumps(final, indent=2) + '\n')
    except BaseException:
        run.finish(exit_code=1)
        raise
    else:
        run.finish()
        (directory / 'COMPLETE').write_text('Training and final test evaluation finished.\n')
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='src/surround_attention/config.yaml')
    parser.add_argument('overrides', nargs='*', metavar='KEY=VALUE')
    args = parser.parse_args()
    c = load_config(args.config, args.overrides)
    directory = Path(c['checkpoint']['dir']) / 'runs' / (
        f'{TIMESTAMP}_{c["model"]["family"]}_{c["model"]["modulation"]["variant"]}_seed{c["seed"]}')
    directory.mkdir(parents=True, exist_ok=False)
    print(f'Run directory: {directory}', flush=True)
    run_experiment(c, directory)


if __name__ == '__main__':
    main()
