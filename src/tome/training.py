"""Train a CIFAR-scale ToMe ViT and trace original patches into merged tokens."""
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
from .models import ToMeViT

TIMESTAMP = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')


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


def pass_epoch(model, loader, device, config, *, optimizer=None, scaler=None):
    model.train(optimizer is not None)
    total_loss = total_correct = count = 0
    amp = config['training']['amp'] and device.type == 'cuda'
    maximum = config['training']['max_batches']
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
            total_correct += (logits.detach().argmax(1) == labels).sum().item()
            count += len(labels)
    if not count:
        raise ValueError('Empty loader')
    return dict(loss=total_loss/count, accuracy=total_correct/count)


def load_config(path, overrides=()):
    config = yaml.safe_load(Path(path).read_text())
    for override in overrides:
        key, separator, value = override.partition('=')
        if not separator:
            raise ValueError('Overrides must be dotted.key=YAML-value')
        target = config
        parts = key.split('.')
        for part in parts[:-1]:
            target = target[part]
        if parts[-1] not in target:
            raise ValueError(f'Unknown key: {key}')
        target[parts[-1]] = yaml.safe_load(value)
    specification(Dataset(config['dataset']))
    if config['device'] not in ('cuda', 'cpu'):
        raise ValueError('device must be cuda or cpu')
    if type(config['seed']) is not int or not 0 <= config['seed'] < 2**32:
        raise ValueError('seed must be an integer in [0, 2**32)')
    if config['wandb']['project'] != 'tome' or config['wandb']['mode'] not in ('online', 'offline'):
        raise ValueError('Use project tome and explicit online/offline mode')
    if Path(config['checkpoint']['dir']) != Path('checkpoints/tome'):
        raise ValueError('checkpoint.dir must be checkpoints/tome')
    for section, keys in {'training': ('epochs', 'cpu_threads'),
                          'dataloader': ('batch_size', 'eval_batch_size'), 'visualization': ('samples',)}.items():
        for key in keys:
            if type(config[section][key]) is not int or config[section][key] < 1:
                raise ValueError(f'{section}.{key} must be a positive integer')
    maximum = config['training']['max_batches']
    if maximum is not None and (type(maximum) is not int or maximum < 1):
        raise ValueError('max_batches must be null or positive integer')
    warmup = config['scheduler']['warmup_epochs']
    if type(warmup) is not int or not 0 <= warmup < config['training']['epochs']:
        raise ValueError('warmup_epochs must be a nonnegative integer below epochs')
    if not 0 <= config['scheduler']['min_lr_ratio'] <= 1:
        raise ValueError('Invalid min_lr_ratio')
    if not 0 <= config['training']['label_smoothing'] < 1 or config['training']['gradient_clip_norm'] <= 0:
        raise ValueError('Invalid label smoothing or gradient clipping')
    if config['optimizer']['learning_rate'] <= 0 or config['optimizer']['weight_decay'] < 0:
        raise ValueError('Invalid learning rate or weight decay')
    betas = config['optimizer']['betas']
    if len(betas) != 2 or any(not 0 <= b < 1 for b in betas):
        raise ValueError('Invalid AdamW betas')
    if type(config['dataloader']['num_workers']) is not int or config['dataloader']['num_workers'] < 0:
        raise ValueError('num_workers must be nonnegative integer')
    for section, key in ((config, 'augmentation'), (config['training'], 'amp'),
                         (config['dataloader'], 'pin_memory'), (config['checkpoint'], 'save_best'),
                         (config['checkpoint'], 'save_last')):
        if type(section[key]) is not bool:
            raise ValueError(f'{key} must be boolean')
    with torch.random.fork_rng(devices=[]):
        ToMeViT(dataset=Dataset(config['dataset']), **config['model'])
    return config


def save_merge_figures(model, images, labels, directory, epoch, config):
    """Equal group numbers/colors within a layer identify the same merged token."""
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        logits, traces = model(images.to(device), trace=True)
    _, _, mean, std = specification(Dataset(config['dataset']))
    originals = (images * torch.tensor(std)[:, None, None] + torch.tensor(mean)[:, None, None]).clamp(0, 1)
    directory.mkdir(parents=True, exist_ok=True)
    patch = config['model']['patch_size']
    arrays = {'images': originals.numpy(), 'labels': labels.numpy(),
              'predictions': logits.argmax(1).cpu().numpy()}
    paths = []
    for sample in range(len(images)):
        fig, axes = plt.subplots(2, (len(traces) + 2) // 2, figsize=(18, 7), layout='constrained')
        axes = axes.flatten()
        original = originals[sample].permute(1, 2, 0).numpy()
        axes[0].imshow(original)
        axes[0].set_title(f'CIFAR-100 validation sample {sample}\nlabel={labels[sample].item()}, pred={logits[sample].argmax().item()}')
        axes[0].axis('off')
        for block, trace in enumerate(traces):
            # [current tokens, original tokens]; exclude CLS original column.
            groups = trace[sample, :, 1:].argmax(0).reshape(model.grid).numpy()
            arrays[f'sample_{sample}_block_{block + 1}_groups'] = groups
            # Hash token index for visually distinct adjacent group colors.
            colors = plt.get_cmap('turbo')(((groups * 0.61803398875) % 1.0))[..., :3]
            overlay = np.repeat(np.repeat(colors, patch, axis=0), patch, axis=1)
            ax = axes[block + 1]
            ax.imshow(0.5 * original + 0.5 * overlay)
            for y in range(model.grid[0]):
                for x in range(model.grid[1]):
                    ax.text(x * patch + (patch - 1) / 2, y * patch + (patch - 1) / 2,
                            str(groups[y, x]), fontsize=5, ha='center', va='center', color='white',
                            bbox=dict(facecolor='black', alpha=0.4, edgecolor='none', pad=0.1))
            ax.set_title(f'Block {block + 1}: {trace.shape[1] - 1} spatial tokens')
            ax.axis('off')
        for ax in axes[len(traces) + 1:]:
            ax.axis('off')
        fig.suptitle(f'Epoch {epoch}: same color and number in a panel = patches merged into one token')
        path = directory / f'epoch_{epoch:03d}_sample_{sample:02d}.png'
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    np.savez_compressed(directory / f'epoch_{epoch:03d}_groups.npz', **arrays)
    return paths


def save_curves(history, directory, epoch):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for split in ('train', 'validation'):
        for ax, key in zip(axes, ('loss', 'accuracy')):
            ax.plot([r['epoch'] for r in history], [r[f'{split}/{key}'] for r in history], label=split)
            ax.set(xlabel='Epoch', ylabel=key)
            ax.legend()
    path = directory / f'epoch_{epoch:03d}.png'
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def save_checkpoint(path, model, optimizer, scaler, epoch, config, best_accuracy):
    temporary = path.with_suffix('.tmp')
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                    epoch=epoch, config=config, best_validation_accuracy=best_accuracy,
                    python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                    torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()), temporary)
    temporary.replace(path)


def run_experiment(config, *, data=None):
    seed_all(config['seed'])
    torch.set_num_threads(config['training']['cpu_threads'])
    device = torch.device(config['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    directory = Path(config['checkpoint']['dir']) / 'runs' / TIMESTAMP
    for folder in ('images', 'images/patch_merges', 'logs', 'checkpoints'):
        (directory / folder).mkdir(parents=True, exist_ok=True)
    model = ToMeViT(dataset=Dataset(config['dataset']), **config['model']).to(device)
    data = loaders(config, Dataset(config['dataset']), config['seed']) if data is None else data
    sample_count = min(config['visualization']['samples'], len(data['validation'].dataset))
    samples = [data['validation'].dataset[i] for i in range(sample_count)]
    fixed_images = torch.stack([sample[0] for sample in samples])
    fixed_labels = torch.stack([torch.as_tensor(sample[1]) for sample in samples])
    parameters = [p for p in model.parameters() if p.requires_grad]
    groups = [dict(params=[p for p in parameters if p.ndim > 1 and p is not model.cls_token and p is not model.position],
                   weight_decay=config['optimizer']['weight_decay']),
              dict(params=[p for p in parameters if p.ndim <= 1 or p is model.cls_token or p is model.position], weight_decay=0.0)]
    optimizer = torch.optim.AdamW(groups, lr=config['optimizer']['learning_rate'], betas=config['optimizer']['betas'])
    scaler = torch.amp.GradScaler('cuda', enabled=config['training']['amp'] and device.type == 'cuda')
    resolved = {**config, 'timestamp': TIMESTAMP, 'parameter_count': sum(p.numel() for p in parameters),
                'split_examples': {split: len(loader.dataset) for split, loader in data.items()},
                'input_shape': list(model.input_shape), 'patch_grid': list(model.grid),
                'tokens_after_blocks_including_cls': model.token_counts,
                'training_tokens_after_blocks_including_cls': (model.token_counts if model.merge_during_training
                    else [model.grid[0] * model.grid[1] + 1] * len(model.blocks)),
                'visualization_source': 'fixed unaugmented CIFAR-100 validation samples',
                'visualization_sample_indices': list(range(sample_count)),
                'torch_version': str(torch.__version__), 'cuda_runtime': torch.version.cuda,
                'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
                'initialization': 'from scratch; no pretrained weights',
                'algorithm_reference': 'https://github.com/facebookresearch/ToMe'}
    (directory / 'config.yaml').write_text(yaml.safe_dump(resolved))
    run = wandb.init(project=config['wandb']['project'], entity=config['wandb']['entity'],
                     mode=config['wandb']['mode'], group=TIMESTAMP,
                     name=f'vit-tome-train{int(model.merge_during_training)}-cifar100-seed{config["seed"]}-{TIMESTAMP}', config=resolved,
                     tags=config['wandb']['tags'], dir=str(directory / 'logs'),
                     settings=wandb.Settings(x_disable_stats=True, console='off'))
    if run is None or (config['wandb']['mode'] == 'online' and run.offline):
        raise RuntimeError('W&B did not initialize in requested mode')
    (directory / 'wandb_url.txt').write_text(str(run.url) + '\n')
    (Path(config['checkpoint']['dir']) / 'latest_run.txt').write_text(str(directory) + '\n')
    print(f'Run directory: {directory}; parameters={resolved["parameter_count"]}; W&B: {run.url}', flush=True)
    history, best_accuracy, best_epoch = [], -1.0, 0
    try:
        run.define_metric('epoch')
        for split in ('train', 'validation', 'optimizer'):
            run.define_metric(f'{split}/*', step_metric='epoch')
        save_merge_figures(model, fixed_images, fixed_labels, directory / 'images/patch_merges', 0, config)
        for epoch in range(1, config['training']['epochs'] + 1):
            for group in optimizer.param_groups:
                group['lr'] = learning_rate(config, epoch)
            metrics = {'epoch': epoch, 'optimizer/learning_rate': optimizer.param_groups[0]['lr']}
            for split in ('train', 'validation'):
                values = pass_epoch(model, data[split], device, config,
                                    optimizer=optimizer if split == 'train' else None, scaler=scaler)
                metrics.update({f'{split}/{key}': value for key, value in values.items()})
            history.append(metrics)
            with (directory / 'metrics.jsonl').open('a') as handle:
                handle.write(json.dumps(metrics) + '\n')
            improved = metrics['validation/accuracy'] > best_accuracy
            if improved:
                best_accuracy, best_epoch = metrics['validation/accuracy'], epoch
            for name, enabled in (('last', config['checkpoint']['save_last']),
                                  ('best', config['checkpoint']['save_best'] and improved)):
                if enabled:
                    save_checkpoint(directory / f'checkpoints/{name}.pt', model, optimizer, scaler,
                                    epoch, resolved, best_accuracy)
            save_merge_figures(model, fixed_images, fixed_labels, directory / 'images/patch_merges', epoch, config)
            save_curves(history, directory / 'images', epoch)
            run.log(metrics)
            print(f'epoch={epoch}/{config["training"]["epochs"]} train_acc={metrics["train/accuracy"]:.4f} '
                  f'val_acc={metrics["validation/accuracy"]:.4f}', flush=True)
        final = {'best_validation_accuracy': best_accuracy, 'best_epoch': best_epoch}
        run.summary.update(final)
        (directory / 'result.json').write_text(json.dumps(final, indent=2) + '\n')
    except BaseException:
        run.finish(exit_code=1)
        raise
    else:
        run.finish()
        (directory / 'COMPLETE').write_text('Training and validation completed.\n')
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('overrides', nargs='*', help='Sweep overrides: dotted.key=YAML-value')
    args = parser.parse_args()
    run_experiment(load_config(args.config, args.overrides))


if __name__ == '__main__':
    main()
