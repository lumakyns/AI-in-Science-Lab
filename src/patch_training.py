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


TIMESTAMP = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')


def load_config(path, overrides=(), *, experiment, Dataset, specification, Model):
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
    if Path(c['checkpoint']['dir']) != Path(f'checkpoints/{experiment}'):
        raise ValueError(f'checkpoint.dir must be checkpoints/{experiment}')
    if c['wandb']['project'] != experiment or c['wandb']['mode'] not in ('online', 'offline'):
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
    if not math.isfinite(c['training']['prediction_weight']) or c['training']['prediction_weight'] < 0:
        raise ValueError('prediction_weight must be finite and nonnegative')
    # Validate all model settings without changing subsequent run initialization.
    with torch.random.fork_rng(devices=[]):
        Model(dataset=Dataset(c['dataset']), **c['model'])
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
    totals = {}
    if device.type == "cuda":
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(device)
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
                prediction_loss = model.auxiliary()
                loss = nn.functional.cross_entropy(logits, labels, label_smoothing=(
                    config['training']['label_smoothing'] if optimizer is not None else 0.0))
                task_loss = loss
                loss = loss + config['training']['prediction_weight'] * prediction_loss
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
            for key,value in {**model.diagnostics(), 'objective': loss.item(), 'prediction_loss': prediction_loss.item()}.items():
                totals[key] = totals.get(key,0) + value * len(labels)
            total_loss += task_loss.item() * len(labels)
            predictions = logits.detach().topk(5, dim=1).indices
            total_correct += (predictions[:, 0] == labels).sum().item()
            total_top5 += (predictions == labels[:, None]).any(1).sum().item()
            count += len(labels)
            if report is not None:
                report(batch, total_loss / count, total_correct / count)
    if not count:
        raise ValueError('Empty loader')
    if device.type == 'cuda':
        torch.cuda.synchronize()
    seconds = time.monotonic()-started
    return dict(loss=total_loss/count, accuracy=total_correct/count, top5_accuracy=total_top5/count,
                examples=count, seconds=seconds, examples_per_second=count/seconds,
                peak_memory_mib=torch.cuda.max_memory_allocated(device)/2**20 if device.type == 'cuda' else 0,
                **{key:value/count for key,value in totals.items()})


def save_figures(model, directory, epoch, history):
    fig,axes = plt.subplots(1,4,figsize=(16,3.5),layout='constrained')
    block = model.routing_blocks()[0]
    axes[0].imshow(block.last_error.float().cpu(),cmap='magma'); axes[0].set_title('Prediction error, first block')
    axes[1].imshow(block.last_mask.float().cpu(),vmin=0,vmax=1); axes[1].set_title('Attended patches')
    for split in ('train','validation'):
        axes[2].plot([r['epoch'] for r in history],[r[f'{split}/accuracy'] for r in history],label=split)
        axes[3].plot([r['epoch'] for r in history],[r[f'{split}/query_fraction'] for r in history],label=split)
    axes[2].set_title('Accuracy'); axes[3].set_title('Spatial query fraction')
    axes[2].legend();axes[3].legend()
    path = directory/f'epoch_{epoch:03d}.png'
    fig.savefig(path,dpi=120);plt.close(fig)
    np.savez(directory/f'routing_{epoch:03d}.npz',**{
        f'{i}_{key}':getattr(layer,key).detach().float().cpu().numpy()
        for i,layer in enumerate(model.routing_blocks()) for key in ('last_mask','last_error','frequency','threshold')})
    return path


def run_experiment(c, directory, data=None, *, Dataset, loaders, Model):
    seed_all(c['seed'])
    torch.set_num_threads(c['training']['cpu_threads'])
    device = torch.device(c['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    for folder in ('images', 'logs', 'checkpoints'):
        (directory / folder).mkdir(parents=True, exist_ok=True)
    model = Model(dataset=Dataset(c['dataset']), **c['model']).to(device)
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
                'attention_backend': 'gathered local K/V, selected spatial Q; dense ViL globals',
                'token_grids': [16,8,4], 'torch_version': str(torch.__version__),
                'cuda_runtime': torch.version.cuda,
                'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
                'selection': c['model']['routing'],
                'scope': 'exploratory single-seed CIFAR-scale adaptations; ghost means spatial feature synthesis'}
    (directory / 'config.yaml').write_text(yaml.safe_dump(resolved))
    family, variant = c['model']['family'], c['model']['routing']['policy']
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
        for split in ('train', 'validation', 'optimizer'):
            run.define_metric(f'{split}/*', step_metric='epoch')
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


