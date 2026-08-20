"""Supervised MNIST: compare BP with three output-to-input feedback bridges."""
import argparse
import copy
import json
import math
import random
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import wandb
import yaml

from .datasets import Dataset, Split, INPUT_DIMENSION, NUM_CLASSES, require_mnist, build_loader, audit_batch
from .models import RecirculationClassifier


def load_config(path):
    config = yaml.safe_load(Path(path).read_text())
    if not isinstance(config, dict):
        raise ValueError('Config must be a mapping')
    return config


def merge_config(config, updates):
    for key, value in updates.items():
        target = config
        parts = key.split('.')
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                raise ValueError(f'Unknown config field: {key}')
            target = target[part]
        leaf = parts[-1]
        if leaf not in target:
            raise ValueError(f'Unknown config field: {key}')
        if isinstance(target[leaf], dict) and isinstance(value, dict):
            merge_config(target[leaf], value)
        else:
            target[leaf] = value
    return config


def validate_config(c):
    require_mnist(Dataset(c['dataset']))
    if type(c['seed']) is not int:
        raise ValueError('seed must be an integer')
    if c['learning']['method'] not in ('bp', 'random', 'calibrated', 'scrambled'):
        raise ValueError('Unknown learning.method')
    sizes = c['model']['hidden_sizes']
    if not isinstance(sizes, list) or not sizes or any(type(n) is not int or n < 1 for n in sizes):
        raise ValueError('hidden_sizes must be positive integers')
    for section, keys in {'feedback': ['probes', 'refresh_every_steps'],
                          'training': ['epochs', 'cpu_threads'],
                          'dataloader': ['batch_size', 'eval_batch_size'],
                          'logging': ['every_steps', 'audit_every_steps', 'audit_examples'],
                          'checkpoint': ['every_epochs']}.items():
        for key in keys:
            if type(c[section][key]) is not int or c[section][key] < 1:
                raise ValueError(f'{section}.{key} must be a positive integer')
    if not NUM_CLASSES <= c['feedback']['probes'] <= INPUT_DIMENSION:
        raise ValueError('Use at least output-width and at most input-width probes')
    if not math.isfinite(c['feedback']['damping']) or c['feedback']['damping'] <= 0:
        raise ValueError('feedback.damping must be finite and positive')
    limit = c['training']['max_batches']
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError('max_batches must be null or a positive integer')
    o = c['optimizer']
    if o['type'] != 'sgd' or not math.isfinite(o['learning_rate']) or o['learning_rate'] <= 0:
        raise ValueError('Use SGD with a finite positive learning rate')
    if not math.isfinite(o['momentum']) or not 0 <= o['momentum'] < 1:
        raise ValueError('Invalid SGD momentum')
    if not math.isfinite(o['weight_decay']) or o['weight_decay'] < 0:
        raise ValueError('Invalid SGD weight decay')
    if c['scheduler']['type'] != 'none':
        raise ValueError('This experiment uses a constant learning rate')
    if type(c['dataloader']['num_workers']) is not int or c['dataloader']['num_workers'] < 0:
        raise ValueError('Invalid dataloader.num_workers')
    if c['wandb']['mode'] not in ('online', 'offline'):
        raise ValueError('wandb.mode must explicitly be online or offline')
    if not Path(c['checkpoint']['dir']).resolve().is_relative_to(Path('checkpoints/recirculation_supervised').resolve()):
        raise ValueError('Use checkpoints/recirculation_supervised for artifacts')


@torch.no_grad()
def evaluate(model, loader, device):
    total, correct, count = 0.0, 0, 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        predictions = model(images)
        total += .5 * (predictions-targets).square().sum().item()
        correct += (predictions.argmax(1) == targets.argmax(1)).sum().item()
        count += len(images)
    if not count or not math.isfinite(total):
        raise FloatingPointError('Empty split or nonfinite evaluation loss')
    return {'loss': total/count, 'mse': 2*total/count/NUM_CLASSES, 'accuracy': correct/count}


def run_training(c, run, timestamp):
    validate_config(c)
    torch.set_num_threads(c['training']['cpu_threads'])
    random.seed(c['seed']); np.random.seed(c['seed']); torch.manual_seed(c['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(c['seed'])
    device = torch.device(c['device'])
    settings = dict(c['dataloader'])
    evaluation_batch = settings.pop('eval_batch_size')
    loaders = {split: build_loader(data_root=c['data_root'], dataset=Dataset(c['dataset']),
               split=split, seed=c['seed'], **{**settings, 'batch_size': settings['batch_size'] if split is Split.TRAIN else evaluation_batch})
               for split in Split}
    model = RecirculationClassifier(dataset=Dataset(c['dataset']), **c['model'],
                probes=c['feedback']['probes'], feedback_seed=c['seed']+10003).to(device)
    o = c['optimizer']
    optimizer = torch.optim.SGD(model.parameters(), lr=o['learning_rate'],
                                momentum=o['momentum'], weight_decay=o['weight_decay'])
    method = c['learning']['method']
    audit_images, audit_targets = audit_batch(loaders[Split.VALIDATION], c['logging']['audit_examples'], device)
    run.summary.update({'model/parameters': sum(p.numel() for p in model.parameters()),
        'model/input_dimension': INPUT_DIMENSION, 'model/output_classes': NUM_CLASSES,
        'learning/method': method, 'run/timestamp': timestamp,
        **{f'data/{split.value}_examples': len(loader.dataset) for split, loader in loaders.items()}})
    directory = Path(c['checkpoint']['dir']) / f'{timestamp}_{run.id}'
    directory.mkdir(parents=True, exist_ok=True)
    (directory/'config.yaml').write_text(yaml.safe_dump(c))
    step, probe_vectors, history = 0, 0, []
    started = time.perf_counter()
    for epoch in range(1, c['training']['epochs']+1):
        total, correct, count = 0.0, 0, 0
        for batch, (images, targets) in enumerate(loaders[Split.TRAIN]):
            if c['training']['max_batches'] is not None and batch >= c['training']['max_batches']:
                break
            metrics = {}
            if method in ('calibrated', 'scrambled') and step % c['feedback']['refresh_every_steps'] == 0:
                metrics.update(model.calibrate(c['feedback']['damping'], scrambled=method == 'scrambled'))
                probe_vectors += c['feedback']['probes']
            if step % c['logging']['audit_every_steps'] == 0:
                metrics.update(model.audit(audit_images, audit_targets, method=method))
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(method == 'bp'):
                values = model.activities(images)
                loss = .5 * (values[-1]-targets).square().sum(1).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss at epoch {epoch}, step {step}, method {method}')
            if method == 'bp':
                loss.backward()
            else:
                model.local_gradients(values, targets)
            if any(not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError(f'Nonfinite gradient at step {step}')
            n = len(images)
            batch_correct = (values[-1].detach().argmax(1) == targets.argmax(1)).sum().item()
            total += loss.item()*n; correct += batch_correct; count += n
            if metrics or step % c['logging']['every_steps'] == 0:
                run.log({**metrics, 'train/loss': loss.item(), 'train/mse': 2*loss.item()/NUM_CLASSES,
                    'train/accuracy': batch_correct/n, 'train/step': step, 'epoch': epoch,
                    'optimizer/learning_rate': o['learning_rate'], 'compute/probe_vectors': probe_vectors,
                    'compute/elapsed_seconds': time.perf_counter()-started})
            optimizer.step()
            step += 1
        metrics = {'epoch': epoch, 'train/epoch_loss': total/count, 'train/epoch_accuracy': correct/count,
                   'optimizer/learning_rate': o['learning_rate'], 'compute/elapsed_seconds': time.perf_counter()-started}
        metrics.update({f'validation/{key}': value for key, value in evaluate(model, loaders[Split.VALIDATION], device).items()})
        metrics.update(model.audit(audit_images, audit_targets, method=method))
        metrics['train/step'] = step
        history.append(metrics); run.log(metrics)
        (directory/'metrics.json').write_text(json.dumps(history, indent=2))
        print(f"epoch={epoch} method={method} train/loss={total/count:.6g} validation/accuracy={metrics['validation/accuracy']:.4f}")
        if c['checkpoint']['enabled'] and (epoch % c['checkpoint']['every_epochs'] == 0 or epoch == c['training']['epochs']):
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': None,
                        'config': c, 'epoch': epoch, 'step': step, 'timestamp': timestamp}, directory/f'epoch{epoch}.pt')
    test = {f'test/{key}': value for key, value in evaluate(model, loaders[Split.TEST], device).items()}
    run.summary.update(test)
    run.summary.update({'artifacts/directory': str(directory), 'compute/probe_vectors': probe_vectors,
                        'compute/elapsed_seconds': time.perf_counter()-started})
    (directory/'test_metrics.json').write_text(json.dumps(test, indent=2))


def main():
    timestamp = datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='src/recirculation_supervised/config.yaml')
    config = load_config(parser.parse_args().config)
    validate_config(config)
    tracking = config['wandb']
    run = wandb.init(project=f'recirculation_supervised_{timestamp}', config=config,
                     mode=tracking['mode'], entity=tracking['entity'], tags=tracking['tags'])
    if run is None:
        raise RuntimeError('W&B initialization failed')
    try:
        resolved = merge_config(copy.deepcopy(config), dict(run.config))
        validate_config(resolved)
        run.config.update(resolved, allow_val_change=True)
        run_training(resolved, run, timestamp)
    except BaseException:
        run.finish(exit_code=1)
        raise
    run.finish()


if __name__ == '__main__':
    main()
