"""YAML-controlled training for the recirculation experiment."""
from __future__ import annotations

import argparse
import copy
import math
import random
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import wandb
import yaml

from .datasets import Dataset, Split, build_loader, require_mnist, INPUT_DIMENSION
from .models import CircularAutoencoder


def load_config(path):
    with Path(path).open() as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError('Config must be a YAML mapping')
    return config


def merge_config(config, updates):
    """Apply W&B nested config and dotted sweep overrides to known fields."""
    for key, value in updates.items():
        target = config
        parts = key.split('.')
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                raise ValueError(f'Unknown config key: {key}')
            target = target[part]
        leaf = parts[-1]
        if leaf not in target:
            raise ValueError(f'Unknown config key: {key}')
        if isinstance(value, dict) and isinstance(target[leaf], dict):
            merge_config(target[leaf], value)
        else:
            target[leaf] = value
    return config


def validate_config(c):
    require_mnist(Dataset(c['dataset']))
    if type(c['model']['tied_weights']) is not bool:
        raise ValueError('model.tied_weights must be a boolean')
    if c['learning']['algorithm'] not in ('bp', 'recirculation'):
        raise ValueError('learning.algorithm must be bp or recirculation')
    for key in ('every_steps', 'diagnostics_every_steps'):
        if type(c['logging'][key]) is not int or c['logging'][key] < 1:
            raise ValueError(f'logging.{key} must be a positive integer')
    if type(c['training']['cpu_threads']) is not int or c['training']['cpu_threads'] < 1:
        raise ValueError('training.cpu_threads must be a positive integer')
    if type(c['dataloader']['eval_batch_size']) is not int or c['dataloader']['eval_batch_size'] < 1:
        raise ValueError('dataloader.eval_batch_size must be a positive integer')
    if type(c['seed']) is not int:
        raise ValueError('seed must be an integer')
    for section, key in [('training', 'epochs'), ('dataloader', 'batch_size'),
                         ('checkpoint', 'every_epochs'), ('scheduler', 'step_size')]:
        value = c[section][key]
        if type(value) is not int or value < 1:
            raise ValueError(f'{section}.{key} must be a positive integer')
    limit = c['training']['max_batches']
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError('training.max_batches must be null or a positive integer')
    o = c['optimizer']
    if o['type'] != 'sgd' or not math.isfinite(o['learning_rate']) or o['learning_rate'] <= 0:
        raise ValueError('Use SGD with a finite positive learning rate')
    if not math.isfinite(o['momentum']) or not 0 <= o['momentum'] < 1:
        raise ValueError('SGD momentum must be in [0, 1)')
    if not math.isfinite(o['weight_decay']) or o['weight_decay'] < 0:
        raise ValueError('SGD weight_decay must be finite and nonnegative')
    if c['scheduler']['type'] not in ('none', 'step') or not 0 < c['scheduler']['gamma'] <= 1:
        raise ValueError('Invalid scheduler settings')
    if c['dataloader']['num_workers'] < 0:
        raise ValueError('num_workers must be nonnegative')
    if c['wandb']['mode'] not in ('online', 'offline'):
        raise ValueError('wandb.mode must be online or offline')
    if type(c['plots']['examples']) is not int or not 1 <= c['plots']['examples'] <= 12:
        raise ValueError('plots.examples must be between 1 and 12')
    if not Path(c['checkpoint']['dir']).resolve().is_relative_to(Path('checkpoints/recirculation').resolve()):
        raise ValueError('Checkpoints must be under checkpoints/recirculation')


@torch.no_grad()
def evaluate(model, loader, device):
    totals = {'loss': 0.0, 'mse': 0.0}
    count = 0
    for images, _ in loader:
        images = images.to(device)
        n = len(images)
        squared_error = (model(images) - model.flatten(images)).square()
        totals['loss'] += 0.5 * squared_error.sum().item()
        totals['mse'] += squared_error.mean().item()*n
        count += n
    if count == 0:
        raise ValueError('Empty evaluation split')
    metrics = {key: value/count for key,value in totals.items()}
    if not all(math.isfinite(v) for v in metrics.values()):
        raise FloatingPointError('Nonfinite evaluation metric')
    return metrics


def run_training(c, run, timestamp):
    torch.set_num_threads(c['training']['cpu_threads'])
    random.seed(c['seed'])
    np.random.seed(c['seed'])
    torch.manual_seed(c['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(c['seed'])
    device = torch.device(c['device'])
    dataset = Dataset(c['dataset'])
    loader_settings = dict(c['dataloader'])
    eval_batch_size = loader_settings.pop('eval_batch_size')
    loaders = {}
    for split in Split:
        settings = dict(loader_settings)
        if split is not Split.TRAIN:
            settings['batch_size'] = eval_batch_size
        loaders[split] = build_loader(data_root=c['data_root'], dataset=dataset,
                                     split=split, seed=c['seed'], **settings)
    if any(len(loader.dataset) == 0 for loader in loaders.values()):
        raise ValueError('Dataset splits must be nonempty')
    model = CircularAutoencoder(dataset=dataset, tied_weights=c['model']['tied_weights']).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=c['optimizer']['learning_rate'],
                                momentum=c['optimizer']['momentum'],
                                 weight_decay=c['optimizer']['weight_decay'])
    scheduler = None if c['scheduler']['type'] == 'none' else torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=c['scheduler']['step_size'], gamma=c['scheduler']['gamma'])
    run.summary.update({'model/parameters': sum(p.numel() for p in model.parameters()),
                        'model/input_dimension': INPUT_DIMENSION,
                        'data/train_examples': len(loaders[Split.TRAIN].dataset),
                        'data/validation_examples': len(loaders[Split.VALIDATION].dataset),
                        'data/test_examples': len(loaders[Split.TEST].dataset), 'run/timestamp': timestamp})
    history = []
    global_step = 0
    for epoch in range(1, c['training']['epochs']+1):
        model.train()
        total, count = 0., 0
        for batch, (images, _) in enumerate(loaders[Split.TRAIN]):
            if c['training']['max_batches'] is not None and batch >= c['training']['max_batches']:
                break
            images = images.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(c['learning']['algorithm'] == 'bp'):
                values, reconstruction = model.activities(images)
            inputs = values[0]
            loss = 0.5 * (reconstruction - inputs).square().sum(1).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss at epoch {epoch}')
            diagnostics_due = global_step % c['logging']['diagnostics_every_steps'] == 0
            ratios = model.error_ratios(inputs - reconstruction.detach(), include_signs=True) if diagnostics_due else {}
            if c['learning']['algorithm'] == 'bp':
                loss.backward()
            else:
                model.approximate_gradients(values)
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError('Nonfinite learning signal')
            if global_step % c['logging']['every_steps'] == 0 or diagnostics_due:
                run.log({**ratios, 'train/loss': loss.item(),
                         'train/mse': 2 * loss.item() / INPUT_DIMENSION,
                         'train/step': global_step, 'epoch': epoch})
            global_step += 1
            optimizer.step()
            total += loss.item()*len(images)
            count += len(images)
        model.eval()
        metrics = {'epoch': epoch, 'train/epoch_loss': total/count, 'optimizer/learning_rate': optimizer.param_groups[0]['lr']}
        metrics.update({f'validation/{k}':v for k,v in evaluate(model,loaders[Split.VALIDATION],device).items()})
        history.append(metrics)
        run.log(metrics)
        print(f"epoch={epoch} train/loss={total/count:.6g} validation/loss={metrics['validation/loss']:.6g}")
        if scheduler is not None:
            scheduler.step()
        if c['checkpoint']['enabled'] and (epoch % c['checkpoint']['every_epochs'] == 0 or epoch == c['training']['epochs']):
            directory = Path(c['checkpoint']['dir'])
            directory.mkdir(parents=True, exist_ok=True)
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict() if scheduler else None,
                        'config': c, 'epoch': epoch, 'timestamp': timestamp}, directory / f'{timestamp}_{run.id}_epoch{epoch}.pt')
    run.summary.update({f'test/{k}':v for k,v in evaluate(model,loaders[Split.TEST],device).items()})
    if c['plots']['enabled']:
        import json
        from .plots import save_plots
        directory = Path(c['checkpoint']['dir']) / f'{timestamp}_{run.id}'
        images = loaders[Split.TEST].dataset.tensors[0][:c['plots']['examples']].to(device)
        with torch.no_grad():
            reconstructions = model(images).reshape_as(images).cpu().numpy()
        title = f"MNIST circular autoencoder · seed {c['seed']}"
        save_plots(history, images.cpu().numpy(), reconstructions,
                           directory=directory, title=title)
        (directory / 'metrics.json').write_text(json.dumps(history, indent=2))
        (directory / 'config.yaml').write_text(yaml.safe_dump(c))
        run.summary['plots/directory'] = str(directory)
        print(f'Plots saved to {directory.resolve()}')


def main():
    timestamp = datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='src/recirculation/config.yaml')
    config = load_config(parser.parse_args().config)
    validate_config(config)
    tracking = config['wandb']
    run = wandb.init(project=f'recirculation_{timestamp}', config=config,
                     mode=tracking['mode'], entity=tracking['entity'],
                     group=tracking['group'], tags=tracking['tags'])
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
    else:
        run.finish()


if __name__ == '__main__':
    main()
