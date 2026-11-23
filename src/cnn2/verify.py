"""GPU and synthetic smoke checks; run with python -m src.cnn2.verify."""
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from .datasets import Dataset, loaders
from .models import CNN
from .training import load_config, pass_epoch, run_seed, seed_all


def main():
    print('torch', torch.__version__, 'CUDA runtime', torch.version.cuda,
          'available', torch.cuda.is_available(), flush=True)
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    c = load_config('src/cnn2/config.yaml')
    for dataset in (Dataset.CIFAR10, Dataset.CIFAR100):
        seed_all(0)
        model = CNN(dataset=dataset).cuda()
        convolutions = [layer for layer in model.modules() if isinstance(layer, torch.nn.Conv2d)]
        assert [layer.in_channels for layer in convolutions] == [3, 32, 32]
        assert [layer.out_channels for layer in convolutions] == [32, 32, 32]
        assert [layer.kernel_size for layer in convolutions] == [(3, 3), (20, 20), (3, 3)]
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        x = torch.randn(2, 3, 32, 32, device='cuda')
        y = torch.tensor([0, 1], device='cuda')
        for training in (True, False):
            model.train(training)
            with torch.set_grad_enabled(training):
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(logits, y)
                if training:
                    loss.backward()
                    optimizer.step()
        torch.cuda.synchronize()
        torch.testing.assert_close(model.weight_maps()['features.3'], model.features[3].weight.detach().mean((0,1)).cpu())
        data = loaders(c, dataset, 0)
        assert len(data['train'].dataset) == 50000 and len(data['test'].dataset) == 10000
        print(dataset.value, 'forward/backward/evaluation and weight reduction passed', flush=True)
    try:
        CNN(dataset=Dataset.CIFAR10, channels=(32, 64, 128))
    except ValueError:
        pass
    else:
        raise AssertionError('Unequal channel widths must be rejected')
    # Exercise real runner with synthetic data, both class counts, offline tracking.
    from . import training
    training.loaders = lambda c, dataset, seed: {
        split: DataLoader(TensorDataset(torch.randn(4,3,32,32), torch.tensor([0,1,2,3])), batch_size=2)
        for split in ('train','test')}
    c['wandb']['mode'] = 'offline'
    c['training']['epochs'] = 1
    c['training']['log_every_batches'] = 1
    root = Path('checkpoints/cnn2/smoke') / training.TIMESTAMP
    root.mkdir(parents=True, exist_ok=False)
    for dataset in (Dataset.CIFAR10, Dataset.CIFAR100):
        c['dataset'] = dataset.value
        run_seed(c, root, dataset, 0)
        folder = root / f'{dataset.value}_seed0'
        array = np.load(folder / 'images/features.3_epoch_001.npy')
        assert array.shape == (20,20) and np.isfinite(array).all()
        assert (folder / 'images/features.3_epoch_001.png').is_file()
        state = torch.load(folder / 'checkpoints/last.pt', map_location='cpu', weights_only=False)
        assert state['epoch'] == 1
        assert (folder / 'result.json').is_file()
    print('All synthetic training, tracking, checkpoint, and image checks passed.', flush=True)


if __name__ == "__main__":
    main()
