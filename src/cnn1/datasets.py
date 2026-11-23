"""Prepared CIFAR splits and preprocessing owned by cnn1."""
from enum import Enum
from pathlib import Path

import torch
from torch.utils.data import DataLoader, ConcatDataset, Dataset as TorchDataset
from torchvision import transforms


class Dataset(str, Enum):
    CIFAR10 = 'cifar10'
    CIFAR100 = 'cifar100'
    MNIST = 'mnist'
    FASHION_MNIST = 'fashion_mnist'
    LOWRANK = 'lowrank'


SPECS = {
    Dataset.CIFAR10: ((3, 32, 32), 10, (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    Dataset.CIFAR100: ((3, 32, 32), 100, (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
}


def specification(dataset: Dataset):
    if not isinstance(dataset, Dataset):
        raise TypeError('dataset must be the local Dataset enum')
    if dataset not in SPECS:
        raise ValueError(f'cnn1 supports CIFAR-10/100, not {dataset.value}')
    return SPECS[dataset]


class PreparedCIFAR(TorchDataset):
    def __init__(self, *, root, dataset, split, augment):
        shape, classes, mean, std = specification(dataset)
        path = Path(root) / dataset.value / f'{split}.pt'
        if not path.is_file():
            raise FileNotFoundError(f'Missing {path}; run python data/download.py {dataset.value}')
        payload = torch.load(path, weights_only=True, map_location='cpu')
        self.images, self.labels = payload['images'], payload['labels']
        if (self.images.dtype != torch.uint8 or self.images.ndim != 4
                or tuple(self.images.shape[1:]) != shape or not len(self.images)
                or self.labels.shape != (len(self.images),) or self.labels.dtype != torch.int64
                or torch.any((self.labels < 0) | (self.labels >= classes))):
            raise ValueError(f'Invalid prepared CIFAR split: {path}')
        steps = [transforms.ConvertImageDtype(torch.float32)]
        if split != 'test' and augment:
            steps += [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
        self.transform = transforms.Compose(steps + [transforms.Normalize(mean, std)])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.transform(self.images[index]), self.labels[index]


def seed_worker(worker_id):
    import random
    import numpy as np
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def loaders(config, dataset, seed):
    c = config['dataloader']
    train = ConcatDataset([PreparedCIFAR(root=config['data_root'], dataset=dataset,
                          split=split, augment=config['augmentation'])
                           for split in ('train', 'validation')])
    return {split: DataLoader(
        source, batch_size=c['batch_size'] if split == 'train' else c['eval_batch_size'],
        shuffle=split == 'train', num_workers=c['num_workers'], pin_memory=c['pin_memory'],
        worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(seed),
    ) for split, source in (
        ('train', train),
        ('test', PreparedCIFAR(root=config['data_root'], dataset=dataset, split='test', augment=False)))}
