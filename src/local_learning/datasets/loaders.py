from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def local_contrast_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    return (x - mean) / (std + eps)


def zca_whiten(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = x.mean(axis=0, keepdims=True)
    centered = x - mean
    cov = np.cov(centered, rowvar=False)
    u, s, _ = np.linalg.svd(cov, full_matrices=False)
    whitening = (u * (1.0 / np.sqrt(s + eps))) @ u.T
    return centered @ whitening


class WhitenedCIFAR(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        *,
        lcn_eps: float = 1e-8,
        zca_eps: float = 1e-5,
    ) -> None:
        loader = DataLoader(base_dataset, batch_size=len(base_dataset), shuffle=False)
        images, labels = next(iter(loader))

        x = images.flatten(1).numpy()
        x = local_contrast_normalize(x, eps=lcn_eps)
        x = zca_whiten(x, eps=zca_eps)

        self.images = torch.from_numpy(x).float().view(-1, 3, 32, 32)
        self.labels = labels.long()

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[idx], self.labels[idx]


def _dataset_cls(dataset: str):
    match dataset:
        case "cifar10":
            return datasets.CIFAR10
        case "cifar100":
            return datasets.CIFAR100
        case _:
            raise ValueError("dataset must be 'cifar10' or 'cifar100'.")


def _normalization_stats(dataset: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
    match dataset:
        case "cifar10":
            return CIFAR10_MEAN, CIFAR10_STD
        case "cifar100":
            return CIFAR100_MEAN, CIFAR100_STD
        case _:
            raise ValueError("dataset must be 'cifar10' or 'cifar100'.")


def get_dataset(
    train: bool,
    dataset: str,
    preprocessing: str = "none",
) -> Dataset:
    dataset_cls = _dataset_cls(dataset)

    match preprocessing:
        case "none":
            transform = transforms.ToTensor()
            return dataset_cls(str(DATA_DIR), train=train, download=True, transform=transform)
        case "normalize":
            mean, std = _normalization_stats(dataset)
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
            return dataset_cls(str(DATA_DIR), train=train, download=True, transform=transform)
        case "whiten":
            base = dataset_cls(
                str(DATA_DIR),
                train=train,
                download=True,
                transform=transforms.ToTensor(),
            )
            return WhitenedCIFAR(base)
        case _:
            raise ValueError("preprocessing must be 'none', 'normalize', or 'whiten'.")

