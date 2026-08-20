"""Dataset loading utilities for the separation experiment."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset


class Dataset(str, Enum):
    """Datasets supported by this experiment."""

    CIFAR10 = "cifar10"
    MNIST = "mnist"
    FASHION_MNIST = "fashion_mnist"
    LOWRANK = "lowrank"


class Split(str, Enum):
    """Prepared dataset splits."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@dataclass(frozen=True)
class DatasetSpec:
    """Input and target dimensions plus normalization stats."""

    channels: int
    height: int
    width: int
    num_classes: int
    mean: tuple[float, ...]
    std: tuple[float, ...]


DATASET_SPECS: dict[Dataset, DatasetSpec] = {
    Dataset.CIFAR10: DatasetSpec(
        channels=3,
        height=32,
        width=32,
        num_classes=10,
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
    ),
    Dataset.MNIST: DatasetSpec(
        channels=1,
        height=28,
        width=28,
        num_classes=10,
        mean=(0.1307,),
        std=(0.3081,),
    ),
}


def parse_dataset(value: str | Dataset) -> Dataset:
    """Parse a dataset value from config or code."""

    if isinstance(value, Dataset):
        return value
    try:
        return Dataset(value)
    except ValueError as error:
        choices = ", ".join(dataset.value for dataset in Dataset)
        raise ValueError(f"Unsupported dataset {value!r}; choose one of {choices}") from error


def get_dataset_spec(dataset: Dataset) -> DatasetSpec:
    """Return metadata for a supported dataset."""

    try:
        return DATASET_SPECS[dataset]
    except KeyError as error:
        raise ValueError(f"Unsupported dataset: {dataset!r}") from error


def split_path(data_root: str | Path, dataset: Dataset, split: Split) -> Path:
    """Return the prepared .pt path for a dataset split."""

    return Path(data_root) / dataset.value / f"{split.value}.pt"


def load_split(
    *,
    data_root: str | Path,
    dataset: Dataset,
    split: Split,
    normalize: bool = True,
) -> TensorDataset:
    """Load one prepared split as a TensorDataset.

    Prepared files are written by data/download.py and contain uint8 images in
    [N, C, H, W] plus int64 labels.
    """

    path = split_path(data_root, dataset, split)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing prepared split {path}. Run `python data/download.py` first."
        )

    payload = torch.load(path, map_location="cpu")
    if not {"images", "labels"} <= set(payload):
        raise ValueError(f"Prepared split {path} must contain images and labels")

    images = payload["images"]
    labels = payload["labels"]
    spec = get_dataset_spec(dataset)
    expected_shape = (spec.channels, spec.height, spec.width)
    if tuple(images.shape[1:]) != expected_shape:
        raise ValueError(
            f"{path} has images shaped {tuple(images.shape[1:])}; "
            f"expected {expected_shape}"
        )
    if images.ndim != 4 or labels.ndim != 1 or images.shape[0] != labels.shape[0]:
        raise ValueError(f"Prepared split {path} has incompatible image/label tensors")

    images = images.to(dtype=torch.float32).div_(255.0)
    if normalize:
        mean = torch.tensor(spec.mean, dtype=images.dtype).view(-1, 1, 1)
        std = torch.tensor(spec.std, dtype=images.dtype).view(-1, 1, 1)
        images = (images - mean) / std

    return TensorDataset(images.contiguous(), labels.to(dtype=torch.int64).contiguous())


def build_loader(
    *,
    data_root: str | Path,
    dataset: Dataset,
    split: Split,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    seed: int,
    normalize: bool = True,
) -> DataLoader:
    """Build a deterministic loader for a prepared split."""

    tensor_dataset = load_split(
        data_root=data_root,
        dataset=dataset,
        split=split,
        normalize=normalize,
    )
    shuffle = split == Split.TRAIN
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        tensor_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator if shuffle else None,
    )
