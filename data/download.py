"""Download and prepare the repository's image datasets.

Each prepared split is a PyTorch file containing this mapping:

    {"images": uint8 tensor [N, C, H, W], "labels": int64 tensor [N]}

Run from the repository root with ``python data/download.py``.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any


DATA_DIR = Path(__file__).resolve().parent
SPLIT_NAMES = ("train", "validation", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download image datasets and prepare synthetic Lowrank when prepared splits are missing."
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        choices=("cifar10", "mnist", "fashion_mnist", "lowrank"),
        default=("cifar10", "mnist"),
        help="Datasets to prepare (default: both).",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.1,
        help="Fraction of the official training set used for validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the stratified train/validation split.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild splits even when all output files already exist.",
    )
    args = parser.parse_args()
    if not 0.0 < args.validation_fraction < 1.0:
        parser.error("--validation-fraction must be between 0 and 1")
    return args


def require_dependencies() -> tuple[Any, Any]:
    try:
        import torch
        from torchvision import datasets
    except ImportError as error:
        raise SystemExit(
            "PyTorch and torchvision are required. Install them in the active "
            "Conda/Mamba environment before running this script."
        ) from error
    return torch, datasets


def split_paths(dataset_dir: Path) -> dict[str, Path]:
    return {name: dataset_dir / f"{name}.pt" for name in SPLIT_NAMES}


def stratified_indices(
    labels: Any, validation_fraction: float, seed: int, torch: Any
) -> tuple[Any, Any]:
    """Return deterministic train and validation indices for every class."""

    generator = torch.Generator().manual_seed(seed)
    train_parts = []
    validation_parts = []

    for label in torch.unique(labels, sorted=True):
        class_indices = torch.where(labels == label)[0]
        shuffled = class_indices[
            torch.randperm(class_indices.numel(), generator=generator)
        ]
        validation_count = round(class_indices.numel() * validation_fraction)
        validation_parts.append(shuffled[:validation_count])
        train_parts.append(shuffled[validation_count:])

    train_indices = torch.cat(train_parts)
    validation_indices = torch.cat(validation_parts)
    train_indices = train_indices[
        torch.randperm(train_indices.numel(), generator=generator)
    ]
    validation_indices = validation_indices[
        torch.randperm(validation_indices.numel(), generator=generator)
    ]
    return train_indices, validation_indices


def save_split(path: Path, images: Any, labels: Any, torch: Any) -> None:
    payload = {
        "images": images.contiguous().to(dtype=torch.uint8),
        "labels": labels.contiguous().to(dtype=torch.int64),
    }
    temporary_path = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def prepare_dataset(
    name: str,
    load: Callable[[Path, Any], tuple[Any, Any, Any, Any]],
    validation_fraction: float,
    seed: int,
    force: bool,
    torch: Any,
    torchvision_datasets: Any,
) -> None:
    dataset_dir = DATA_DIR / name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    outputs = split_paths(dataset_dir)

    if not force and all(path.is_file() for path in outputs.values()):
        print(f"{name}: splits already exist; skipping")
        return

    print(f"{name}: downloading source data and preparing splits")
    train_images, train_labels, test_images, test_labels = load(
        dataset_dir, torchvision_datasets
    )
    train_indices, validation_indices = stratified_indices(
        train_labels, validation_fraction, seed, torch
    )

    save_split(
        outputs["train"], train_images[train_indices], train_labels[train_indices], torch
    )
    save_split(
        outputs["validation"],
        train_images[validation_indices],
        train_labels[validation_indices],
        torch,
    )
    save_split(outputs["test"], test_images, test_labels, torch)
    print(
        f"{name}: wrote {len(train_indices)} train, "
        f"{len(validation_indices)} validation, and {len(test_labels)} test examples"
    )


def load_cifar10(dataset_dir: Path, datasets: Any) -> tuple[Any, Any, Any, Any]:
    import torch

    train = datasets.CIFAR10(root=dataset_dir, train=True, download=True)
    test = datasets.CIFAR10(root=dataset_dir, train=False, download=True)
    train_images = torch.from_numpy(train.data).permute(0, 3, 1, 2)
    test_images = torch.from_numpy(test.data).permute(0, 3, 1, 2)
    return (
        train_images,
        torch.tensor(train.targets, dtype=torch.int64),
        test_images,
        torch.tensor(test.targets, dtype=torch.int64),
    )


def load_mnist(dataset_dir: Path, datasets: Any) -> tuple[Any, Any, Any, Any]:
    train = datasets.MNIST(root=dataset_dir, train=True, download=True)
    test = datasets.MNIST(root=dataset_dir, train=False, download=True)
    return (
        train.data.unsqueeze(1),
        train.targets,
        test.data.unsqueeze(1),
        test.targets,
    )


def load_fashion_mnist(dataset_dir: Path, datasets: Any) -> tuple[Any, Any, Any, Any]:
    train = datasets.FashionMNIST(root=dataset_dir, train=True, download=True)
    test = datasets.FashionMNIST(root=dataset_dir, train=False, download=True)
    return train.data.unsqueeze(1), train.targets, test.data.unsqueeze(1), test.targets


def prepare_lowrank(validation_fraction: float, seed: int, force: bool, torch: Any) -> None:
    directory = DATA_DIR / "lowrank"
    directory.mkdir(parents=True, exist_ok=True)
    paths = split_paths(directory)
    if not force and all(p.is_file() for p in paths.values()):
        return
    generator = torch.Generator().manual_seed(seed)
    basis = torch.randn(20, 100, generator=generator)
    validation_count = round(60000 * validation_fraction)
    for split, count in (("train", 60000 - validation_count),
                         ("validation", validation_count), ("test", 10000)):
        images = (torch.randn(count, 20, generator=generator) @ basis).view(count, 1, 1, 100)
        # Preserve real values; image save_split deliberately converts to uint8.
        torch.save({"images": images, "labels": torch.zeros(count, dtype=torch.int64)}, paths[split])


def main() -> None:
    args = parse_args()
    torch, torchvision_datasets = require_dependencies()
    loaders = {"cifar10": load_cifar10, "mnist": load_mnist, "fashion_mnist": load_fashion_mnist}

    for name in args.datasets:
        if name == "lowrank":
            prepare_lowrank(args.validation_fraction, args.seed, args.force, torch)
            continue
        prepare_dataset(
            name=name,
            load=loaders[name],
            validation_fraction=args.validation_fraction,
            seed=args.seed,
            force=args.force,
            torch=torch,
            torchvision_datasets=torchvision_datasets,
        )


if __name__ == "__main__":
    main()
