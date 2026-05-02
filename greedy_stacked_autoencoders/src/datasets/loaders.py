from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .mnist_patches import MNISTPatches
from .cifar10_patches import CIFAR10Patches
from .cifar10_patches_color import CIFAR10PatchesColor
from .cifar10_color import CIFAR10Color

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def get_data_loader(dataset: str, train: bool = True, batch_size: int = 128) -> DataLoader:
    if dataset == "mnist_patches":
        return DataLoader(MNISTPatches(train=train),
                          batch_size=batch_size, shuffle=train, pin_memory=True)

    elif dataset == "cifar10_patches":
        return DataLoader(CIFAR10Patches(train=train),
                          batch_size=batch_size, shuffle=train, pin_memory=True)

    elif dataset == "cifar10_patches_color":
        return DataLoader(CIFAR10PatchesColor(train=train),
                          batch_size=batch_size, shuffle=train, pin_memory=True)

    elif dataset == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            lambda x: x.flatten(),
        ])
        return DataLoader(datasets.MNIST(str(DATA_DIR), train=train, download=True, transform=transform),
                          batch_size=batch_size, shuffle=train, pin_memory=True)

    elif dataset == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=1),
            transforms.Normalize((0.5,), (0.5,)),
            lambda x: x.flatten(),
        ])
        return DataLoader(datasets.CIFAR10(str(DATA_DIR), train=train, download=True, transform=transform),
                          batch_size=batch_size, shuffle=train, pin_memory=True)

    elif dataset == "cifar10_color":
        return DataLoader(CIFAR10Color(train=train),
                          batch_size=batch_size, shuffle=train, pin_memory=True)

    else:
        raise ValueError(f"Unknown dataset: {dataset!r}")


def get_flattened_size(dataset: str) -> int:
    if dataset == "mnist_patches" or dataset == "cifar10_patches":
        return 8 * 8

    elif dataset == "cifar10_patches_color":
        return 8 * 8 * 3

    elif dataset == "mnist":
        return 28 * 28

    elif dataset == "cifar10":
        return 32 * 32

    elif dataset == "cifar10_color":
        return 32 * 32 * 3

    else:
        raise ValueError(f"Unknown dataset: {dataset!r}")


def get_patch_shape(dataset: str) -> tuple:
    """Return (C, H, W) for visualization (patches or full images)."""
    if dataset in ("mnist_patches", "cifar10_patches"):
        return (1, 8, 8)
    elif dataset == "cifar10_patches_color":
        return (3, 8, 8)
    elif dataset == "cifar10_color":
        return (3, 32, 32)
    raise ValueError(f"Unknown dataset for patch shape: {dataset!r}")
