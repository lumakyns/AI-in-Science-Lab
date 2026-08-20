"""MNIST classification splits, targets, and metadata owned by this experiment."""
from enum import Enum
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset


class Dataset(str, Enum):
    # Repository contract: enumerate every data directory; only MNIST is supported here.
    CIFAR10 = "cifar10"
    MNIST = "mnist"
    FASHION_MNIST = "fashion_mnist"
    LOWRANK = "lowrank"


class Split(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


IMAGE_SHAPE = (1, 28, 28)
INPUT_DIMENSION = 784
NUM_CLASSES = 10


def require_mnist(dataset: Dataset):
    if not isinstance(dataset, Dataset):
        raise TypeError("dataset must be the local Dataset enum")
    if dataset is not Dataset.MNIST:
        raise ValueError("This experiment supports only MNIST")


def build_loader(*, data_root, dataset: Dataset, split: Split, batch_size,
                 num_workers, pin_memory, seed):
    require_mnist(dataset)
    path = Path(data_root) / dataset.value / f"{split.value}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run python data/download.py mnist")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    images, labels = payload["images"], payload["labels"]
    if images.ndim != 4 or tuple(images.shape[1:]) != IMAGE_SHAPE:
        raise ValueError(f"{path}: expected [N, 1, 28, 28] images")
    if images.dtype != torch.uint8 or labels.ndim != 1 or len(images) != len(labels) or not len(images):
        raise ValueError(f"{path}: expected nonempty uint8 images and matching labels")
    if labels.dtype != torch.int64 or torch.any((labels < 0) | (labels >= NUM_CLASSES)):
        raise ValueError(f"{path}: expected class labels in [0, {NUM_CLASSES})")
    targets = torch.nn.functional.one_hot(labels, num_classes=NUM_CLASSES).float()
    tensors = TensorDataset(images.float() / 255, targets)
    return DataLoader(tensors, batch_size=batch_size, shuffle=split is Split.TRAIN,
                      num_workers=num_workers, pin_memory=pin_memory,
                      generator=torch.Generator().manual_seed(seed))



def audit_batch(loader, count, device):
    """Fixed held-out examples for a rank audit at frozen weights."""
    images, targets = loader.dataset.tensors
    return images[:count].to(device), targets[:count].to(device)
