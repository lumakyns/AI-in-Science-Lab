import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from .cifar10_patches_color import local_contrast_normalize, zca_whiten

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CIFAR10_MIRROR_URL = "https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-python.tar.gz"
datasets.CIFAR10.url = CIFAR10_MIRROR_URL


class CIFAR10Color(Dataset):
    def __init__(
        self,
        train: bool = True,
        lcn_eps: float = 1e-8,
        zca_eps: float = 1e-5,
    ) -> None:
        raw = datasets.CIFAR10(
            str(DATA_DIR),
            train=train,
            download=True,
            transform=transforms.ToTensor(),
        )
        loader = DataLoader(raw, batch_size=len(raw), shuffle=False)
        images, _ = next(iter(loader))
        self.lcn_eps = lcn_eps
        self.zca_eps = zca_eps

        X = images.flatten(1).numpy()
        X = local_contrast_normalize(X, eps=lcn_eps)
        X = zca_whiten(X, eps=zca_eps)
        self.images = torch.from_numpy(X).float()

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple:
        return self.images[idx], 0
