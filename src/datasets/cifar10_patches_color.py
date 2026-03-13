import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm.auto import tqdm


def local_contrast_normalize(patches: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mean = patches.mean(axis=1, keepdims=True)
    std = patches.std(axis=1, keepdims=True)
    return (patches - mean) / (std + eps)


def zca_whiten(patches: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    X = patches - patches.mean(axis=0)
    cov = (X.T @ X) / (X.shape[0])
    U, S, _ = np.linalg.svd(cov)
    zca_matrix = U @ np.diag(1.0 / np.sqrt(S + eps)) @ U.T
    X_white = X @ zca_matrix
    return X_white


class CIFAR10PatchesColor(Dataset):
    def __init__(
        self,
        train: bool = True,
        num_samples: int = 1_000_000,
        patch_size: int = 8,
        lcn_eps: float = 1e-8,
        zca_eps: float = 1e-5,
    ):
        raw = datasets.CIFAR10(
            "../data",
            train=train,
            download=True,
            transform=transforms.ToTensor(),
        )
        loader = DataLoader(raw, batch_size=len(raw), shuffle=False)
        images, _ = next(iter(loader))
        self.rgb_cifar10 = images  # (num_images, 3, H, W)
        self.num_samples = num_samples
        self.patch_size  = patch_size
        self.lcn_eps = lcn_eps
        self.zca_eps = zca_eps

        image_height = self.rgb_cifar10.shape[2]
        image_width  = self.rgb_cifar10.shape[3]
        grid_h = image_height // patch_size
        grid_w = image_width // patch_size

        # Extract patches: (num_samples, 3*8*8)
        patches_list = []
        for _ in tqdm(range(num_samples), desc="Extracting patches"):
            image_idx = torch.randint(0, self.rgb_cifar10.shape[0], (1,)).item()
            grid_x    = torch.randint(0, grid_h, (1,)).item()
            grid_y    = torch.randint(0, grid_w, (1,)).item()
            x_start   = grid_x * patch_size
            y_start   = grid_y * patch_size

            patch = self.rgb_cifar10[
                image_idx,
                :,
                x_start : x_start + patch_size,
                y_start : y_start + patch_size,
            ]
            patches_list.append(patch.flatten())

        patches = torch.stack(patches_list).numpy()
        patches = local_contrast_normalize(patches, eps=lcn_eps)
        patches = zca_whiten(patches, eps=zca_eps)

        self.patches = torch.from_numpy(patches).float()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.patches[idx], 0
