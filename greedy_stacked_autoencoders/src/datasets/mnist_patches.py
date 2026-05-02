import torch
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


class MNISTPatches(Dataset):
    def __init__(self, train: bool = True):
        raw              = datasets.MNIST(str(DATA_DIR), train=train, download=True,
                                          transform=transforms.ToTensor())
        loader           = DataLoader(raw, batch_size=len(raw), shuffle=False)
        images, _        = next(iter(loader))
        self.wh_mnist    = images[:, 0]       # (num_images, H, W)
        self.num_samples = 10000
        self.patch_size  = 8
        image_height     = self.wh_mnist.shape[1]
        image_width      = self.wh_mnist.shape[2]

        grid_h = image_height // self.patch_size
        grid_w = image_width  // self.patch_size

        self.patches = []
        for _ in range(self.num_samples):
            image_idx = torch.randint(0, self.wh_mnist.shape[0], (1,)).item()
            grid_x    = torch.randint(0, grid_h, (1,)).item()
            grid_y    = torch.randint(0, grid_w, (1,)).item()

            x_start = grid_x * self.patch_size
            y_start = grid_y * self.patch_size

            patch = self.wh_mnist[
                image_idx,
                x_start : x_start + self.patch_size,
                y_start : y_start + self.patch_size,
            ]
            self.patches.append(patch.flatten())

        self.patches = torch.stack(self.patches)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.patches[idx], 0
