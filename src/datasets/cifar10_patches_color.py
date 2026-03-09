import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class CIFAR10PatchesColor(Dataset):
    """CIFAR-10 8x8 RGB patches (color)."""

    def __init__(self, train: bool = True, num_samples: int = 10000, patch_size: int = 8):
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
        self.patch_size = patch_size
        image_height = self.rgb_cifar10.shape[2]
        image_width = self.rgb_cifar10.shape[3]

        grid_h = image_height // self.patch_size
        grid_w = image_width // self.patch_size

        self.patches = []
        for _ in range(self.num_samples):
            image_idx = torch.randint(0, self.rgb_cifar10.shape[0], (1,)).item()
            grid_x = torch.randint(0, grid_h, (1,)).item()
            grid_y = torch.randint(0, grid_w, (1,)).item()

            x_start = grid_x * self.patch_size
            y_start = grid_y * self.patch_size

            patch = self.rgb_cifar10[
                image_idx,
                :,
                x_start : x_start + self.patch_size,
                y_start : y_start + self.patch_size,
            ]
            self.patches.append(patch.flatten())

        self.patches = torch.stack(self.patches)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.patches[idx], 0
