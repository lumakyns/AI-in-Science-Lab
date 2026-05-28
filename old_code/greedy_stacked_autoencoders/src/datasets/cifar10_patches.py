import torch
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm.auto import tqdm

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CIFAR10_MIRROR_URL = "https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-python.tar.gz"
datasets.CIFAR10.url = CIFAR10_MIRROR_URL


class CIFAR10Patches(Dataset):
    def __init__(
        self,
        train: bool = True,
        num_samples: int = 10_000,
        patch_size: int = 8,
    ):
        raw = datasets.CIFAR10(
            str(DATA_DIR),
            train=train,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.Grayscale(num_output_channels=1),
                    transforms.ToTensor(),
                ]
            ),
        )
        loader = DataLoader(raw, batch_size=len(raw), shuffle=False)
        images, _ = next(iter(loader))

        # (num_images, H, W)
        self.cifar10_gray = images[:, 0]
        self.num_samples = num_samples
        self.patch_size = patch_size

        image_height = self.cifar10_gray.shape[1]
        image_width = self.cifar10_gray.shape[2]

        grid_h = image_height // patch_size
        grid_w = image_width // patch_size

        patches_list = []
        for _ in tqdm(range(num_samples), desc="Extracting patches"):
            image_idx = torch.randint(0, self.cifar10_gray.shape[0], (1,)).item()
            grid_x = torch.randint(0, grid_h, (1,)).item()
            grid_y = torch.randint(0, grid_w, (1,)).item()

            x_start = grid_x * patch_size
            y_start = grid_y * patch_size

            patch = self.cifar10_gray[
                image_idx,
                x_start : x_start + patch_size,
                y_start : y_start + patch_size,
            ]
            patches_list.append(patch.flatten())

        self.patches = torch.stack(patches_list)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.patches[idx], 0
