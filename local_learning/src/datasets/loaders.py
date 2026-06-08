import gzip
import shutil
import tarfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


LOCAL_LEARNING_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = LOCAL_LEARNING_ROOT.parent
DATA_DIR_CANDIDATES = (
    LOCAL_LEARNING_ROOT / "data",
    REPO_ROOT / "data",
)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PATCHES_PER_IMAGE = 5
PATCH_SIZE = 8
SMALL_CIFAR10_FRACTION = 0.1
IMAGENET_VAL_SUBSET_SIZE = 5000
IMAGENET_VAL_SUBSET_TRAIN_FRACTION = 0.8


def _format_dir_contents(path: Path) -> str:
    if not path.exists():
        return f"{path} does not exist"
    if not path.is_dir():
        return f"{path} exists but is not a directory"
    contents = sorted(child.name for child in path.iterdir())
    if not contents:
        return f"{path} is empty"
    return f"{path} contains: {', '.join(contents[:30])}"


def _data_dir() -> Path:
    for path in DATA_DIR_CANDIDATES:
        if (path / "val_blurred.tar.gz").exists() or (path / "val_devkit.json.gz").exists():
            return path
    return DATA_DIR_CANDIDATES[0]


def _missing_file_error(label: str, path: Path) -> FileNotFoundError:
    searched = "\n".join(f"- {_format_dir_contents(candidate)}" for candidate in DATA_DIR_CANDIDATES)
    return FileNotFoundError(f"Missing {label}: {path}\nSearched data folders:\n{searched}")


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

        self.images = torch.from_numpy(x).float().view_as(images)
        self.labels = labels.long()

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[idx], self.labels[idx]


class CIFARPatches(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        *,
        patches_per_image: int = PATCHES_PER_IMAGE,
        patch_size: int = PATCH_SIZE,
        seed: int = 0,
    ) -> None:
        self.base_dataset = base_dataset
        self.patches_per_image = int(patches_per_image)
        self.patch_size = int(patch_size)

        if self.patches_per_image <= 0:
            raise ValueError("patches_per_image must be positive.")
        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive.")

        image, _ = self.base_dataset[0]
        if image.ndim != 3:
            raise ValueError("Patch datasets expect images shaped as (C, H, W).")
        _, image_height, image_width = image.shape
        if self.patch_size > image_height or self.patch_size > image_width:
            raise ValueError("patch_size must fit inside the source images.")

        generator = torch.Generator().manual_seed(seed)
        shape = (len(self.base_dataset), self.patches_per_image)
        self.tops = torch.randint(
            0,
            image_height - self.patch_size + 1,
            shape,
            generator=generator,
        )
        self.lefts = torch.randint(
            0,
            image_width - self.patch_size + 1,
            shape,
            generator=generator,
        )

    def __len__(self) -> int:
        return len(self.base_dataset) * self.patches_per_image

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_idx = idx // self.patches_per_image
        patch_idx = idx % self.patches_per_image
        image, label = self.base_dataset[image_idx]
        top = int(self.tops[image_idx, patch_idx])
        left = int(self.lefts[image_idx, patch_idx])
        patch = image[:, top : top + self.patch_size, left : left + self.patch_size]
        return patch, torch.as_tensor(label).long()


def stratified_subset(dataset: Dataset, *, fraction: float, seed: int = 0) -> Subset:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in the interval (0, 1].")
    if not hasattr(dataset, "targets"):
        raise ValueError("stratified_subset requires a dataset with a targets attribute.")

    targets = torch.as_tensor(dataset.targets)
    generator = torch.Generator().manual_seed(seed)
    indices: list[int] = []
    for label in torch.unique(targets, sorted=True):
        class_indices = torch.nonzero(targets == label, as_tuple=False).flatten()
        shuffled = class_indices[torch.randperm(len(class_indices), generator=generator)]
        keep_count = max(1, int(round(len(class_indices) * fraction)))
        indices.extend(shuffled[:keep_count].tolist())

    return Subset(dataset, sorted(indices))


def deterministic_subset(dataset: Dataset, *, size: int, train: bool, seed: int = 0) -> Subset:
    """Split a deterministic subset into train/test views without requiring train archives."""
    if size <= 0:
        raise ValueError("size must be positive.")
    subset_size = min(int(size), len(dataset))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:subset_size]
    split_idx = max(1, int(round(subset_size * IMAGENET_VAL_SUBSET_TRAIN_FRACTION)))
    split_idx = min(split_idx, subset_size - 1) if subset_size > 1 else subset_size
    selected = indices[:split_idx] if train else indices[split_idx:]
    if selected.numel() == 0:
        selected = indices[:1]
    return Subset(dataset, sorted(selected.tolist()))


def _safe_extract_tar(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive, "r:*") as tar:
        for member in tar.getmembers():
            target = (destination / member.name).resolve()
            if destination != target and destination not in target.parents:
                raise RuntimeError(f"Refusing to extract unsafe archive member: {member.name}")
        tar.extractall(destination)


def _prepare_imagenet_val_subset() -> None:
    """Prepare local ImageNet validation archives for torchvision.datasets.ImageNet."""
    data_dir = _data_dir()
    imagenet_root = data_dir / "imagenet"
    imagenet_meta_file = imagenet_root / "meta.bin"
    imagenet_val_archive = data_dir / "val_blurred.tar.gz"
    imagenet_val_devkit = data_dir / "val_devkit.json.gz"
    imagenet_val_annotations = data_dir / "face_annotations_ILSVRC.json"

    if not imagenet_val_annotations.exists():
        if not imagenet_val_devkit.exists():
            raise _missing_file_error("ImageNet validation devkit", imagenet_val_devkit)
        with gzip.open(imagenet_val_devkit, "rb") as source:
            with imagenet_val_annotations.open("wb") as destination:
                shutil.copyfileobj(source, destination)

    val_root = imagenet_root / "val"
    has_val_classes = val_root.exists() and any(path.is_dir() for path in val_root.iterdir())
    if not has_val_classes:
        if not imagenet_val_archive.exists():
            raise _missing_file_error("ImageNet validation archive", imagenet_val_archive)
        imagenet_root.mkdir(parents=True, exist_ok=True)
        _safe_extract_tar(imagenet_val_archive, imagenet_root)
        extracted_root = imagenet_root / "val_blurred"
        if extracted_root.exists() and val_root.exists() and not any(val_root.iterdir()):
            val_root.rmdir()
        if extracted_root.exists() and not val_root.exists():
            extracted_root.rename(val_root)

    if not imagenet_meta_file.exists():
        if not val_root.exists():
            raise RuntimeError(f"Expected ImageNet validation folder at {val_root}")
        wnids = sorted(path.name for path in val_root.iterdir() if path.is_dir())
        if not wnids:
            raise RuntimeError(f"No ImageNet class folders found in {val_root}")
        wnid_to_classes = {wnid: (wnid,) for wnid in wnids}
        torch.save((wnid_to_classes, []), imagenet_meta_file)


def _base_dataset_name(dataset: str) -> str:
    return dataset.removesuffix("_patches").removesuffix("_val_subset")


def _is_patch_dataset(dataset: str) -> bool:
    return dataset.endswith("_patches")


def _dataset_cls(dataset: str):
    match _base_dataset_name(dataset):
        case "cifar10" | "smallcifar10":
            return datasets.CIFAR10
        case "cifar100":
            return datasets.CIFAR100
        case "imagenet":
            return datasets.ImageNet
        case _:
            raise ValueError(
                "dataset must be 'cifar10', 'cifar100', 'imagenet', 'imagenet_val_subset', "
                "'smallcifar10', 'cifar10_patches', 'cifar100_patches', "
                "'smallcifar10_patches', 'imagenet_patches', or 'imagenet_val_subset_patches'."
            )


def _normalization_stats(dataset: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
    match _base_dataset_name(dataset):
        case "cifar10" | "smallcifar10":
            return CIFAR10_MEAN, CIFAR10_STD
        case "cifar100":
            return CIFAR100_MEAN, CIFAR100_STD
        case "imagenet":
            return IMAGENET_MEAN, IMAGENET_STD
        case _:
            raise ValueError(
                "dataset must be 'cifar10', 'cifar100', 'imagenet', 'imagenet_val_subset', "
                "'smallcifar10', 'cifar10_patches', 'cifar100_patches', "
                "'smallcifar10_patches', 'imagenet_patches', or 'imagenet_val_subset_patches'."
            )


def _maybe_patch_dataset(base: Dataset, dataset: str, train: bool) -> Dataset:
    if not _is_patch_dataset(dataset):
        return base
    seed = 0 if train else 1
    return CIFARPatches(base, seed=seed)


def _imagenet_transform(train: bool, *, normalize: bool) -> transforms.Compose:
    image_transforms: list = [
        transforms.RandomResizedCrop(224) if train else transforms.Resize(256),
    ]
    if not train:
        image_transforms.append(transforms.CenterCrop(224))
    image_transforms.append(transforms.ToTensor())
    if normalize:
        image_transforms.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    return transforms.Compose(image_transforms)


def _make_base_dataset(dataset: str, train: bool, transform) -> Dataset:
    dataset_cls = _dataset_cls(dataset)
    base_dataset = _base_dataset_name(dataset)
    if base_dataset == "imagenet":
        imagenet_root = _data_dir() / "imagenet"
        if dataset.removesuffix("_patches") == "imagenet_val_subset":
            _prepare_imagenet_val_subset()
            val_dataset = dataset_cls(str(imagenet_root), split="val", transform=transform)
            return deterministic_subset(
                val_dataset,
                size=IMAGENET_VAL_SUBSET_SIZE,
                train=train,
                seed=0,
            )
        split = "train" if train else "val"
        return dataset_cls(str(imagenet_root), split=split, transform=transform)
    cifar_dataset = dataset_cls(str(_data_dir()), train=train, download=True, transform=transform)
    if base_dataset == "smallcifar10":
        return stratified_subset(
            cifar_dataset,
            fraction=SMALL_CIFAR10_FRACTION,
            seed=0 if train else 1,
        )
    return cifar_dataset


def get_dataset(
    train: bool,
    dataset: str,
    preprocessing: str = "none",
) -> Dataset:
    match preprocessing:
        case "none":
            if _base_dataset_name(dataset) == "imagenet":
                transform = _imagenet_transform(train, normalize=False)
            else:
                transform = transforms.ToTensor()
            base = _make_base_dataset(dataset, train, transform)
            return _maybe_patch_dataset(base, dataset, train)
        case "normalize":
            if _base_dataset_name(dataset) == "imagenet":
                transform = _imagenet_transform(train, normalize=True)
            else:
                mean, std = _normalization_stats(dataset)
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ])
            base = _make_base_dataset(dataset, train, transform)
            return _maybe_patch_dataset(base, dataset, train)
        case "whiten":
            if _base_dataset_name(dataset) == "imagenet":
                raise ValueError("preprocessing='whiten' is only supported for CIFAR datasets.")
            base = _make_base_dataset(dataset, train, transforms.ToTensor())
            base = _maybe_patch_dataset(base, dataset, train)
            return WhitenedCIFAR(base)
        case _:
            raise ValueError("preprocessing must be 'none', 'normalize', or 'whiten'.")
