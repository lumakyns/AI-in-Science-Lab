from typing import Any, NamedTuple

import torch
import torch.nn as nn


class FeatureMapEntry(NamedTuple):
    name: str
    feature_map: torch.Tensor
    conv: nn.Conv2d | None
    conv_input: torch.Tensor | None


class Module(NamedTuple):
    name: str
    tensor: torch.Tensor


class _RunningFeatureMapStats(NamedTuple):
    count: int
    mean: torch.Tensor
    m2: torch.Tensor
    min: torch.Tensor
    max: torch.Tensor


class LayerCaptureMixin:
    layer_outputs: dict[str, torch.Tensor] | None

    def _reset_layer_outputs(self) -> None:
        """Start a fresh last-forward feature-map snapshot."""
        self.layer_outputs = {}

    def _save_layer_output(self, name: str, output: torch.Tensor) -> None:
        """Store the last layer output and update running per-location statistics."""
        if not isinstance(output, torch.Tensor):
            return
        self._ensure_feature_map_stats()
        snapshot = output.detach().cpu()
        if self.layer_outputs is not None:
            self.layer_outputs[name] = snapshot
        self._update_feature_map_stats(name, snapshot)

    def _ensure_feature_map_stats(self) -> None:
        if not hasattr(self, "_feature_map_stats"):
            self._feature_map_stats: dict[str, _RunningFeatureMapStats] = {}

    def _batch_feature_map_stats(
        self,
        tensor: torch.Tensor,
    ) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        values = tensor.detach().to(torch.float32).cpu()
        if values.ndim == 0:
            zeros = torch.zeros_like(values)
            return 1, values, zeros, values, values

        batch_size = int(values.shape[0])
        if batch_size == 0:
            raise ValueError("Cannot update feature-map stats from an empty batch.")

        batch_mean = values.mean(dim=0)
        batch_m2 = ((values - batch_mean) ** 2).sum(dim=0)
        batch_min = values.amin(dim=0)
        batch_max = values.amax(dim=0)
        return batch_size, batch_mean, batch_m2, batch_min, batch_max

    def _update_feature_map_stats(self, name: str, tensor: torch.Tensor) -> None:
        batch_count, batch_mean, batch_m2, batch_min, batch_max = self._batch_feature_map_stats(tensor)
        current = self._feature_map_stats.get(name)

        if current is None or current.mean.shape != batch_mean.shape:
            self._feature_map_stats[name] = _RunningFeatureMapStats(
                count=batch_count,
                mean=batch_mean,
                m2=batch_m2,
                min=batch_min,
                max=batch_max,
            )
            return

        total_count = current.count + batch_count
        delta = batch_mean - current.mean
        mean = current.mean + delta * (batch_count / total_count)
        m2 = current.m2 + batch_m2 + (delta ** 2) * (current.count * batch_count / total_count)
        self._feature_map_stats[name] = _RunningFeatureMapStats(
            count=total_count,
            mean=mean,
            m2=m2,
            min=torch.minimum(current.min, batch_min),
            max=torch.maximum(current.max, batch_max),
        )

    def get_weights(self) -> list[Module]:
        """Return named weight tensors in network order."""
        return [
            Module(name, parameter.detach())
            for name, parameter in self.named_parameters()
            if name.endswith("weight")
        ]

    def get_feature_map_stat(self, stat: str) -> list[Module]:
        """Return named running feature-map statistics."""
        self._ensure_feature_map_stats()
        if stat not in {"min", "max", "mean", "var"}:
            raise ValueError("stat must be one of 'min', 'max', 'mean', or 'var'.")

        modules: list[Module] = []
        for name, stats in self._feature_map_stats.items():
            if stat == "min":
                tensor = stats.min
            elif stat == "max":
                tensor = stats.max
            elif stat == "mean":
                tensor = stats.mean
            else:
                tensor = stats.m2 / max(1, stats.count)
            modules.append(Module(name, tensor.detach()))
        return modules

    def clear_stats(self) -> None:
        """Forget running feature-map mean, variance, min, and max."""
        self._feature_map_stats = {}


def get_num_classes(dataset: str) -> int:
    """Map supported dataset names to their classifier output widths."""
    match dataset.removesuffix("_patches"):
        case "cifar10" | "smallcifar10":
            return 10
        case "cifar100":
            return 100
        case "imagenet":
            return 1000
        case _:
            raise ValueError(
                "dataset must be 'cifar10', 'cifar100', 'imagenet', "
                "'smallcifar10', 'cifar10_patches', 'cifar100_patches', "
                "'smallcifar10_patches', or 'imagenet_patches'."
            )


def get_input_dim(dataset: str) -> tuple[int, int, int]:
    """Map supported dataset names to model input shapes."""
    match dataset:
        case "cifar10" | "smallcifar10" | "cifar100":
            return (3, 32, 32)
        case "imagenet":
            return (3, 224, 224)
        case "cifar10_patches" | "smallcifar10_patches" | "cifar100_patches" | "imagenet_patches":
            return (3, 8, 8)
        case _:
            raise ValueError(
                "dataset must be 'cifar10', 'cifar100', 'imagenet', "
                "'smallcifar10', 'cifar10_patches', 'cifar100_patches', "
                "'smallcifar10_patches', or 'imagenet_patches'."
            )


def _first_hidden_channel(cfg: dict[str, Any]) -> int:
    hidden_channels = cfg["hidden_channels"]
    if isinstance(hidden_channels, int):
        return int(hidden_channels)
    if not hidden_channels:
        raise ValueError("hidden_channels must contain at least one layer width.")
    return int(hidden_channels[0])


def _gsa_hidden_channels(cfg: dict[str, Any]) -> int | list[int | str] | tuple[int | str, ...]:
    return cfg["gsa_hidden_channels"] if "gsa_hidden_channels" in cfg else cfg["hidden_channels"]


def get_model(cfg: dict[str, Any]) -> nn.Module:
    """Build the requested model from the experiment config dictionary."""
    from .densenet import TorchvisionDenseNet121
    from .greedy_stacked_autoencoder import GreedyStackedAutoencoder
    from .resnet import TorchvisionResNet18
    from .vgg import TorchvisionVGG16
    from .wta_conv_ae import WTA_CONV_AE

    num_classes = get_num_classes(cfg["dataset"])
    input_dim = get_input_dim(cfg["dataset"])
    architecture_type = cfg["architecture_type"]
    weights = str(cfg.get("weights", "random"))
    match weights:
        case "default":
            use_default_weights = True
        case "random":
            use_default_weights = False
        case _:
            raise ValueError("weights must be 'default' or 'random'.")
    if use_default_weights and architecture_type not in {"vgg16", "resnet18", "densenet121"}:
        raise ValueError(
            "weights='default' is only supported for architecture_type in "
            "{'vgg16', 'resnet18', 'densenet121'}."
        )

    match architecture_type:
        case "resnet18":
            return TorchvisionResNet18(
                num_classes=num_classes,
                pretrained=use_default_weights,
                freeze_backbone=False,
            )
        case "densenet121":
            return TorchvisionDenseNet121(
                num_classes=num_classes,
                pretrained=use_default_weights,
                freeze_backbone=False,
            )
        case "vgg16":
            return TorchvisionVGG16(
                num_classes=num_classes,
                dataset=str(cfg["dataset"]),
                pretrained=use_default_weights,
                freeze_backbone=False,
            )
        case "wta_conv_ae":
            return WTA_CONV_AE(
                dim=input_dim,
                hidden_channels=_first_hidden_channel(cfg),
                k_spatial=cfg.get("k_spatial"),
                k_population=cfg.get("k_population"),
                k_lifetime=cfg.get("k_lifetime"),
                total_epochs=int(cfg["epochs"]),
                dataset_size=int(cfg.get("dataset_size", 1)),
                a=float(cfg.get("k_population_alpha", 1.0)),
            )
        case "greedy_stacked_autoencoder":
            return GreedyStackedAutoencoder(
                dim=input_dim,
                hidden_channels=_gsa_hidden_channels(cfg),
                num_classes=num_classes if cfg["training_mode"] == "classification" else None,
                k_spatial=cfg.get("k_spatial"),
                k_population=cfg.get("k_population"),
                k_lifetime=cfg.get("k_lifetime"),
                total_epochs=int(cfg["epochs"]),
                dataset_size=int(cfg.get("dataset_size", 1)),
                a=float(cfg.get("k_population_alpha", 1.0)),
                local_training=bool(cfg.get("gsa_local_training", False)),
            )
        case _:
            raise ValueError(f"Unknown architecture_type={architecture_type!r}.")
