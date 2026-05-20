from typing import Any

import torch


def _parse_feature_map(feature_map_item, *, idx: int) -> tuple[str, torch.Tensor]:
    if isinstance(feature_map_item, torch.Tensor):
        return f"fm_{idx}", feature_map_item

    if hasattr(feature_map_item, "name") and hasattr(feature_map_item, "feature_map"):
        return feature_map_item.name, feature_map_item.feature_map

    if isinstance(feature_map_item, (tuple, list)) and len(feature_map_item) >= 2:
        return str(feature_map_item[0]), feature_map_item[1]

    raise TypeError(
        "feature maps must be tensors, FeatureMapEntry items, or tuples like (name, feature_map)."
    )


def log_channel_stats(feature_maps: list) -> dict[str, torch.Tensor]:
    stats: dict[str, torch.Tensor] = {}

    for idx, feature_map_item in enumerate(feature_maps):
        layer_name, feature_map = _parse_feature_map(feature_map_item, idx=idx)
        if feature_map.ndim != 4:
            raise ValueError(
                f"Expected feature map {layer_name!r} to have shape [B, C, H, W], "
                f"got {tuple(feature_map.shape)}."
            )

        fmap = feature_map.detach()
        safe_name = layer_name.replace(".", "__").replace("/", "__")
        stats[f"activations/mean_{safe_name}"] = fmap.mean(dim=(0, 2, 3)).cpu()
        stats[f"activations/min_{safe_name}"] = fmap.amin(dim=(0, 2, 3)).cpu()
        stats[f"activations/max_{safe_name}"] = fmap.amax(dim=(0, 2, 3)).cpu()
        stats[f"activations/variance_{safe_name}"] = fmap.var(dim=(0, 2, 3), unbiased=False).cpu()

    return stats


def _safe_layer_name(layer_name: str) -> str:
    return layer_name.replace(".", "__").replace("/", "__")


class ChannelActivationStatsLogger:
    """Collect epoch-level activation sparsity and channel collapse diagnostics."""

    def __init__(
        self,
        *,
        zero_threshold: float = 1e-6,
        dead_active_fraction: float = 0.01,
        dominant_share_multiplier: float = 2.0,
    ) -> None:
        self.zero_threshold = zero_threshold
        self.dead_active_fraction = dead_active_fraction
        self.dominant_share_multiplier = dominant_share_multiplier
        self._stats: dict[str, dict[str, torch.Tensor]] = {}

    def update(self, feature_maps: list) -> None:
        for idx, feature_map_item in enumerate(feature_maps):
            layer_name, feature_map = _parse_feature_map(feature_map_item, idx=idx)
            if feature_map.ndim != 4:
                continue

            fmap = feature_map.detach().to(torch.float32)
            reduce_dims = (0, 2, 3)
            zero_count = (fmap.abs() <= self.zero_threshold).sum(dim=reduce_dims).cpu()
            total_count = torch.full_like(
                zero_count,
                fmap.shape[0] * fmap.shape[2] * fmap.shape[3],
            )
            activation_mass = fmap.abs().sum(dim=reduce_dims).cpu()

            if layer_name not in self._stats:
                self._stats[layer_name] = {
                    "zero_count": zero_count,
                    "total_count": total_count,
                    "activation_mass": activation_mass,
                }
                continue

            self._stats[layer_name]["zero_count"] += zero_count
            self._stats[layer_name]["total_count"] += total_count
            self._stats[layer_name]["activation_mass"] += activation_mass

    def log_epoch(self) -> dict[str, torch.Tensor | int | float]:
        payload: dict[str, torch.Tensor | int | float] = {}

        for layer_name, stats in self._stats.items():
            safe_name = _safe_layer_name(layer_name)
            zero_count = stats["zero_count"]
            total_count = stats["total_count"].clamp_min(1)
            activation_mass = stats["activation_mass"]

            sparsity = zero_count / total_count
            active_fraction = 1.0 - sparsity
            mass_total = activation_mass.sum()
            if float(mass_total) > 0.0:
                mass_share = activation_mass / mass_total
            else:
                mass_share = torch.zeros_like(activation_mass)

            channel_count = max(1, int(activation_mass.numel()))
            dominant_threshold = self.dominant_share_multiplier / channel_count
            dead_channels = active_fraction <= self.dead_active_fraction
            dominant_channels = mass_share >= dominant_threshold

            payload[f"activations/sparsity_{safe_name}"] = sparsity
            payload[f"activations/active_fraction_{safe_name}"] = active_fraction
            payload[f"activations/dead_channels_{safe_name}"] = int(dead_channels.sum())
            payload[f"activations/dominant_channels_{safe_name}"] = int(dominant_channels.sum())
            payload[f"activations/dominant_mass_share_{safe_name}"] = float(
                mass_share[dominant_channels].sum()
            )

        self.reset()
        return payload

    def reset(self) -> None:
        self._stats.clear()


def _normalize_plot_values(values: torch.Tensor, *, low: int, high: int) -> torch.Tensor:
    finite_values = values[torch.isfinite(values)]
    if finite_values.numel() == 0:
        return torch.full_like(values, (low + high) / 2, dtype=torch.float32)

    min_value = finite_values.amin()
    max_value = finite_values.amax()
    if float(max_value - min_value) == 0.0:
        return torch.full_like(values, (low + high) / 2, dtype=torch.float32)

    normalized = (values - min_value) / (max_value - min_value)
    return low + normalized.clamp(0.0, 1.0) * (high - low)


def _draw_scatter_cell(
    image: torch.Tensor,
    *,
    row: int,
    col: int,
    x: torch.Tensor,
    y: torch.Tensor,
    cell_size: int,
    padding: int,
) -> None:
    y0 = row * cell_size
    x0 = col * cell_size
    image[:, y0 : y0 + cell_size, x0 : x0 + cell_size] = 1.0

    plot_min = padding
    plot_max = cell_size - padding - 1
    axis_color = torch.tensor([0.82, 0.84, 0.86], dtype=image.dtype)
    point_color = torch.tensor([0.05, 0.28, 0.65], dtype=image.dtype)

    image[:, y0 + plot_max, x0 + plot_min : x0 + plot_max + 1] = axis_color[:, None]
    image[:, y0 + plot_min : y0 + plot_max + 1, x0 + plot_min] = axis_color[:, None]

    finite = torch.isfinite(x) & torch.isfinite(y)
    if not bool(finite.any()):
        return

    px = (
        _normalize_plot_values(x[finite], low=plot_min + 1, high=plot_max - 1)
        .round()
        .to(torch.long)
    )
    py = (
        _normalize_plot_values(y[finite], low=plot_min + 1, high=plot_max - 1)
        .round()
        .to(torch.long)
    )
    py = plot_max - (py - plot_min)

    for point_x, point_y in zip(px.tolist(), py.tolist()):
        yy = y0 + point_y
        xx = x0 + point_x
        y_slice = slice(max(y0, yy - 1), min(y0 + cell_size, yy + 2))
        x_slice = slice(max(x0, xx - 1), min(x0 + cell_size, xx + 2))
        image[:, y_slice, x_slice] = point_color[:, None, None]


def _make_channel_scatter_grid(
    activations: torch.Tensor,
    *,
    cell_size: int = 72,
    padding: int = 8,
) -> torch.Tensor:
    channel_count = int(activations.shape[1])
    if channel_count < 2:
        raise ValueError("Cannot build a scatter grid with fewer than two channels.")

    grid_size = channel_count - 1
    image = torch.ones(
        3,
        grid_size * cell_size,
        grid_size * cell_size,
        dtype=torch.float32,
    )
    for channel_a in range(channel_count):
        for channel_b in range(channel_a + 1, channel_count):
            _draw_scatter_cell(
                image,
                row=channel_b - 1,
                col=channel_a,
                x=activations[:, channel_a],
                y=activations[:, channel_b],
                cell_size=cell_size,
                padding=padding,
            )

    return image


class ChannelActivationScatterLogger:
    """Collect per-image channel means and build one scatter-grid image per layer."""

    def __init__(self, wandb_module: Any, *, prefix: str = "activation_channel_scatter") -> None:
        self.wandb = wandb_module
        self.prefix = prefix
        self._layer_batches: dict[str, list[torch.Tensor]] = {}

    def update(self, feature_maps: list) -> None:
        """Store each image's mean activation per channel for the current batch."""
        for idx, feature_map_item in enumerate(feature_maps):
            layer_name, feature_map = _parse_feature_map(feature_map_item, idx=idx)
            if feature_map.ndim != 4:
                continue
            channel_means = feature_map.detach().to(torch.float32).mean(dim=(2, 3)).cpu()
            self._layer_batches.setdefault(layer_name, []).append(channel_means)

    def log_epoch(self, *, epoch: int) -> dict[str, Any]:
        """Return one WandB image per layer containing all channel-pair scatter plots."""
        payload: dict[str, Any] = {}

        for layer_name, batches in self._layer_batches.items():
            if not batches:
                continue

            activations = torch.cat(batches, dim=0)
            if activations.ndim != 2 or activations.shape[0] == 0 or activations.shape[1] < 2:
                continue

            safe_name = _safe_layer_name(layer_name)
            scatter_grid = _make_channel_scatter_grid(activations)
            image = scatter_grid.permute(1, 2, 0).numpy()
            payload[f"media/{self.prefix}_{safe_name}"] = self.wandb.Image(
                image,
                caption=(
                    f"{safe_name} epoch {epoch}: all channel mean scatter plots "
                    f"({activations.shape[1]} channels)"
                ),
            )

        self.reset()
        return payload

    def reset(self) -> None:
        self._layer_batches.clear()
