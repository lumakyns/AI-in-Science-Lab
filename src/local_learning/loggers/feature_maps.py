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


def _safe_layer_name(layer_name: str) -> str:
    return layer_name.replace(".", "__").replace("/", "__")


def _mean_channel_distance_to_geometric_mean(feature_map: torch.Tensor) -> torch.Tensor:
    """Measure how far layer channels are, on average, from the channel centroid."""
    channel_vectors = feature_map.to(torch.float32).permute(1, 0, 2, 3).flatten(start_dim=1)
    if channel_vectors.numel() == 0:
        return torch.tensor(0.0)

    geometric_mean = channel_vectors.mean(dim=0, keepdim=True)
    return (channel_vectors - geometric_mean).norm(dim=1).mean().cpu()


class ChannelActivationStatsLogger:
    """Collect epoch-level activation sparsity and channel collapse diagnostics."""

    def __init__(
        self,
        *,
        prefix: str = "viz-test-activation-epoch",
        distance_prefix: str = "viz-test-activation-geometric-mean-distance",
        zero_threshold: float = 1e-6,
        dead_active_fraction: float = 0.01,
        dominant_share_multiplier: float = 2.0,
    ) -> None:
        self.prefix = prefix
        self.distance_prefix = distance_prefix
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
            mean_geometric_distance = _mean_channel_distance_to_geometric_mean(fmap)

            if layer_name not in self._stats:
                self._stats[layer_name] = {
                    "zero_count": zero_count,
                    "total_count": total_count,
                    "activation_mass": activation_mass,
                    "geometric_distance_sum": mean_geometric_distance,
                    "geometric_distance_count": torch.tensor(1),
                }
                continue

            self._stats[layer_name]["zero_count"] += zero_count
            self._stats[layer_name]["total_count"] += total_count
            self._stats[layer_name]["activation_mass"] += activation_mass
            self._stats[layer_name]["geometric_distance_sum"] += mean_geometric_distance
            self._stats[layer_name]["geometric_distance_count"] += 1

    def log_epoch(self) -> dict[str, torch.Tensor | int | float]:
        payload: dict[str, torch.Tensor | int | float] = {}

        for layer_name, stats in self._stats.items():
            safe_name = _safe_layer_name(layer_name)
            zero_count = stats["zero_count"]
            total_count = stats["total_count"].clamp_min(1)
            activation_mass = stats["activation_mass"]
            mean_geometric_distance = stats["geometric_distance_sum"] / stats[
                "geometric_distance_count"
            ].clamp_min(1)

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

            payload[f"{self.prefix}-sparsity/{safe_name}"] = sparsity
            payload[f"{self.prefix}-active-fraction/{safe_name}"] = active_fraction
            payload[f"{self.prefix}-dead-channels/{safe_name}"] = int(dead_channels.sum())
            payload[f"{self.prefix}-dominant-channels/{safe_name}"] = int(dominant_channels.sum())
            payload[f"{self.prefix}-dominant-mass-share/{safe_name}"] = float(
                mass_share[dominant_channels].sum()
            )
            payload[f"{self.distance_prefix}/{safe_name}"] = float(mean_geometric_distance)

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


def _sample_channel_pairs(
    channel_count: int,
    *,
    max_pairs: int,
    seed: int,
) -> list[tuple[int, int]]:
    pair_count = (channel_count * (channel_count - 1)) // 2
    if pair_count <= max_pairs:
        return [(a, b) for a in range(channel_count) for b in range(a + 1, channel_count)]

    generator = torch.Generator().manual_seed(int(seed))
    pairs: set[tuple[int, int]] = set()
    while len(pairs) < max_pairs:
        sampled = torch.randint(channel_count, (max_pairs * 4, 2), generator=generator)
        for channel_a, channel_b in sampled.tolist():
            if channel_a == channel_b:
                continue
            if channel_a > channel_b:
                channel_a, channel_b = channel_b, channel_a
            pairs.add((channel_a, channel_b))
            if len(pairs) == max_pairs:
                break

    return sorted(pairs)


def _make_sampled_channel_pair_scatter_grid(
    channel_means: torch.Tensor,
    *,
    seed: int,
    grid_size: int = 5,
    cell_size: int = 72,
    padding: int = 8,
) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    if channel_means.ndim != 2 or channel_means.shape[1] < 2:
        raise ValueError("Cannot build a scatter grid with fewer than two channels.")

    max_pairs = grid_size * grid_size
    pairs = _sample_channel_pairs(
        int(channel_means.shape[1]),
        max_pairs=max_pairs,
        seed=seed,
    )
    image = torch.ones(
        3,
        grid_size * cell_size,
        grid_size * cell_size,
        dtype=torch.float32,
    )

    for idx, (channel_a, channel_b) in enumerate(pairs):
        _draw_scatter_cell(
            image,
            row=idx // grid_size,
            col=idx % grid_size,
            x=channel_means[:, channel_a],
            y=channel_means[:, channel_b],
            cell_size=cell_size,
            padding=padding,
        )

    return image, pairs


def _scatter_seed(layer_name: str, epoch: int | str) -> int:
    seed_text = f"{layer_name}:{epoch}"
    return sum((idx + 1) * ord(char) for idx, char in enumerate(seed_text)) % (2**31)


class ChannelActivationScatterLogger:
    """Collect image-level channel means and build sampled channel-pair scatters."""

    def __init__(
        self,
        wandb_module: Any,
        *,
        prefix: str = "viz-test-activation-channel-pair-scatter",
        max_points: int = 256,
    ) -> None:
        self.wandb = wandb_module
        self.prefix = prefix
        self.max_points = max(1, int(max_points))
        self._layer_batches: dict[str, list[torch.Tensor]] = {}

    def update(self, feature_maps: list) -> None:
        """Store each image's mean activation per channel for the current batch."""
        for idx, feature_map_item in enumerate(feature_maps):
            layer_name, feature_map = _parse_feature_map(feature_map_item, idx=idx)
            if feature_map.ndim != 4 or feature_map.shape[1] < 2:
                continue
            channel_means = feature_map.detach().to(torch.float32).mean(dim=(2, 3)).cpu()
            self._layer_batches.setdefault(layer_name, []).append(channel_means)

    def log_epoch(self, *, epoch: int | str) -> dict[str, Any]:
        """Return one sampled 5x5 channel-pair scatter grid per layer."""
        payload: dict[str, Any] = {}

        for layer_name, batches in self._layer_batches.items():
            if not batches:
                continue

            channel_means = torch.cat(batches, dim=0)
            if channel_means.shape[0] == 0:
                continue

            channel_means = channel_means[: self.max_points]
            safe_name = _safe_layer_name(layer_name)
            scatter, pairs = _make_sampled_channel_pair_scatter_grid(
                channel_means,
                seed=_scatter_seed(safe_name, epoch),
            )
            image = scatter.permute(1, 2, 0).numpy()
            payload[f"{self.prefix}/{safe_name}"] = self.wandb.Image(
                image,
                caption=(
                    f"{safe_name} epoch {epoch}: sampled {len(pairs)} channel-pair "
                    "activation scatters on a 5x5 grid"
                ),
            )

        self.reset()
        return payload

    def reset(self) -> None:
        self._layer_batches.clear()


class FeatureMapChannelLineLogger:
    """Log per-channel feature-map mean/stddev histories as one line plot per layer."""

    def __init__(
        self,
        wandb_module: Any,
        *,
        mean_prefix: str = "viz-train-feature-map-channel-mean",
        std_prefix: str = "viz-train-feature-map-channel-std",
        max_snapshots: int = 64,
    ) -> None:
        self.wandb = wandb_module
        self.mean_prefix = mean_prefix
        self.std_prefix = std_prefix
        self.max_snapshots = max(1, int(max_snapshots))
        self._mean_history: dict[str, list[tuple[int, torch.Tensor]]] = {}
        self._std_history: dict[str, list[tuple[int, torch.Tensor]]] = {}

    def log_step(self, feature_maps: list, *, step: int) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for idx, feature_map_item in enumerate(feature_maps):
            layer_name, feature_map = _parse_feature_map(feature_map_item, idx=idx)
            if feature_map.ndim != 4:
                continue
            safe_name = _safe_layer_name(layer_name)
            fmap = feature_map.detach().to(torch.float32)
            channel_values = fmap.flatten(start_dim=2)
            channel_means = channel_values.mean(dim=(0, 2)).cpu()
            channel_stds = channel_values.std(dim=(0, 2), unbiased=False).cpu()
            payload.update(
                self._log_values(
                    self._mean_history,
                    self.mean_prefix,
                    safe_name,
                    channel_means,
                    step=step,
                    title=f"{safe_name} feature-map mean by channel",
                    yname="feature_map_mean",
                )
            )
            payload.update(
                self._log_values(
                    self._std_history,
                    self.std_prefix,
                    safe_name,
                    channel_stds,
                    step=step,
                    title=f"{safe_name} feature-map stddev by channel",
                    yname="feature_map_stddev",
                )
            )
        return payload

    def _log_values(
        self,
        history: dict[str, list[tuple[int, torch.Tensor]]],
        prefix: str,
        safe_name: str,
        values: torch.Tensor,
        *,
        step: int,
        title: str,
        yname: str,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        channel_values = values.detach().to(torch.float32).flatten().cpu()
        if channel_values.numel() == 0:
            return payload

        layer_history = history.setdefault(safe_name, [])
        if layer_history and layer_history[-1][1].numel() != channel_values.numel():
            layer_history.clear()
        layer_history.append((int(step), channel_values))
        del layer_history[:-self.max_snapshots]

        xs = [snapshot_step for snapshot_step, _ in layer_history]
        ys = [
            [float(snapshot_values[channel_idx]) for _, snapshot_values in layer_history]
            for channel_idx in range(int(channel_values.numel()))
        ]
        payload[f"{prefix}/{safe_name}"] = self.wandb.plot.line_series(
            xs=xs,
            ys=ys,
            keys=[f"channel_{idx}" for idx in range(int(channel_values.numel()))],
            title=title,
            xname="step",
            yname=yname,
        )
        return payload
