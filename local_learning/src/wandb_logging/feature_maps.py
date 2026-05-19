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
        stats[f"FeatureMean/{safe_name}"] = fmap.mean(dim=(0, 2, 3)).cpu()
        stats[f"FeatureMin/{safe_name}"] = fmap.amin(dim=(0, 2, 3)).cpu()
        stats[f"FeatureMax/{safe_name}"] = fmap.amax(dim=(0, 2, 3)).cpu()
        stats[f"FeatureVariance/{safe_name}"] = fmap.var(dim=(0, 2, 3), unbiased=False).cpu()

    return stats


def _safe_layer_name(layer_name: str) -> str:
    return layer_name.replace(".", "__").replace("/", "__")


class ChannelActivationScatterLogger:
    """Collect per-image channel means and build per-epoch channel-pair scatter plots."""

    def __init__(self, wandb_module: Any, *, prefix: str = "ActivationChannelScatter") -> None:
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
        """Return WandB scatter plots for every channel pair in every recorded layer."""
        payload: dict[str, Any] = {}

        for layer_name, batches in self._layer_batches.items():
            if not batches:
                continue

            activations = torch.cat(batches, dim=0)
            if activations.ndim != 2 or activations.shape[0] == 0 or activations.shape[1] < 2:
                continue

            safe_name = _safe_layer_name(layer_name)
            columns = [f"c{channel_idx}" for channel_idx in range(activations.shape[1])]
            table = self.wandb.Table(
                columns=columns,
                data=activations.tolist(),
            )
            payload[f"{self.prefix}/{safe_name}/epoch_{epoch:03d}_channel_means"] = table

            for channel_a in range(activations.shape[1]):
                for channel_b in range(channel_a + 1, activations.shape[1]):
                    x_col = columns[channel_a]
                    y_col = columns[channel_b]
                    payload[f"{self.prefix}/{safe_name}/epoch_{epoch:03d}_{x_col}_vs_{y_col}"] = (
                        self.wandb.plot.scatter(
                            table,
                            x_col,
                            y_col,
                            title=f"{safe_name} epoch {epoch}: {x_col} vs {y_col}",
                        )
                    )

        self.reset()
        return payload

    def reset(self) -> None:
        self._layer_batches.clear()
