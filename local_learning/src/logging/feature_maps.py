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
        stats[f"{layer_name}/channel_mean"] = fmap.mean(dim=(0, 2, 3)).cpu()
        stats[f"{layer_name}/channel_min"] = fmap.amin(dim=(0, 2, 3)).cpu()
        stats[f"{layer_name}/channel_max"] = fmap.amax(dim=(0, 2, 3)).cpu()

    return stats

