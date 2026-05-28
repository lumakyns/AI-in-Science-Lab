from typing import Any

import torch
import torch.nn as nn

ConvLike = nn.Conv2d | nn.ConvTranspose2d


def _conv_modules(model: nn.Module) -> list[tuple[str, ConvLike]]:
    """Collect every convolution and transpose convolution module in model order."""
    return [
        (module_name, module)
        for module_name, module in model.named_modules()
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d))
    ]


def _safe_layer_name(module_name: str, module: ConvLike) -> str:
    """Convert a module path to a WandB-friendly layer name."""
    layer_type = "deconv" if isinstance(module, nn.ConvTranspose2d) else "conv"
    return module_name.replace(".", "__") or layer_type


def _normalize_filter(filter_tensor: torch.Tensor) -> torch.Tensor:
    """Normalize one filter image into [0, 1] for display."""
    filt = filter_tensor.detach().to(torch.float32).cpu()
    filt = filt - filt.amin()
    denom = filt.amax()
    if float(denom) > 0:
        filt = filt / denom
    return filt


def _filter_to_image(filter_tensor: torch.Tensor) -> torch.Tensor:
    """Convert one convolutional filter to a 3-channel image tensor."""
    if filter_tensor.ndim != 3:
        raise ValueError(f"Expected filter shape [C, H, W], got {tuple(filter_tensor.shape)}.")

    if filter_tensor.shape[0] == 1:
        image = filter_tensor.repeat(3, 1, 1)
    elif filter_tensor.shape[0] == 3:
        image = filter_tensor
    else:
        image = filter_tensor.mean(dim=0, keepdim=True).repeat(3, 1, 1)
    return _normalize_filter(image)


def _conv_filter_bank(module: ConvLike) -> torch.Tensor:
    """Return weights as [num_filters, channels, height, width] for conv and deconv layers."""
    weight = module.weight.detach()
    if isinstance(module, nn.ConvTranspose2d):
        return weight
    return weight


def _make_filter_grid(filters: torch.Tensor, *, grid_size: int = 5) -> torch.Tensor:
    """Lay the first filters into a fixed square image grid."""
    max_filters = grid_size * grid_size
    selected = filters[:max_filters]
    images = [_filter_to_image(filt) for filt in selected]

    if not images:
        raise ValueError("Cannot log an empty filter bank.")

    channels, height, width = images[0].shape
    grid = torch.zeros(channels, grid_size * height, grid_size * width)
    for idx, image in enumerate(images):
        row = idx // grid_size
        col = idx % grid_size
        grid[:, row * height : (row + 1) * height, col * width : (col + 1) * width] = image
    return grid


def _channel_values(tensor: torch.Tensor) -> torch.Tensor:
    """Flatten each output/filter channel to one row for per-channel statistics."""
    return tensor.detach().to(torch.float32).flatten(start_dim=1).cpu()


def _image_channel_distribution_stats(tensor: torch.Tensor) -> dict[str, float]:
    """Summarize spatial means/stddevs across images first, then channels."""
    image_channel_values = tensor.detach().to(torch.float32).flatten(start_dim=2).cpu()
    spatial_means = image_channel_values.mean(dim=2)
    image_means = spatial_means.mean(dim=0)
    image_stds = spatial_means.std(dim=0, unbiased=False)
    return {
        "mean_mean": float(image_means.mean()),
        "mean_stddev": float(image_means.std(unbiased=False)),
        "stddev_mean": float(image_stds.mean()),
        "stddev_stddev": float(image_stds.std(unbiased=False)),
    }


def _kde_density_on_x(values: torch.Tensor, xs: torch.Tensor) -> list[float]:
    """Estimate a KDE for values on a shared x-axis."""
    vals = values.detach().to(torch.float32).flatten().cpu()
    vals = vals[torch.isfinite(vals)]
    if vals.numel() == 0 or xs.numel() == 0:
        return []

    std = vals.std(unbiased=False)
    if vals.numel() == 1 or float(std) == 0.0:
        scale = torch.clamp(vals.abs().mean() * 1e-3, min=torch.tensor(1e-6))
        bandwidth = scale
    else:
        bandwidth = 1.06 * std * (vals.numel() ** -0.2)
        bandwidth = torch.clamp(bandwidth, min=torch.tensor(1e-6))

    density = torch.exp(-0.5 * ((xs[:, None] - vals[None, :]) / bandwidth) ** 2)
    density = density.mean(dim=1) / (bandwidth * (2.0 * torch.pi) ** 0.5)
    return density.tolist()


def log_conv_weight_snapshot(
    model: nn.Module,
    wandb_module: Any,
    norm_kde_history_logger: "ConvNormKDEHistoryLogger",
    *,
    step: int,
    filter_grid_prefix: str = "viz-test-filter-grid",
    distribution_prefix: str = "viz-train-encoder-weight-distribution",
    gradient_prefix: str = "viz-train-gradient-distribution",
) -> dict[str, Any]:
    """Log weight diagnostics in one pass over conv/deconv modules."""
    payload: dict[str, Any] = {}

    for module_name, module in _conv_modules(model):
        filter_bank = _conv_filter_bank(module)
        safe_name = _safe_layer_name(module_name, module)

        filter_grid = _make_filter_grid(filter_bank)
        image = filter_grid.permute(1, 2, 0).numpy()
        layer_type = "deconv" if isinstance(module, nn.ConvTranspose2d) else "conv"
        payload[f"{filter_grid_prefix}/{safe_name}_{layer_type}"] = wandb_module.Image(image)

        values = _channel_values(filter_bank)
        norms = values.norm(dim=1)

        if isinstance(module, nn.Conv2d):
            for stat_name, stat_value in _image_channel_distribution_stats(filter_bank).items():
                payload[f"{distribution_prefix}/{stat_name}/{safe_name}"] = stat_value

        if module.weight.grad is not None:
            for stat_name, stat_value in _image_channel_distribution_stats(module.weight.grad).items():
                payload[f"{gradient_prefix}/{stat_name}/{safe_name}"] = stat_value

        payload.update(norm_kde_history_logger.log_norms(safe_name, norms, step=step))

    return payload


class ConvNormKDEHistoryLogger:
    """Track filter norm KDEs over training and log per-layer overlay plots."""

    def __init__(
        self,
        wandb_module: Any,
        *,
        prefix: str = "viz-train-filter-norm-kde-progress",
        max_snapshots: int = 12,
        points: int = 64,
    ) -> None:
        self.wandb = wandb_module
        self.prefix = prefix
        self.max_snapshots = max(1, int(max_snapshots))
        self.points = max(2, int(points))
        self._history: dict[str, list[tuple[int, torch.Tensor]]] = {}

    def log(self, model: nn.Module, *, step: int) -> dict[str, Any]:
        payload: dict[str, Any] = {}

        for module_name, module in _conv_modules(model):
            safe_name = _safe_layer_name(module_name, module)
            norms = _channel_values(_conv_filter_bank(module)).norm(dim=1)
            payload.update(self.log_norms(safe_name, norms, step=step))

        return payload

    def log_norms(self, safe_name: str, norms: torch.Tensor, *, step: int) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        finite_norms = norms.detach().to(torch.float32).cpu()
        finite_norms = finite_norms[torch.isfinite(finite_norms)]
        if finite_norms.numel() == 0:
            return payload

        layer_history = self._history.setdefault(safe_name, [])
        layer_history.append((int(step), finite_norms))
        del layer_history[:-self.max_snapshots]

        all_values = torch.cat([snapshot_norms for _, snapshot_norms in layer_history])
        if all_values.numel() == 0:
            return payload

        std = all_values.std(unbiased=False)
        padding = torch.clamp(all_values.abs().mean() * 1e-3, min=torch.tensor(1e-6))
        if float(std) > 0.0:
            padding = torch.clamp(3.0 * std, min=padding)
        xs = torch.linspace(
            float(all_values.amin() - padding),
            float(all_values.amax() + padding),
            self.points,
        )
        ys = [_kde_density_on_x(snapshot_norms, xs) for _, snapshot_norms in layer_history]
        if not ys or any(not y for y in ys):
            return payload

        payload[f"{self.prefix}/{safe_name}"] = self.wandb.plot.line_series(
            xs=xs.tolist(),
            ys=ys,
            keys=[f"step_{snapshot_step}" for snapshot_step, _ in layer_history],
            title=f"{safe_name} filter norm KDE over training",
            xname="filter_norm",
        )
        return payload


class ConvWeightChangeLogger:
    """Track conv/deconv weight update size and cosine drift from snapshots."""

    def __init__(self, model: nn.Module) -> None:
        self._initial_weights = self._snapshot(model)
        self._previous_weights = {
            name: weight.clone()
            for name, weight in self._initial_weights.items()
        }

    def log(
        self,
        model: nn.Module,
        *,
        update_prefix: str = "viz-train-relative-update",
        init_drift_prefix: str = "viz-train-cosine-drift-from-init",
        last_drift_prefix: str = "viz-train-cosine-drift-since-last-log",
    ) -> dict[str, float]:
        payload: dict[str, float] = {}
        current_weights = self._snapshot(model)

        for module_name, current_weight in current_weights.items():
            previous_weight = self._previous_weights.get(module_name)
            initial_weight = self._initial_weights.get(module_name)
            if previous_weight is None or initial_weight is None:
                continue

            safe_name = module_name.replace(".", "__") or "conv"
            payload[f"{update_prefix}/{safe_name}"] = _relative_update(
                current_weight,
                previous_weight,
            )
            payload[f"{init_drift_prefix}/{safe_name}"] = _cosine_drift(
                current_weight,
                initial_weight,
            )
            payload[f"{last_drift_prefix}/{safe_name}"] = _cosine_drift(
                current_weight,
                previous_weight,
            )

        self._previous_weights = {
            name: weight.clone()
            for name, weight in current_weights.items()
        }
        return payload

    @staticmethod
    def _snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
        return {
            module_name: _conv_filter_bank(module).detach().to(torch.float32).flatten().cpu()
            for module_name, module in _conv_modules(model)
        }


def _relative_update(current_weight: torch.Tensor, previous_weight: torch.Tensor) -> float:
    denom = previous_weight.norm().clamp_min(1e-12)
    return float((current_weight - previous_weight).norm() / denom)


def _cosine_drift(current_weight: torch.Tensor, reference_weight: torch.Tensor) -> float:
    denom = current_weight.norm() * reference_weight.norm()
    if float(denom) == 0.0:
        return 0.0
    cosine_similarity = torch.dot(current_weight, reference_weight) / denom
    return float(1.0 - cosine_similarity.clamp(-1.0, 1.0))


def _conv2d_flops(module: nn.Conv2d, output: torch.Tensor) -> int:
    """Estimate multiply-add FLOPs for one Conv2d output tensor."""
    batch_size, out_channels, out_h, out_w = output.shape
    kernel_h, kernel_w = module.kernel_size
    in_channels = module.in_channels // module.groups
    return int(batch_size * out_channels * out_h * out_w * in_channels * kernel_h * kernel_w * 2)


def _conv_transpose2d_flops(module: nn.ConvTranspose2d, output: torch.Tensor) -> int:
    """Estimate multiply-add FLOPs for one ConvTranspose2d output tensor."""
    batch_size, out_channels, out_h, out_w = output.shape
    kernel_h, kernel_w = module.kernel_size
    in_channels = module.in_channels // module.groups
    return int(batch_size * out_channels * out_h * out_w * in_channels * kernel_h * kernel_w * 2)


def _linear_flops(module: nn.Linear, output: torch.Tensor) -> int:
    """Estimate multiply-add FLOPs for one Linear output tensor."""
    batch_size = int(output.shape[0]) if output.ndim > 1 else 1
    return int(batch_size * module.in_features * module.out_features * 2)


def log_inference_flops(
    model: nn.Module,
    sample_input: torch.Tensor,
    *,
    forward_kwargs: dict[str, Any] | None = None,
    prefix: str = "viz-test-flops",
) -> dict[str, int]:
    """Run one eval forward pass with hooks and log estimated conv/deconv/linear FLOPs."""
    payload: dict[str, int] = {}
    handles = []
    was_training = model.training
    sample_input = sample_input[:1].detach()

    def hook_for(module_name: str, module: nn.Module):
        def hook(_module, _inputs, output) -> None:
            if not isinstance(output, torch.Tensor):
                return
            safe_name = module_name.replace(".", "__") or module.__class__.__name__.lower()
            if isinstance(module, nn.Conv2d):
                payload[f"{prefix}/{safe_name}"] = _conv2d_flops(module, output)
            elif isinstance(module, nn.ConvTranspose2d):
                payload[f"{prefix}/{safe_name}"] = _conv_transpose2d_flops(module, output)
            elif isinstance(module, nn.Linear):
                payload[f"{prefix}/{safe_name}"] = _linear_flops(module, output)

        return hook

    for module_name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            handles.append(module.register_forward_hook(hook_for(module_name, module)))

    try:
        model.eval()
        with torch.no_grad():
            model(sample_input, **(forward_kwargs or {}))
    finally:
        for handle in handles:
            handle.remove()
        if was_training:
            model.train()

    payload[f"{prefix}/total"] = int(sum(payload.values()))
    return payload
