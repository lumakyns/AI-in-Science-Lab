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


def _kde_curve(values: torch.Tensor, *, points: int = 64) -> tuple[list[float], list[float]]:
    """Estimate a Gaussian KDE over a 1D tensor of values."""
    vals = values.detach().to(torch.float32).flatten().cpu()
    vals = vals[torch.isfinite(vals)]
    if vals.numel() == 0 or points <= 0:
        return [], []

    std = vals.std(unbiased=False)
    if vals.numel() == 1 or float(std) == 0.0:
        scale = torch.clamp(vals.abs().mean() * 1e-3, min=torch.tensor(1e-6))
        bandwidth = scale
    else:
        bandwidth = 1.06 * std * (vals.numel() ** -0.2)
        bandwidth = torch.clamp(bandwidth, min=torch.tensor(1e-6))

    center_min = vals.amin()
    center_max = vals.amax()
    xs = torch.linspace(
        float(center_min - 3.0 * bandwidth),
        float(center_max + 3.0 * bandwidth),
        max(2, points),
    )
    density = torch.exp(-0.5 * ((xs[:, None] - vals[None, :]) / bandwidth) ** 2)
    density = density.mean(dim=1) / (bandwidth * (2.0 * torch.pi) ** 0.5)
    return xs.tolist(), density.tolist()


def log_weight_filter_grids(model: nn.Module, wandb_module: Any, *, prefix: str = "filter_grid") -> dict[str, Any]:
    """Create WandB images for the first 25 filters in every conv/deconv layer."""
    payload: dict[str, Any] = {}

    for module_name, module in _conv_modules(model):
        filter_grid = _make_filter_grid(_conv_filter_bank(module))
        image = filter_grid.permute(1, 2, 0).numpy()
        layer_type = "deconv" if isinstance(module, nn.ConvTranspose2d) else "conv"
        safe_name = _safe_layer_name(module_name, module)
        payload[f"media/{prefix}_{safe_name}_{layer_type}"] = wandb_module.Image(image)

    return payload


def log_conv_weight_channel_stats(model: nn.Module) -> dict[str, torch.Tensor]:
    """Log per-filter max, min, mean, variance, and norm for every conv/deconv layer."""
    payload: dict[str, torch.Tensor] = {}

    for module_name, module in _conv_modules(model):
        values = _channel_values(_conv_filter_bank(module))
        safe_name = _safe_layer_name(module_name, module)
        payload[f"weights/max_{safe_name}"] = values.amax(dim=1)
        payload[f"weights/min_{safe_name}"] = values.amin(dim=1)
        payload[f"weights/mean_{safe_name}"] = values.mean(dim=1)
        payload[f"weights/variance_{safe_name}"] = values.var(dim=1, unbiased=False)
        payload[f"weights/norm_{safe_name}"] = values.norm(dim=1)

    return payload


def log_conv_gradient_channel_stats(model: nn.Module, *, prefix: str = "magnitude") -> dict[str, torch.Tensor]:
    """Log the per-filter gradient magnitude for every conv/deconv layer with gradients."""
    payload: dict[str, torch.Tensor] = {}

    for module_name, module in _conv_modules(model):
        if module.weight.grad is None:
            continue
        grad_values = _channel_values(module.weight.grad)
        safe_name = _safe_layer_name(module_name, module)
        payload[f"gradients/{prefix}_{safe_name}"] = grad_values.norm(dim=1)

    return payload


def log_conv_norm_kdes(model: nn.Module, wandb_module: Any, *, prefix: str = "norm_kde") -> dict[str, Any]:
    """Log a simple KDE of filter norms for every conv/deconv layer."""
    payload: dict[str, Any] = {}

    for module_name, module in _conv_modules(model):
        norms = _channel_values(_conv_filter_bank(module)).norm(dim=1)
        xs, ys = _kde_curve(norms)
        if not xs:
            continue
        safe_name = _safe_layer_name(module_name, module)
        payload[f"weights/{prefix}_{safe_name}"] = wandb_module.plot.line_series(
            xs=xs,
            ys=[ys],
            keys=["norm_kde"],
            title=f"{safe_name} filter norm KDE",
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
        update_prefix: str = "relative_update",
        init_drift_prefix: str = "cosine_drift_from_init",
        last_drift_prefix: str = "cosine_drift_since_last_log",
    ) -> dict[str, float]:
        payload: dict[str, float] = {}
        current_weights = self._snapshot(model)

        for module_name, current_weight in current_weights.items():
            previous_weight = self._previous_weights.get(module_name)
            initial_weight = self._initial_weights.get(module_name)
            if previous_weight is None or initial_weight is None:
                continue

            safe_name = module_name.replace(".", "__") or "conv"
            payload[f"weights/{update_prefix}_{safe_name}"] = _relative_update(
                current_weight,
                previous_weight,
            )
            payload[f"weights/{init_drift_prefix}_{safe_name}"] = _cosine_drift(
                current_weight,
                initial_weight,
            )
            payload[f"weights/{last_drift_prefix}_{safe_name}"] = _cosine_drift(
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
    prefix: str = "flops",
) -> dict[str, int]:
    """Run one eval forward pass with hooks and log estimated conv/deconv/linear FLOPs."""
    payload: dict[str, int] = {}
    handles = []
    was_training = model.training

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

    payload["flops/total"] = int(sum(payload.values()))
    return payload
