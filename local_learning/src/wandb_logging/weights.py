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
    """Estimate a simple Gaussian KDE over a 1D tensor of values."""
    vals = values.detach().to(torch.float32).flatten().cpu()
    if vals.numel() == 0:
        return [], []
    if vals.numel() == 1 or float(vals.std(unbiased=False)) == 0.0:
        center = float(vals.mean())
        return [center], [1.0]

    std = vals.std(unbiased=False)
    bandwidth = 1.06 * std * (vals.numel() ** -0.2)
    bandwidth = torch.clamp(bandwidth, min=torch.tensor(1e-6))
    xs = torch.linspace(float(vals.amin()), float(vals.amax()), points)
    density = torch.exp(-0.5 * ((xs[:, None] - vals[None, :]) / bandwidth) ** 2)
    density = density.mean(dim=1) / (bandwidth * (2.0 * torch.pi) ** 0.5)
    return xs.tolist(), density.tolist()


def log_weight_filter_grids(model: nn.Module, wandb_module: Any, *, prefix: str = "FilterGrids") -> dict[str, Any]:
    """Create WandB images for the first 25 filters in every conv/deconv layer."""
    payload: dict[str, Any] = {}

    for module_name, module in _conv_modules(model):
        filter_grid = _make_filter_grid(_conv_filter_bank(module))
        image = filter_grid.permute(1, 2, 0).numpy()
        layer_type = "deconv" if isinstance(module, nn.ConvTranspose2d) else "conv"
        safe_name = _safe_layer_name(module_name, module)
        payload[f"{prefix}/{safe_name}_{layer_type}"] = wandb_module.Image(image)

    return payload


def log_conv_weight_channel_stats(model: nn.Module) -> dict[str, torch.Tensor]:
    """Log per-filter max, min, mean, variance, and norm for every conv/deconv layer."""
    payload: dict[str, torch.Tensor] = {}

    for module_name, module in _conv_modules(model):
        values = _channel_values(_conv_filter_bank(module))
        safe_name = _safe_layer_name(module_name, module)
        payload[f"WeightMax/{safe_name}"] = values.amax(dim=1)
        payload[f"WeightMin/{safe_name}"] = values.amin(dim=1)
        payload[f"WeightMean/{safe_name}"] = values.mean(dim=1)
        payload[f"WeightVariance/{safe_name}"] = values.var(dim=1, unbiased=False)
        payload[f"WeightNorm/{safe_name}"] = values.norm(dim=1)

    return payload


def log_conv_gradient_channel_stats(model: nn.Module, *, prefix: str = "GradientMagnitude") -> dict[str, torch.Tensor]:
    """Log the per-filter gradient magnitude for every conv/deconv layer with gradients."""
    payload: dict[str, torch.Tensor] = {}

    for module_name, module in _conv_modules(model):
        if module.weight.grad is None:
            continue
        grad_values = _channel_values(module.weight.grad)
        safe_name = _safe_layer_name(module_name, module)
        payload[f"{prefix}/{safe_name}"] = grad_values.norm(dim=1)

    return payload


def log_conv_norm_kdes(model: nn.Module, wandb_module: Any, *, prefix: str = "WeightNormKDE") -> dict[str, Any]:
    """Log a simple KDE of filter norms for every conv/deconv layer."""
    payload: dict[str, Any] = {}

    for module_name, module in _conv_modules(model):
        norms = _channel_values(_conv_filter_bank(module)).norm(dim=1)
        xs, ys = _kde_curve(norms)
        if not xs:
            continue
        safe_name = _safe_layer_name(module_name, module)
        payload[f"{prefix}/{safe_name}"] = wandb_module.plot.line_series(
            xs=xs,
            ys=[ys],
            keys=["norm_kde"],
            title=f"{safe_name} filter norm KDE",
            xname="filter_norm",
        )

    return payload


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
    prefix: str = "FLOPsByLayer",
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

    payload["FLOPsTotal/value"] = int(sum(payload.values()))
    return payload
