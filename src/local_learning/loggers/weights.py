from typing import Any

import torch
import torch.nn as nn

from .names import wandb_safe_layer_name

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
    return wandb_safe_layer_name(module_name or layer_type)


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


def _plot_lines_to_image(
    xs: torch.Tensor,
    ys: list[list[float]],
    *,
    width: int = 480,
    height: int = 320,
    padding: int = 36,
) -> torch.Tensor:
    image = torch.ones(3, height, width, dtype=torch.float32)
    if xs.numel() == 0 or not ys:
        return image

    y_tensors = [torch.tensor(y, dtype=torch.float32) for y in ys if y]
    if not y_tensors:
        return image

    x_min = float(xs.amin())
    x_max = float(xs.amax())
    y_min = 0.0
    y_max = float(torch.cat(y_tensors).amax())
    if x_max == x_min or y_max <= y_min:
        return image

    axis_color = torch.tensor([0.72, 0.74, 0.76], dtype=image.dtype)
    image[:, height - padding, padding : width - padding] = axis_color[:, None]
    image[:, padding : height - padding, padding] = axis_color[:, None]

    colors = (
        torch.tensor([0.08, 0.32, 0.72]),
        torch.tensor([0.82, 0.20, 0.18]),
        torch.tensor([0.20, 0.58, 0.26]),
        torch.tensor([0.55, 0.32, 0.72]),
        torch.tensor([0.90, 0.55, 0.12]),
        torch.tensor([0.16, 0.60, 0.66]),
        torch.tensor([0.55, 0.55, 0.55]),
        torch.tensor([0.20, 0.20, 0.20]),
    )
    plot_w = max(1, width - (2 * padding) - 1)
    plot_h = max(1, height - (2 * padding) - 1)

    for line_idx, y_values in enumerate(y_tensors):
        color = colors[line_idx % len(colors)].to(image.dtype)
        x_pixels = padding + ((xs - x_min) / (x_max - x_min) * plot_w).round().to(torch.long)
        y_pixels = height - padding - ((y_values - y_min) / (y_max - y_min) * plot_h).round().to(torch.long)
        points = zip(x_pixels.tolist(), y_pixels.tolist())
        previous: tuple[int, int] | None = None
        for x_pixel, y_pixel in points:
            x_pixel = min(max(0, x_pixel), width - 1)
            y_pixel = min(max(0, y_pixel), height - 1)
            if previous is None:
                image[:, y_pixel, x_pixel] = color
            else:
                _draw_line(image, previous, (x_pixel, y_pixel), color)
            previous = (x_pixel, y_pixel)

    return image


def _draw_line(
    image: torch.Tensor,
    start: tuple[int, int],
    end: tuple[int, int],
    color: torch.Tensor,
) -> None:
    x0, y0 = start
    x1, y1 = end
    steps = max(abs(x1 - x0), abs(y1 - y0), 1)
    for step in range(steps + 1):
        t = step / steps
        x = round(x0 + ((x1 - x0) * t))
        y = round(y0 + ((y1 - y0) * t))
        y_slice = slice(max(0, y - 1), min(image.shape[1], y + 2))
        x_slice = slice(max(0, x - 1), min(image.shape[2], x + 2))
        image[:, y_slice, x_slice] = color[:, None, None]


def log_conv_weight_snapshot(
    model: nn.Module,
    wandb_module: Any,
    norm_kde_history_logger: "ConvNormKDEHistoryLogger",
    *,
    step: int,
    filter_grid_prefix: str = "test-filter-grid",
    distribution_prefix: str = "train-encoder-weight-distribution",
    gradient_prefix: str = "train-gradient-distribution",
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
        prefix: str = "train-filter-norm-kde-progress",
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

        image = _plot_lines_to_image(xs, ys)
        payload[f"{self.prefix}/{safe_name}"] = self.wandb.Image(
            image.permute(1, 2, 0).numpy(),
            caption=(
                f"{safe_name} filter norm KDE over training; "
                f"lines are steps {[snapshot_step for snapshot_step, _ in layer_history]}"
            ),
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
    prefix: str = "test-flops",
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
            safe_name = wandb_safe_layer_name(module_name or module.__class__.__name__.lower())
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
