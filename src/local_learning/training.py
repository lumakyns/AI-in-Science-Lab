from io import BytesIO
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from datasets import get_dataset
from losses import get_loss
from models import get_model
from loggers.names import wandb_safe_layer_name


CLASSIFICATION_ARCHITECTURES = {
    "densenet121",
    "resnet18",
    "vgg16",
    "greedy_stacked_autoencoder",
}
RECONSTRUCTION_ARCHITECTURES = {"wta_conv_ae"}
DEFAULT_PREPROCESSING = "none"
DEFAULT_MAX_EVAL_BATCHES = 25


def _parse_data_config(data: str) -> tuple[str, str]:
    """Split compact data config strings like cifar10:whiten."""
    for separator in (":", "/", "|"):
        if separator in data:
            dataset, preprocessing = data.split(separator, 1)
            return dataset, preprocessing
    return data, DEFAULT_PREPROCESSING


def _training_mode_for_architecture(architecture_type: str) -> str:
    """Infer the training target from the selected model architecture."""
    if architecture_type in CLASSIFICATION_ARCHITECTURES:
        return "classification"
    if architecture_type in RECONSTRUCTION_ARCHITECTURES:
        return "reconstruction"
    raise ValueError(f"Unknown architecture_type={architecture_type!r}.")


def normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    """Fill derived config fields used internally by training, models, and losses."""
    cfg = dict(config)
    if "data" in cfg:
        cfg["dataset"], cfg["preprocessing"] = _parse_data_config(str(cfg["data"]))
    elif "dataset" in cfg:
        cfg["preprocessing"] = str(cfg.get("preprocessing", DEFAULT_PREPROCESSING))
        cfg["data"] = f"{cfg['dataset']}:{cfg['preprocessing']}"
    else:
        raise KeyError("config must define data, e.g. 'cifar10:whiten'.")

    cfg["weights"] = str(cfg.get("weights", "random"))
    if cfg["weights"] not in {"default", "random"}:
        raise ValueError("weights must be 'default' or 'random'.")
    cfg["training_mode"] = _training_mode_for_architecture(str(cfg["architecture_type"]))
    return cfg


def get_loaders(cfg: dict[str, Any], device: torch.device) -> tuple[DataLoader, DataLoader]:
    """Build train and test DataLoaders from a config dictionary."""
    train_ds = get_dataset(
        train=True,
        dataset=cfg["dataset"],
        preprocessing=cfg["preprocessing"],
    )
    test_ds = get_dataset(
        train=False,
        dataset=cfg["dataset"],
        preprocessing=cfg["preprocessing"],
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["num_workers"]),
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["num_workers"]),
        pin_memory=(device.type == "cuda"),
    )
    return train_loader, test_loader


def get_config_name(cfg: dict[str, Any]) -> str:
    """Create a concise WandB run name from the main experiment settings."""
    architecture_mode = ""
    if cfg["architecture_type"] == "greedy_stacked_autoencoder":
        architecture_mode = f"-gsa_local_{bool(cfg.get('gsa_local_training', False))}"

    return (
        f"{cfg['training_mode']}"
        f"-{cfg['architecture_type']}"
        f"-{cfg['data']}"
        f"-weights_{cfg['weights']}"
        f"-{cfg['loss_type']}"
        f"{architecture_mode}"
        f"-lr_{cfg['learning_rate']}"
    )


def configure_wandb_metrics(run: Any) -> None:
    """Configure W&B without creating extra visible trainer metrics."""
    del run


def wandb_log(run: Any, payload: dict[str, object]) -> None:
    """Log one already-batched payload to W&B."""
    if payload:
        run.log(payload)


def _weight_mean_line_image(
    values: torch.Tensor,
    names: list[str],
    *,
    width: int = 1440,
    height: int = 720,
) -> Image.Image:
    dpi = 100
    plot_width = max(width, 72 * max(1, len(names)))
    plot_height = max(height, 840 if len(names) > 20 else height)
    fig, ax = plt.subplots(figsize=(plot_width / dpi, plot_height / dpi), dpi=dpi)
    x = np.arange(len(names))
    y = values.detach().cpu().numpy()

    ax.plot(x, y, marker="o", linewidth=2.0, markersize=5.0, color="#1452b8")
    ax.set_ylabel("Weight mean")
    ax.set_xlabel("Layer")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=75, ha="right", fontsize=7)
    ax.grid(axis="y", color="#d8dde5", linewidth=0.8)
    ax.margins(x=0.02)

    if values.numel() > 0:
        y_min = float(values.amin())
        y_max = float(values.amax())
        y_span = y_max - y_min
        y_offset = (y_span * 0.16) if y_span else 0.08
        if len(names) <= 40:
            label_offsets = (18, 34, 50, -18, -34, -50)
            for idx, (name, value) in enumerate(zip(names, y.tolist(), strict=True)):
                offset = label_offsets[idx % len(label_offsets)]
                ax.annotate(
                    name,
                    xy=(idx, value),
                    xytext=(0, offset),
                    textcoords="offset points",
                    ha="center",
                    va="bottom" if offset > 0 else "top",
                    fontsize=6,
                    rotation=25,
                    color="#23272f",
                    bbox={
                        "boxstyle": "round,pad=0.15",
                        "facecolor": "white",
                        "edgecolor": "#d8dde5",
                        "alpha": 0.85,
                    },
                )
        ax.set_ylim(y_min - y_offset, y_max + y_offset)

    fig.subplots_adjust(left=0.06, right=0.99, top=0.96, bottom=0.36)
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi)
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def _weight_mean_payload(model: torch.nn.Module, *, model_type: str, phase: str) -> dict[str, object]:
    if not hasattr(model, "get_weights"):
        return {}

    weight_modules = model.get_weights()
    if not weight_modules:
        return {}

    means = torch.tensor(
        [float(module.tensor.detach().to(torch.float32).mean().cpu()) for module in weight_modules],
        dtype=torch.float32,
    )
    names = [str(module.name) for module in weight_modules]
    image = _weight_mean_line_image(means, names)
    caption = " | ".join(
        f"{module.name}: {value:.4g}"
        for module, value in zip(weight_modules, means.tolist(), strict=True)
    )
    return {
        f"weight-mean/{model_type}": wandb.Image(
            image,
            caption=f"{phase}: {caption}",
        )
    }


def _loss_value(
    cfg: dict[str, Any],
    criterion,
    output,
    xb: torch.Tensor,
    yb: torch.Tensor,
    feature_maps: list,
) -> torch.Tensor:
    """Route model outputs to the loss signature implied by the training mode."""
    if cfg["training_mode"] == "classification":
        return criterion(y=yb, y_pred=output, fm_list=feature_maps)
    if isinstance(output, torch.Tensor) and output.shape != xb.shape:
        xb = xb.view_as(output)
    return criterion(reconstruction=output, target=xb, feature_maps=feature_maps)


def _metric_payload(cfg: dict[str, Any], criterion, output, yb: torch.Tensor) -> dict[str, float]:
    """Build scalar train metrics from the current model output and criterion state."""
    payload: dict[str, float] = {}
    if cfg["training_mode"] == "classification":
        acc = (output.argmax(dim=1) == yb).to(torch.float32).mean()
        ce_loss = getattr(criterion, "last_ce_loss", torch.tensor(0.0))
        payload.update(
            {
                "losses/ce": float(ce_loss.detach().cpu()),
                "losses/final_layer_ce": float(ce_loss.detach().cpu()),
                "losses/mse": float(
                    getattr(criterion, "last_mse_loss", torch.tensor(0.0)).detach().cpu()
                ),
                "train/acc": float(acc.detach().cpu()),
            }
        )
    else:
        payload["train/mse"] = float(
            getattr(criterion, "last_mse_loss", torch.tensor(0.0)).detach().cpu()
        )

    for layer_name, layer_mse in getattr(criterion, "last_mse_by_layer", {}).items():
        safe_name = wandb_safe_layer_name(str(layer_name))
        layer_mse_value = float(layer_mse.detach().cpu())
        payload[f"losses/autoencoder_mse/{safe_name}"] = layer_mse_value
        payload[f"losses/reconstruction_mse/{safe_name}"] = layer_mse_value

    return payload


def _log_loss_parts(criterion, payload: dict[str, object]) -> None:
    """Append optional criterion sub-losses to an existing WandB payload."""
    payload["losses/correlation_total"] = float(
        getattr(criterion, "last_corr_total", torch.tensor(0.0)).detach().cpu()
    )
    for layer_name, layer_loss in getattr(criterion, "last_corr_by_layer", {}).items():
        safe_name = wandb_safe_layer_name(str(layer_name))
        payload[f"losses/{safe_name}"] = float(layer_loss.detach().cpu())


def _forward_model(
    cfg: dict[str, Any],
    model: torch.nn.Module,
    xb: torch.Tensor,
    *,
    epoch: int,
    inputs_processed_in_epoch: int,
):
    """Call models with annealing kwargs only when reconstruction-style forwards need them."""
    if cfg["architecture_type"] in {"wta_conv_ae", "greedy_stacked_autoencoder"}:
        return model(
            xb,
            epoch=epoch,
            inputs_processed_in_epoch=inputs_processed_in_epoch,
        )
    return model(xb)


def train(config: dict[str, Any], *, device: torch.device | None = None) -> dict[str, float]:
    """Train one configured experiment and return final evaluation metrics."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = wandb.init(project=config["project"], config=config)
    # Normalize config once so later code can assume explicit fields.
    cfg = normalize_config(dict(wandb.config))
    run.name = get_config_name(cfg)
    configure_wandb_metrics(run)

    # Build data first so dataset_size is available to sparse models.
    train_loader, test_loader = get_loaders(cfg, device)
    cfg["dataset_size"] = len(train_loader.dataset)

    # Model, loss, and optimizer all come from the normalized config.
    model = get_model(cfg).to(device)
    frozen = bool(cfg.get("frozen", False))
    if frozen:
        for param in model.parameters():
            param.requires_grad = False

    criterion = get_loss(cfg).to(device)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = (
        torch.optim.Adam(trainable_params, lr=float(cfg["learning_rate"]))
        if trainable_params
        else None
    )
    wandb_log(run, _weight_mean_payload(model, model_type=str(cfg["architecture_type"]), phase="initial"))

    def eval_model(epoch: int) -> dict[str, float]:
        # Run a bounded validation pass to keep logging cost predictable.
        model.eval()
        loss_sum = 0.0
        acc_sum = 0.0
        mse_sum = 0.0
        n = 0
        with torch.no_grad():
            for batch_idx, (xb, yb) in enumerate(test_loader):
                if batch_idx >= DEFAULT_MAX_EVAL_BATCHES:
                    break
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                output, feature_maps = _forward_model(
                    cfg,
                    model,
                    xb,
                    epoch=epoch,
                    inputs_processed_in_epoch=0,
                )
                loss = _loss_value(cfg, criterion, output, xb, yb, feature_maps)
                batch_size = int(xb.shape[0])
                loss_sum += float(loss.detach().cpu()) * batch_size
                if cfg["training_mode"] == "classification":
                    acc = (output.argmax(dim=1) == yb).to(torch.float32).mean()
                    acc_sum += float(acc.detach().cpu()) * batch_size
                else:
                    mse_sum += float(criterion.last_mse_loss.detach().cpu()) * batch_size
                n += batch_size
        model.train()
        metrics = {"test/loss": loss_sum / max(1, n)}
        if cfg["training_mode"] == "classification":
            metrics["test/acc"] = acc_sum / max(1, n)
        else:
            metrics["test/mse"] = mse_sum / max(1, n)
        return metrics

    global_step = 0
    log_every_steps = int(cfg["log_every_steps"])
    eval_interval_steps = max(1, len(train_loader) // 4)

    num_epochs = int(cfg["epochs"])
    total_train_examples = len(train_loader.dataset)
    for epoch in range(num_epochs):
        if hasattr(model, "clear_stats"):
            model.clear_stats()
        if frozen:
            model.eval()
        else:
            model.train()
        inputs_processed_in_epoch = 0

        progress = tqdm(
            total=total_train_examples,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            unit="example",
        )
        for step_in_epoch, (xb, yb) in enumerate(train_loader):
            batch_size = int(xb.shape[0])
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            # Some models need epoch/sample counts for sparsity annealing.
            with torch.set_grad_enabled(not frozen):
                output, feature_maps = _forward_model(
                    cfg,
                    model,
                    xb,
                    epoch=epoch,
                    inputs_processed_in_epoch=inputs_processed_in_epoch,
                )

                # Route outputs through the correct classification/reconstruction loss.
                loss = _loss_value(cfg, criterion, output, xb, yb, feature_maps)
                if optimizer is not None:
                    loss.backward()

            log_payload: dict[str, object] | None = None
            if global_step % log_every_steps == 0:
                # Batch scalar metrics and optional diagnostics into one WandB log.
                log_payload = {
                    "train/loss": float(loss.detach().cpu()),
                }
                log_payload.update(_metric_payload(cfg, criterion, output, yb))
                if hasattr(model, "last_k"):
                    log_payload["general/last_k"] = int(model.last_k)
                _log_loss_parts(criterion, log_payload)

            if optimizer is not None:
                optimizer.step()

            if log_payload is not None:
                wandb_log(run, log_payload)

            inputs_processed_in_epoch += batch_size
            progress.update(batch_size)
            progress.set_postfix(
                examples=f"{inputs_processed_in_epoch}/{total_train_examples}",
                device=device.type,
            )
            global_step += 1

            if (step_in_epoch + 1) % eval_interval_steps == 0:
                metrics = eval_model(epoch)
                wandb_log(run, metrics)
                if frozen:
                    model.eval()
                else:
                    model.train()
        progress.close()
        wandb_log(
            run,
            _weight_mean_payload(
                model,
                model_type=str(cfg["architecture_type"]),
                phase=f"epoch_{epoch}",
            ),
        )
    metrics = eval_model(int(cfg["epochs"]) - 1)
    run.finish()
    return metrics
