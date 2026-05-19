from typing import Any

import torch
import wandb
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from datasets import get_dataset
from losses import get_loss
from models import get_model
from wandb_logging import (
    ChannelActivationScatterLogger,
    log_conv_gradient_channel_stats,
    log_conv_norm_kdes,
    log_conv_weight_channel_stats,
    log_inference_flops,
    log_weight_filter_grids,
)


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
    return (
        f"{cfg['training_mode']}"
        f"-{cfg['architecture_type']}"
        f"-{cfg['dataset']}"
        f"-{cfg['preprocessing']}"
        f"-{cfg['loss_type']}"
        f"-local_{cfg.get('local_training', False)}"
        f"-lr_{cfg['learning_rate']}"
    )


def _capitalize_top_folder(key: str) -> str:
    """Capitalize the top WandB namespace while leaving nested names readable."""
    if "/" not in key:
        return key
    top, rest = key.split("/", 1)
    if not top:
        return key
    return f"{top[:1].upper()}{top[1:]}" + f"/{rest}"


def wandb_log(payload: dict[str, object], *, step: int | None = None) -> None:
    """Log a payload to WandB after normalizing top-level folder names."""
    payload = {_capitalize_top_folder(key): value for key, value in payload.items()}
    if step is None:
        wandb.log(payload)
        return
    wandb.log(payload, step=step)


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
    if cfg["training_mode"] == "classification":
        acc = (output.argmax(dim=1) == yb).to(torch.float32).mean()
        return {
            "Losses/ce": float(getattr(criterion, "last_ce_loss", torch.tensor(0.0)).detach().cpu()),
            "Losses/mse": float(getattr(criterion, "last_mse_loss", torch.tensor(0.0)).detach().cpu()),
            "General/acc": float(acc.detach().cpu()),
        }
    return {
        "Train/mse": float(getattr(criterion, "last_mse_loss", torch.tensor(0.0)).detach().cpu()),
    }


def _log_loss_parts(criterion, payload: dict[str, float | int]) -> None:
    """Append optional criterion sub-losses to an existing WandB payload."""
    payload["Losses/correlation_total"] = float(
        getattr(criterion, "last_corr_total", torch.tensor(0.0)).detach().cpu()
    )
    for layer_name, layer_loss in getattr(criterion, "last_corr_by_layer", {}).items():
        payload[f"Losses/{layer_name}"] = float(layer_loss.detach().cpu())


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
    cfg = dict(wandb.config)
    run.name = get_config_name(cfg)

    train_loader, test_loader = get_loaders(cfg, device)
    cfg["dataset_size"] = len(train_loader.dataset)

    model = get_model(cfg).to(device)
    criterion = get_loss(cfg).to(device)
    optimizer = torch.optim.Adam(
        [param for param in model.parameters() if param.requires_grad],
        lr=float(cfg["learning_rate"]),
    )

    def eval_model(epoch: int) -> dict[str, float]:
        model.eval()
        loss_sum = 0.0
        acc_sum = 0.0
        mse_sum = 0.0
        n = 0
        with torch.no_grad():
            for batch_idx, (xb, yb) in enumerate(test_loader):
                if batch_idx >= int(cfg["max_eval_batches"]):
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
        metrics = {"Test/loss": loss_sum / max(1, n)}
        if cfg["training_mode"] == "classification":
            metrics["Test/acc"] = acc_sum / max(1, n)
        else:
            metrics["Test/mse"] = mse_sum / max(1, n)
        return metrics

    global_step = 0
    log_every_steps = int(cfg["log_every_steps"])
    weight_log_every_steps = int(cfg.get("log_weight_grids_every_steps", log_every_steps))
    flop_log_every_steps = int(cfg.get("log_flops_every_steps", 0))
    eval_interval_steps = max(1, len(train_loader) // 4)
    activation_scatter_logger = ChannelActivationScatterLogger(wandb)

    for epoch in tqdm(range(int(cfg["epochs"])), desc="Epochs"):
        model.train()
        inputs_processed_in_epoch = 0
        activation_scatter_logger.reset()

        for step_in_epoch, (xb, yb) in enumerate(train_loader):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            output, feature_maps = _forward_model(
                cfg,
                model,
                xb,
                epoch=epoch,
                inputs_processed_in_epoch=inputs_processed_in_epoch,
            )

            loss = _loss_value(cfg, criterion, output, xb, yb, feature_maps)
            activation_scatter_logger.update(feature_maps)
            loss.backward()

            if global_step % log_every_steps == 0:
                log_payload: dict[str, float | int] = {
                    "Train/loss": float(loss.detach().cpu()),
                    "General/epoch": epoch,
                    "General/step": global_step,
                }
                log_payload.update(_metric_payload(cfg, criterion, output, yb))
                if hasattr(model, "last_k"):
                    log_payload["General/last_k"] = int(model.last_k)
                _log_loss_parts(criterion, log_payload)
                wandb_log(log_payload, step=global_step)

                if weight_log_every_steps > 0 and global_step % weight_log_every_steps == 0:
                    conv_payload = {}
                    conv_payload.update(log_weight_filter_grids(model, wandb))
                    conv_payload.update(log_conv_weight_channel_stats(model))
                    conv_payload.update(log_conv_gradient_channel_stats(model))
                    conv_payload.update(log_conv_norm_kdes(model, wandb))
                    wandb_log(conv_payload, step=global_step)

                if flop_log_every_steps > 0 and global_step % flop_log_every_steps == 0:
                    forward_kwargs = {}
                    if cfg["architecture_type"] in {"wta_conv_ae", "greedy_stacked_autoencoder"}:
                        forward_kwargs = {
                            "epoch": epoch,
                            "inputs_processed_in_epoch": inputs_processed_in_epoch,
                        }
                    wandb_log(
                        log_inference_flops(model, xb[:1], forward_kwargs=forward_kwargs),
                        step=global_step,
                    )

            optimizer.step()

            batch_size = int(xb.shape[0])
            inputs_processed_in_epoch += batch_size
            global_step += 1

            if (step_in_epoch + 1) % eval_interval_steps == 0:
                metrics = eval_model(epoch)
                metrics.update({"General/epoch": epoch, "General/step": global_step})
                wandb_log(metrics, step=global_step)

        activation_scatter_payload = activation_scatter_logger.log_epoch(epoch=epoch)
        if activation_scatter_payload:
            wandb_log(activation_scatter_payload, step=global_step)

    metrics = eval_model(int(cfg["epochs"]) - 1)
    wandb.finish()
    return metrics
