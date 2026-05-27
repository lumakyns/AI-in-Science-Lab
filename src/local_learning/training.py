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
    ChannelActivationStatsLogger,
    ConvNormKDEHistoryLogger,
    ConvWeightChangeLogger,
    log_conv_weight_snapshot,
    log_inference_flops,
)


CLASSIFICATION_ARCHITECTURES = {
    "cnn1",
    "cnn2",
    "resnet18",
    "pretrained_resnet18",
    "greedy_stacked_autoencoder",
}
RECONSTRUCTION_ARCHITECTURES = {"wta_conv_ae"}
DEFAULT_PREPROCESSING = "none"
DEFAULT_WEIGHT_LOG_EVERY_STEPS = 100
DEFAULT_FLOP_LOG_EVERY_STEPS = 100
DEFAULT_MAX_EVAL_BATCHES = 25
DEFAULT_ACTIVATION_VIZ_BATCHES = 1


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
    return (
        f"{cfg['training_mode']}"
        f"-{cfg['architecture_type']}"
        f"-{cfg['data']}"
        f"-{cfg['loss_type']}"
        f"-local_{cfg.get('local_training', False)}"
        f"-lr_{cfg['learning_rate']}"
    )


def configure_wandb_metrics(run: Any) -> None:
    """Configure W&B without creating extra visible trainer metrics."""
    del run


def wandb_log(run: Any, payload: dict[str, object]) -> None:
    """Log one already-batched payload to W&B."""
    if payload:
        run.log(payload)


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
        safe_name = str(layer_name).replace(".", "__").replace("/", "__")
        payload[f"losses/autoencoder_mse/{safe_name}"] = float(layer_mse.detach().cpu())

    return payload


def _log_loss_parts(criterion, payload: dict[str, object]) -> None:
    """Append optional criterion sub-losses to an existing WandB payload."""
    payload["losses/correlation_total"] = float(
        getattr(criterion, "last_corr_total", torch.tensor(0.0)).detach().cpu()
    )
    for layer_name, layer_loss in getattr(criterion, "last_corr_by_layer", {}).items():
        safe_name = str(layer_name).replace(".", "__").replace("/", "__")
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
    cfg = normalize_config(dict(wandb.config))
    run.name = get_config_name(cfg)
    configure_wandb_metrics(run)

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
    weight_log_every_steps = DEFAULT_WEIGHT_LOG_EVERY_STEPS
    flop_log_every_steps = DEFAULT_FLOP_LOG_EVERY_STEPS
    activation_viz_batches = DEFAULT_ACTIVATION_VIZ_BATCHES
    eval_interval_steps = max(1, len(train_loader) // 4)
    activation_scatter_logger = ChannelActivationScatterLogger(wandb)
    activation_stats_logger = ChannelActivationStatsLogger(
        zero_threshold=float(cfg.get("activation_zero_threshold", 1e-6)),
        dead_active_fraction=float(cfg.get("activation_dead_active_fraction", 0.01)),
        dominant_share_multiplier=float(cfg.get("activation_dominant_share_multiplier", 2.0)),
    )
    weight_change_logger = ConvWeightChangeLogger(model)
    norm_kde_history_logger = ConvNormKDEHistoryLogger(wandb)

    def log_activation_viz_snapshot(epoch: int | str) -> dict[str, Any]:
        if activation_viz_batches == 0:
            return {}

        was_training = model.training
        activation_scatter_logger.reset()
        activation_stats_logger.reset()
        model.eval()
        with torch.no_grad():
            for batch_idx, (xb, _yb) in enumerate(test_loader):
                if batch_idx >= activation_viz_batches:
                    break
                xb = xb.to(device, non_blocking=True)
                _output, feature_maps = _forward_model(
                    cfg,
                    model,
                    xb,
                    epoch=0 if isinstance(epoch, str) else epoch,
                    inputs_processed_in_epoch=0,
                )
                activation_scatter_logger.update(feature_maps)
                activation_stats_logger.update(feature_maps)

        payload = {}
        payload.update(activation_scatter_logger.log_epoch(epoch=epoch))
        payload.update(activation_stats_logger.log_epoch())
        if was_training:
            model.train()
        return payload

    wandb_log(run, log_activation_viz_snapshot(epoch="initial"))

    for epoch in tqdm(range(int(cfg["epochs"])), desc="Epochs"):
        model.train()
        inputs_processed_in_epoch = 0

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
            loss.backward()

            log_payload: dict[str, object] | None = None
            if global_step % log_every_steps == 0:
                log_payload = {
                    "train/loss": float(loss.detach().cpu()),
                }
                log_payload.update(_metric_payload(cfg, criterion, output, yb))
                if hasattr(model, "last_k"):
                    log_payload["general/last_k"] = int(model.last_k)
                _log_loss_parts(criterion, log_payload)

                if weight_log_every_steps > 0 and global_step % weight_log_every_steps == 0:
                    log_payload.update(
                        log_conv_weight_snapshot(
                            model,
                            wandb,
                            norm_kde_history_logger,
                            step=global_step,
                        )
                    )

                if flop_log_every_steps > 0 and global_step % flop_log_every_steps == 0:
                    forward_kwargs = {}
                    if cfg["architecture_type"] in {"wta_conv_ae", "greedy_stacked_autoencoder"}:
                        forward_kwargs = {
                            "epoch": epoch,
                            "inputs_processed_in_epoch": inputs_processed_in_epoch,
                        }
                    log_payload.update(
                        log_inference_flops(model, xb, forward_kwargs=forward_kwargs)
                    )

            optimizer.step()

            if weight_log_every_steps > 0 and global_step % weight_log_every_steps == 0:
                weight_change_payload = weight_change_logger.log(model)
                if weight_change_payload:
                    if log_payload is None:
                        log_payload = {}
                    log_payload.update(weight_change_payload)

            if log_payload is not None:
                wandb_log(run, log_payload)

            batch_size = int(xb.shape[0])
            inputs_processed_in_epoch += batch_size
            global_step += 1

            if (step_in_epoch + 1) % eval_interval_steps == 0:
                metrics = eval_model(epoch)
                wandb_log(run, metrics)

        wandb_log(run, log_activation_viz_snapshot(epoch=epoch))

    metrics = eval_model(int(cfg["epochs"]) - 1)
    run.finish()
    return metrics
