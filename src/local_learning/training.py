from typing import Any

import torch
import wandb
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from datasets import get_dataset
from losses import get_loss
from models import get_model
from loggers import (
    ChannelActivationScatterLogger,
    ChannelActivationStatsLogger,
    ConvNormKDEHistoryLogger,
    FeatureMapDistributionLogger,
    FirstLayerReconstructionImageLogger,
    log_conv_weight_snapshot,
    log_inference_flops,
)


CLASSIFICATION_ARCHITECTURES = {
    "cnn1",
    "cnn2",
    "resnet18",
    "pretrained_resnet18",
    "vgg16",
    "pretrained_vgg16",
    "greedy_stacked_autoencoder",
}
RECONSTRUCTION_ARCHITECTURES = {"wta_conv_ae"}
DEFAULT_PREPROCESSING = "none"
DEFAULT_WEIGHT_LOG_EVERY_STEPS = 100
DEFAULT_FLOP_LOG_EVERY_STEPS = 100
DEFAULT_MAX_EVAL_BATCHES = 25
DEFAULT_ACTIVATION_VIZ_BATCHES = 1
NORMALIZATION_STATS = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "smallcifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    "imagenet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}


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


def _normalization_stats_for_config(
    cfg: dict[str, Any],
) -> tuple[tuple[float, ...], tuple[float, ...]] | tuple[None, None]:
    if cfg.get("preprocessing") != "normalize":
        return None, None
    dataset = str(cfg["dataset"]).removesuffix("_patches")
    return NORMALIZATION_STATS.get(dataset, (None, None))


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
    architecture_mode = ""
    if cfg["architecture_type"] in {"vgg16", "pretrained_vgg16"}:
        first_full_epoch = cfg.get("vgg16_first_full_training_epoch")
        vgg16_mode = (
            "manual"
            if first_full_epoch is None
            else f"full_epoch_{first_full_epoch}"
        )
        architecture_mode = f"-vgg16_{vgg16_mode}"
    elif cfg["architecture_type"] == "greedy_stacked_autoencoder":
        architecture_mode = f"-gsa_local_{bool(cfg.get('gsa_local_training', False))}"

    return (
        f"{cfg['training_mode']}"
        f"-{cfg['architecture_type']}"
        f"-{cfg['data']}"
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
        safe_name = str(layer_name).replace(".", "__").replace("/", "__")
        payload[f"losses/{safe_name}"] = float(layer_loss.detach().cpu())


def _vgg16_first_full_training_epoch(cfg: dict[str, Any]) -> int | None:
    first_full_epoch = cfg.get("vgg16_first_full_training_epoch")
    return None if first_full_epoch is None else int(first_full_epoch)


def _vgg16_pretrain_mode(cfg: dict[str, Any], *, epoch: int) -> bool:
    if cfg["architecture_type"] not in {"vgg16", "pretrained_vgg16"}:
        return False
    first_full_epoch = _vgg16_first_full_training_epoch(cfg)
    return first_full_epoch is not None and epoch < first_full_epoch


def _vgg16_forward_flags(cfg: dict[str, Any], *, epoch: int) -> tuple[bool, bool]:
    first_full_epoch = _vgg16_first_full_training_epoch(cfg)
    if first_full_epoch is None:
        return (
            bool(cfg.get("vgg16_deconv_training", False)),
            bool(cfg.get("vgg16_local_training", False)),
        )

    pretrain_mode = _vgg16_pretrain_mode(cfg, epoch=epoch)
    return pretrain_mode, pretrain_mode


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
    if cfg["architecture_type"] in {"vgg16", "pretrained_vgg16"}:
        deconv_training, local_training = _vgg16_forward_flags(cfg, epoch=epoch)
        return model(
            xb,
            deconv_training=deconv_training,
            local_training=local_training,
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
    criterion = get_loss(cfg).to(device)
    optimizer = torch.optim.Adam(
        [param for param in model.parameters() if param.requires_grad],
        lr=float(cfg["learning_rate"]),
    )

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
    weight_log_every_steps = DEFAULT_WEIGHT_LOG_EVERY_STEPS
    flop_log_every_steps = DEFAULT_FLOP_LOG_EVERY_STEPS
    activation_viz_batches = DEFAULT_ACTIVATION_VIZ_BATCHES
    eval_interval_steps = max(1, len(train_loader) // 4)
    # Stateful loggers accumulate snapshots between WandB writes.
    activation_scatter_logger = ChannelActivationScatterLogger(wandb)
    activation_stats_logger = ChannelActivationStatsLogger()
    norm_kde_history_logger = ConvNormKDEHistoryLogger(wandb)
    feature_map_distribution_logger = FeatureMapDistributionLogger(wandb)
    reconstruction_mean, reconstruction_std = _normalization_stats_for_config(cfg)
    first_layer_reconstruction_logger = FirstLayerReconstructionImageLogger(
        wandb,
        mean=reconstruction_mean,
        std=reconstruction_std,
    )

    def log_activation_viz_snapshot(epoch: int | str) -> dict[str, Any]:
        # Capture feature-map summaries without affecting training mode.
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

            # Some models need epoch/sample counts for sparsity annealing.
            output, feature_maps = _forward_model(
                cfg,
                model,
                xb,
                epoch=epoch,
                inputs_processed_in_epoch=inputs_processed_in_epoch,
            )

            # Route outputs through the correct classification/reconstruction loss.
            loss = _loss_value(cfg, criterion, output, xb, yb, feature_maps)
            loss.backward()

            log_payload: dict[str, object] | None = None
            if global_step % log_every_steps == 0:
                # Batch scalar metrics and optional diagnostics into one WandB log.
                log_payload = {
                    "train/loss": float(loss.detach().cpu()),
                }
                log_payload.update(_metric_payload(cfg, criterion, output, yb))
                log_payload.update(
                    feature_map_distribution_logger.log_step(feature_maps)
                )
                log_payload.update(
                    first_layer_reconstruction_logger.log_step(
                        feature_maps,
                        step=global_step,
                        model=model,
                    )
                )
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
                    elif cfg["architecture_type"] in {"vgg16", "pretrained_vgg16"}:
                        deconv_training, local_training = _vgg16_forward_flags(cfg, epoch=epoch)
                        forward_kwargs = {
                            "deconv_training": deconv_training,
                            "local_training": local_training,
                        }
                    log_payload.update(
                        log_inference_flops(model, xb, forward_kwargs=forward_kwargs)
                    )

            optimizer.step()

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
