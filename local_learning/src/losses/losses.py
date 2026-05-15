from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class CorrelationRedundancy:
    @staticmethod
    def reduce_corr_matrix(corr_mat: torch.Tensor, correlation_loss: str) -> torch.Tensor:
        match correlation_loss:
            case "sum":
                return corr_mat.sum()
            case "max":
                return corr_mat.amax()
            case _:
                raise ValueError("correlation_loss must be 'sum' or 'max'.")

    @staticmethod
    def parse_feature_map(
        feature_map_item,
        *,
        idx: int,
    ) -> tuple[str, torch.Tensor, nn.Conv2d | None, torch.Tensor | None]:
        if isinstance(feature_map_item, torch.Tensor):
            return f"fm_{idx}", feature_map_item, None, None

        if hasattr(feature_map_item, "name") and hasattr(feature_map_item, "feature_map"):
            return (
                feature_map_item.name,
                feature_map_item.feature_map,
                getattr(feature_map_item, "conv", None),
                getattr(feature_map_item, "conv_input", None),
            )

        if isinstance(feature_map_item, (tuple, list)):
            if len(feature_map_item) >= 4 and isinstance(feature_map_item[0], str):
                return feature_map_item[0], feature_map_item[1], feature_map_item[2], feature_map_item[3]
            if len(feature_map_item) >= 2 and isinstance(feature_map_item[0], str):
                return feature_map_item[0], feature_map_item[1], None, None

        raise TypeError(
            "feature maps must be tensors, FeatureMapEntry items, or tuples like (name, feature_map)."
        )

    @staticmethod
    def luca_fn(
        feature_map: torch.Tensor,
        kernel_radius: int,
        correlation_mode: str,
        comparison_mode: str,
        normalization_mode: str,
        postcomp_mode: str,
    ) -> torch.Tensor:
        if correlation_mode not in {"mean", "max"}:
            raise ValueError("correlation_mode must be 'mean' or 'max'.")
        if comparison_mode not in {"shift", "batch", "both"}:
            raise ValueError("comparison_mode must be 'shift', 'batch', or 'both'.")
        if normalization_mode not in {"relu", "relu_log", "chnorm_relu"}:
            raise ValueError("normalization_mode must be 'relu', 'relu_log', or 'chnorm_relu'.")
        if postcomp_mode not in {"squared", "thresh"}:
            raise ValueError("postcomp_mode must be 'squared' or 'thresh'.")

        eps = 1e-6
        batch_size, channels, _, _ = feature_map.shape
        fmap = feature_map
        radius = kernel_radius

        if normalization_mode == "relu":
            fmap = F.relu(fmap)
        elif normalization_mode == "relu_log":
            fmap = torch.log1p(F.relu(fmap))
        elif normalization_mode == "chnorm_relu":
            ch_mean = fmap.mean(dim=(0, 2, 3), keepdim=True)
            ch_std = fmap.std(dim=(0, 2, 3), keepdim=True, unbiased=False)
            fmap = F.relu((fmap - ch_mean) / (ch_std + eps))

        patch_hw = (2 * radius) + 1
        patch_len = patch_hw * patch_hw
        patches = F.unfold(fmap, kernel_size=patch_hw, padding=radius)
        patches = patches.view(batch_size, channels, patch_len, -1)
        center = patches[:, :, patch_len // 2, :]

        if comparison_mode == "shift":
            corr = torch.einsum("bip,bjkp->ijbp", center, patches) / patch_len
        elif comparison_mode == "batch":
            corr = torch.einsum("bip,bjkp->ijkp", center, patches) / batch_size
        else:
            corr = torch.einsum("bip,bjkp->ijp", center, patches) / (patch_len * batch_size)

        if postcomp_mode == "squared":
            corr = corr * corr
        elif postcomp_mode == "thresh":
            corr = F.relu(corr)

        if correlation_mode == "max":
            corr = corr.amax(dim=tuple(range(2, corr.ndim)))
        elif correlation_mode == "mean":
            corr = corr.mean(dim=tuple(range(2, corr.ndim)))

        mask = torch.triu(
            torch.ones((channels, channels), dtype=torch.bool, device=feature_map.device),
            diagonal=1,
        )
        return corr * mask


class ClassificationLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.last_ce_loss = torch.tensor(0.0)
        self.last_corr_total = torch.tensor(0.0)
        self.last_corr_by_layer: dict[str, torch.Tensor] = {}

    def forward(self, y: torch.Tensor, y_pred: torch.Tensor, fm_list: list | None = None) -> torch.Tensor:
        del fm_list
        ce_loss = self.ce(y_pred, y)
        self.last_ce_loss = ce_loss.detach()
        self.last_corr_total = y_pred.new_zeros(()).detach()
        self.last_corr_by_layer = {}
        return ce_loss


class RedundancyLoss(CorrelationRedundancy, nn.Module):
    def __init__(
        self,
        lambda_strength: float,
        *,
        kernel_size: int = 3,
        correlation_mode: str = "max",
        comparison_mode: str = "shift",
        normalization_mode: str = "relu",
        postcomp_mode: str = "squared",
        correlation_loss: str = "sum",
        local: bool = False,
    ) -> None:
        super().__init__()

        kernel_size = int(kernel_size)
        if kernel_size <= 0 or (kernel_size % 2) == 0:
            raise ValueError(f"kernel_size must be an odd positive int, got {kernel_size}")

        self.lambda_strength = float(lambda_strength)
        self.kernel_radius = (kernel_size - 1) // 2
        self.correlation_mode = correlation_mode
        self.comparison_mode = comparison_mode
        self.normalization_mode = normalization_mode
        self.postcomp_mode = postcomp_mode
        self.correlation_loss = correlation_loss
        self.local = bool(local)
        self.ce = nn.CrossEntropyLoss()

        self.last_ce_loss = torch.tensor(0.0)
        self.last_corr_total = torch.tensor(0.0)
        self.last_corr_by_layer: dict[str, torch.Tensor] = {}

    def forward(self, y: torch.Tensor, y_pred: torch.Tensor, fm_list: list | None = None) -> torch.Tensor:
        ce_loss = self.ce(y_pred, y)
        corr_total = y_pred.new_zeros(())
        corr_by_layer: dict[str, torch.Tensor] = {}

        for idx, fm_item in enumerate(fm_list or []):
            layer_name, feature_map, conv, conv_input = self.parse_feature_map(fm_item, idx=idx)
            if self.local:
                if conv is None or conv_input is None:
                    raise ValueError("local=True requires FeatureMapEntry items with conv and conv_input.")
                feature_map = conv(conv_input.detach())

            corr_mat = self.luca_fn(
                feature_map=feature_map,
                kernel_radius=self.kernel_radius,
                correlation_mode=self.correlation_mode,
                comparison_mode=self.comparison_mode,
                normalization_mode=self.normalization_mode,
                postcomp_mode=self.postcomp_mode,
            )
            corr_val = self.reduce_corr_matrix(corr_mat, self.correlation_loss)
            corr_total = corr_total + corr_val
            corr_by_layer[layer_name] = corr_val.detach()

        self.last_ce_loss = ce_loss.detach()
        self.last_corr_total = corr_total.detach()
        self.last_corr_by_layer = corr_by_layer
        return ce_loss + (self.lambda_strength * corr_total)


class ReconstructionLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mse = nn.MSELoss()
        self.last_mse_loss = torch.tensor(0.0)
        self.last_corr_total = torch.tensor(0.0)
        self.last_corr_by_layer: dict[str, torch.Tensor] = {}

    def forward(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
        feature_maps: list | None = None,
    ) -> torch.Tensor:
        del feature_maps
        mse_loss = self.mse(reconstruction, target)
        self.last_mse_loss = mse_loss.detach()
        self.last_corr_total = reconstruction.new_zeros(()).detach()
        self.last_corr_by_layer = {}
        return mse_loss


class RedundancyReconstructionLoss(CorrelationRedundancy, nn.Module):
    def __init__(
        self,
        lambda_strength: float,
        *,
        kernel_size: int = 3,
        correlation_mode: str = "mean",
        comparison_mode: str = "both",
        normalization_mode: str = "chnorm_relu",
        postcomp_mode: str = "thresh",
        correlation_loss: str = "max",
    ) -> None:
        super().__init__()

        kernel_size = int(kernel_size)
        if kernel_size <= 0 or (kernel_size % 2) == 0:
            raise ValueError(f"kernel_size must be an odd positive int, got {kernel_size}")

        self.lambda_strength = float(lambda_strength)
        self.kernel_radius = (kernel_size - 1) // 2
        self.correlation_mode = correlation_mode
        self.comparison_mode = comparison_mode
        self.normalization_mode = normalization_mode
        self.postcomp_mode = postcomp_mode
        self.correlation_loss = correlation_loss
        self.mse = nn.MSELoss()

        self.last_mse_loss = torch.tensor(0.0)
        self.last_corr_total = torch.tensor(0.0)
        self.last_corr_by_layer: dict[str, torch.Tensor] = {}

    def forward(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
        feature_maps: list | None = None,
    ) -> torch.Tensor:
        mse_loss = self.mse(reconstruction, target)
        corr_total = reconstruction.new_zeros(())
        corr_by_layer: dict[str, torch.Tensor] = {}

        for idx, feature_map_item in enumerate(feature_maps or []):
            layer_name, feature_map, _, _ = self.parse_feature_map(feature_map_item, idx=idx)
            corr_mat = self.luca_fn(
                feature_map=feature_map,
                kernel_radius=self.kernel_radius,
                correlation_mode=self.correlation_mode,
                comparison_mode=self.comparison_mode,
                normalization_mode=self.normalization_mode,
                postcomp_mode=self.postcomp_mode,
            )
            corr_val = self.reduce_corr_matrix(corr_mat, self.correlation_loss)
            corr_total = corr_total + corr_val
            corr_by_layer[layer_name] = corr_val.detach()

        self.last_mse_loss = mse_loss.detach()
        self.last_corr_total = corr_total.detach()
        self.last_corr_by_layer = corr_by_layer
        return mse_loss + (self.lambda_strength * corr_total)


def get_loss(cfg: dict[str, Any]) -> nn.Module:
    training_mode = cfg["training_mode"]
    loss_type = cfg["loss_type"]

    if training_mode == "classification" and loss_type == "regular":
        return ClassificationLoss()
    if training_mode == "classification" and loss_type == "redundancy":
        return RedundancyLoss(
            lambda_strength=cfg["lambda_strength"],
            kernel_size=int(cfg["kernel_size"]),
            correlation_mode=cfg["correlation_mode"],
            comparison_mode=cfg["comparison_mode"],
            normalization_mode=cfg["normalization_mode"],
            postcomp_mode=cfg["postcomp_mode"],
            correlation_loss=cfg["correlation_loss"],
            local=bool(cfg.get("local", False)),
        )
    if training_mode == "reconstruction" and loss_type == "regular":
        return ReconstructionLoss()
    if training_mode == "reconstruction" and loss_type == "redundancy":
        return RedundancyReconstructionLoss(
            lambda_strength=cfg["lambda_strength"],
            kernel_size=int(cfg["kernel_size"]),
            correlation_mode=cfg["correlation_mode"],
            comparison_mode=cfg["comparison_mode"],
            normalization_mode=cfg["normalization_mode"],
            postcomp_mode=cfg["postcomp_mode"],
            correlation_loss=cfg["correlation_loss"],
        )

    raise ValueError(
        "Expected training_mode in {'classification', 'reconstruction'} "
        "and loss_type in {'regular', 'redundancy'}."
    )

