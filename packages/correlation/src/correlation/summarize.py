"""Compute channel-correlation summaries from saved activation samples."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq
import torch


PHASE_ORDER = {"before": 0, "after": 1}
DEFAULT_QUANTILES = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)


@dataclass(frozen=True)
class LayerCorrelationSummary:
    run_id: str
    epoch: int
    phase: str
    layer: str
    channels: int
    observations: int
    pair_count: int
    mean: float
    median: float
    std: float
    min: float
    max: float
    q01: float
    q05: float
    q25: float
    q75: float
    q95: float
    q99: float
    fraction_gt_05: float
    fraction_lt_neg_05: float
    bin_edges: list[float]
    bin_counts: list[int]
    bin_density: list[float]
    architecture: str = ""
    dataset: str = ""
    samplesize_image: str = ""
    samplesize_patches: str = ""


def _required_metadata(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"Activation artifact must contain a metadata dictionary: {path}")
    for key in (
        "run_id",
        "epoch",
        "phase",
        "architecture",
        "dataset",
        "samplesize_image",
        "samplesize_patches",
    ):
        if key not in metadata:
            raise ValueError(f"Activation artifact metadata is missing {key!r}: {path}")
    return metadata


def _required_activations(payload: dict[str, Any], path: Path) -> dict[str, torch.Tensor]:
    activations = payload.get("activations")
    if not isinstance(activations, dict):
        raise ValueError(f"Activation artifact must contain an activations dictionary: {path}")
    layers = {str(name): value for name, value in activations.items() if isinstance(value, torch.Tensor)}
    if len(layers) != len(activations):
        raise ValueError(f"Every activation entry must be a torch.Tensor: {path}")
    if not layers:
        raise ValueError(f"No layer activations found in artifact: {path}")
    return layers


def _activation_matrix(tensor: torch.Tensor) -> torch.Tensor:
    values = tensor.detach().to(dtype=torch.float32, device="cpu")
    if values.ndim != 3:
        raise ValueError("Layer activations must be shaped (samplesize_image, samplesize_patches, channels).")
    matrix = values.reshape(-1, values.shape[-1])
    if matrix.shape[1] < 2:
        raise ValueError("At least two channels are required to compute pairwise correlations.")
    if matrix.shape[0] < 2:
        raise ValueError("At least two observations are required to compute Pearson correlations.")
    return matrix


def pairwise_channel_correlations(tensor: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    """Return upper-triangular Pearson correlations between channels."""
    matrix = _activation_matrix(tensor)
    channels_by_observation = matrix.T.contiguous()
    centered = channels_by_observation - channels_by_observation.mean(dim=1, keepdim=True)
    norms = torch.linalg.vector_norm(centered, dim=1, keepdim=True).clamp_min(eps)
    normalized = centered / norms
    corr = normalized @ normalized.T
    corr = corr.clamp(min=-1.0, max=1.0)
    indices = torch.triu_indices(corr.shape[0], corr.shape[1], offset=1)
    return corr[indices[0], indices[1]]


def _histogram(values: torch.Tensor, bins: int) -> tuple[list[float], list[int], list[float]]:
    edges = torch.linspace(-1.0, 1.0, bins + 1)
    counts = torch.histc(values, bins=bins, min=-1.0, max=1.0).to(torch.int64)
    width = 2.0 / bins
    total = int(counts.sum())
    if total:
        density = counts.to(torch.float64) / (total * width)
    else:
        density = torch.zeros_like(counts, dtype=torch.float64)
    return edges.tolist(), [int(value) for value in counts.tolist()], density.tolist()


def _float(value: torch.Tensor | float) -> float:
    result = float(value)
    return result if math.isfinite(result) else float("nan")


def summarize_layer(
    *,
    run_id: str,
    epoch: int,
    phase: str,
    layer: str,
    activations: torch.Tensor,
    bins: int = 100,
    architecture: str = "",
    dataset: str = "",
    samplesize_image: str = "",
    samplesize_patches: str = "",
) -> LayerCorrelationSummary:
    matrix = _activation_matrix(activations)
    correlations = pairwise_channel_correlations(matrix)
    if correlations.numel() == 0:
        raise ValueError(f"Layer {layer!r} does not have enough channels for pairwise correlation.")

    quantiles = torch.quantile(
        correlations,
        torch.tensor(DEFAULT_QUANTILES, dtype=torch.float32),
    )
    bin_edges, bin_counts, bin_density = _histogram(correlations, bins)
    return LayerCorrelationSummary(
        run_id=run_id,
        epoch=int(epoch),
        phase=phase,
        layer=layer,
        channels=int(matrix.shape[1]),
        observations=int(matrix.shape[0]),
        pair_count=int(correlations.numel()),
        mean=_float(correlations.mean()),
        median=_float(quantiles[3]),
        std=_float(correlations.std(unbiased=False)),
        min=_float(correlations.min()),
        max=_float(correlations.max()),
        q01=_float(quantiles[0]),
        q05=_float(quantiles[1]),
        q25=_float(quantiles[2]),
        q75=_float(quantiles[4]),
        q95=_float(quantiles[5]),
        q99=_float(quantiles[6]),
        fraction_gt_05=_float((correlations > 0.5).to(torch.float32).mean()),
        fraction_lt_neg_05=_float((correlations < -0.5).to(torch.float32).mean()),
        bin_edges=bin_edges,
        bin_counts=bin_counts,
        bin_density=bin_density,
        architecture=architecture,
        dataset=dataset,
        samplesize_image=samplesize_image,
        samplesize_patches=samplesize_patches,
    )


def summarize_activation_file(path: Path, *, bins: int = 100, run_id: str | None = None) -> list[LayerCorrelationSummary]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Activation artifact must contain a dictionary: {path}")

    metadata = _required_metadata(payload, path)
    artifact_run_id = str(run_id or metadata["run_id"])
    epoch = int(metadata["epoch"])
    phase = str(metadata["phase"])
    layers = _required_activations(payload, path)

    return [
        summarize_layer(
            run_id=artifact_run_id,
            epoch=epoch,
            phase=phase,
            layer=layer,
            activations=activations,
            bins=bins,
            architecture=str(metadata["architecture"]),
            dataset=str(metadata["dataset"]),
            samplesize_image=str(metadata["samplesize_image"]),
            samplesize_patches=str(metadata["samplesize_patches"]),
        )
        for layer, activations in sorted(layers.items())
    ]


def find_activation_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(
        path
        for path in root.rglob("*.pt")
        if path.name not in {"checkpoint.pt", "model.pt"} and "checkpoint" not in path.parts
    )


def _stats_row(summary: LayerCorrelationSummary) -> dict[str, str | int | float]:
    return {
        "run_id": summary.run_id,
        "epoch": summary.epoch,
        "phase": summary.phase,
        "layer": summary.layer,
        "channels": summary.channels,
        "observations": summary.observations,
        "pair_count": summary.pair_count,
        "mean": summary.mean,
        "median": summary.median,
        "std": summary.std,
        "min": summary.min,
        "max": summary.max,
        "q01": summary.q01,
        "q05": summary.q05,
        "q25": summary.q25,
        "q75": summary.q75,
        "q95": summary.q95,
        "q99": summary.q99,
        "fraction_gt_05": summary.fraction_gt_05,
        "fraction_lt_neg_05": summary.fraction_lt_neg_05,
        "architecture": summary.architecture,
        "dataset": summary.dataset,
        "samplesize_image": summary.samplesize_image,
        "samplesize_patches": summary.samplesize_patches,
    }


def _histogram_rows(summaries: Iterable[LayerCorrelationSummary]) -> list[dict[str, str | int | float]]:
    rows: list[dict[str, str | int | float]] = []
    for summary in summaries:
        for idx, (count, density) in enumerate(zip(summary.bin_counts, summary.bin_density, strict=True)):
            rows.append(
                {
                    "run_id": summary.run_id,
                    "epoch": summary.epoch,
                    "phase": summary.phase,
                    "layer": summary.layer,
                    "bin_index": idx,
                    "bin_left": summary.bin_edges[idx],
                    "bin_right": summary.bin_edges[idx + 1],
                    "bin_center": (summary.bin_edges[idx] + summary.bin_edges[idx + 1]) / 2.0,
                    "count": count,
                    "density": density,
                    "architecture": summary.architecture,
                    "dataset": summary.dataset,
                    "samplesize_image": summary.samplesize_image,
                    "samplesize_patches": summary.samplesize_patches,
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def write_summaries(
    summaries: list[LayerCorrelationSummary],
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_rows = [_stats_row(summary) for summary in summaries]
    histogram_rows = _histogram_rows(summaries)

    stats_path = output_dir / "layer_stats.csv"
    parquet_path = output_dir / "correlations.parquet"
    _write_csv(stats_path, stats_rows)
    _write_parquet(parquet_path, histogram_rows)
    return [stats_path, parquet_path]


def _phase_sort_key(phase: str) -> int:
    return PHASE_ORDER.get(phase, 99)


def _summary_sort_key(summary: LayerCorrelationSummary) -> tuple[int, int, str]:
    return summary.epoch, _phase_sort_key(summary.phase), summary.layer


def summarize_activation_root(
    *,
    activation_root: Path,
    output_root: Path,
    run_id: str | None = None,
    bins: int = 100,
) -> list[Path]:
    activation_root = activation_root.resolve()
    if run_id is None and activation_root.is_dir():
        child_runs = [
            child
            for child in sorted(activation_root.iterdir())
            if child.is_dir() and find_activation_files(child)
        ]
        direct_files = [path for path in activation_root.glob("*.pt") if path.is_file()]
        if child_runs and not direct_files:
            written: list[Path] = []
            for child in child_runs:
                written.extend(
                    summarize_activation_root(
                        activation_root=child,
                        output_root=output_root,
                        run_id=child.name,
                        bins=bins,
                    )
                )
            return written

    files = find_activation_files(activation_root)
    if not files:
        raise FileNotFoundError(f"No .pt activation artifacts found under {activation_root}")

    summaries: list[LayerCorrelationSummary] = []
    for path in files:
        summaries.extend(summarize_activation_file(path, bins=bins, run_id=run_id))

    summaries.sort(key=_summary_sort_key)
    resolved_run_id = run_id or summaries[0].run_id
    return write_summaries(summaries, output_root / resolved_run_id)
