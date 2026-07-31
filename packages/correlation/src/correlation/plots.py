"""Plot correlation PDFs and layer-level summary trends."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
import pyarrow.parquet as pq

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BEFORE_COLOR = "#4b5563"
AFTER_COLOR = "#d95f02"
TRAJECTORY_CMAP = "viridis"
PHASE_ORDER = {"before": 0, "after": 1}


def _safe_layer_name(name: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in name)


def _infer_run_id(summary_root: Path) -> str:
    return summary_root.name


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist()


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _has_summary_files(path: Path) -> bool:
    return (path / "correlations.parquet").exists() and (path / "layer_stats.csv").exists()


def _phase_sort_key(phase: str) -> int:
    return PHASE_ORDER.get(phase, 99)


def _write_figure(fig, output_base: Path, written: list[Path]) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        path = output_base.with_suffix(suffix)
        fig.savefig(path, dpi=160, bbox_inches="tight")
        written.append(path)
    plt.close(fig)


def _group_histograms(rows: list[dict[str, Any]]) -> dict[tuple[str, int, str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["run_id"]),
            int(row["epoch"]),
            str(row["phase"]),
            str(row["layer"]),
        )
        grouped[key].append(row)
    for values in grouped.values():
        values.sort(key=lambda row: int(row["bin_index"]))
    return grouped


def _group_stats(rows: list[dict[str, Any]]) -> dict[tuple[str, int, str, str], dict[str, Any]]:
    return {
        (
            str(row["run_id"]),
            int(row["epoch"]),
            str(row["phase"]),
            str(row["layer"]),
        ): row
        for row in rows
    }


def _label_for(key: tuple[str, int, str, str], stats: dict[tuple[str, int, str, str], dict[str, Any]]) -> str:
    _, epoch, phase, _ = key
    row = stats.get(key, {})
    if row:
        return (
            f"epoch {epoch} {phase} "
            f"(mean={float(row['mean']):.3f}, med={float(row['median']):.3f}, "
            f"std={float(row['std']):.3f})"
        )
    return f"epoch {epoch} {phase}"


def _plot_layer_overlay(
    *,
    layer: str,
    before_key: tuple[str, int, str, str],
    after_key: tuple[str, int, str, str],
    histograms: dict[tuple[str, int, str, str], list[dict[str, Any]]],
    stats: dict[tuple[str, int, str, str], dict[str, Any]],
    output_dir: Path,
    written: list[Path],
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for key, color in ((before_key, BEFORE_COLOR), (after_key, AFTER_COLOR)):
        rows = histograms[key]
        centers = [float(row["bin_center"]) for row in rows]
        density = [float(row["density"]) for row in rows]
        ax.plot(centers, density, linewidth=2.0, color=color, label=_label_for(key, stats))
    ax.set_title(f"{layer} correlation PDF")
    ax.set_xlabel("Pearson correlation")
    ax.set_ylabel("Density")
    ax.set_xlim(-1.0, 1.0)
    ax.grid(axis="y", color="#d7dce2", linewidth=0.8)
    ax.legend(frameon=False, fontsize=8)
    _write_figure(fig, output_dir / "per_layer" / _safe_layer_name(layer), written)


def _plot_layer_trajectory(
    *,
    layer: str,
    keys: list[tuple[str, int, str, str]],
    histograms: dict[tuple[str, int, str, str], list[dict[str, Any]]],
    output_dir: Path,
    written: list[Path],
) -> None:
    keys = sorted(keys, key=lambda key: (key[1], _phase_sort_key(key[2])))
    labels = [f"{epoch} {phase}" for _, epoch, phase, _ in keys]
    density_matrix = [
        [float(row["density"]) for row in histograms[key]]
        for key in keys
    ]
    centers = [float(row["bin_center"]) for row in histograms[keys[0]]]

    fig, ax = plt.subplots(figsize=(8.4, max(3.5, 0.34 * len(keys))))
    image = ax.imshow(
        density_matrix,
        aspect="auto",
        origin="lower",
        cmap=TRAJECTORY_CMAP,
        extent=[min(centers), max(centers), -0.5, len(keys) - 0.5],
    )
    ax.set_title(f"{layer} correlation trajectory")
    ax.set_xlabel("Pearson correlation")
    ax.set_ylabel("Epoch phase")
    ax.set_xlim(-1.0, 1.0)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    fig.colorbar(image, ax=ax, label="Density")
    _write_figure(fig, output_dir / "trajectories" / _safe_layer_name(layer), written)


def _plot_summary_trends(
    *,
    stats_rows: list[dict[str, Any]],
    output_dir: Path,
    written: list[Path],
) -> None:
    by_layer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in stats_rows:
        by_layer[str(row["layer"])].append(row)

    metrics = (
        ("mean", "Mean correlation"),
        ("median", "Median correlation"),
        ("fraction_gt_05", "Fraction > 0.5"),
        ("fraction_lt_neg_05", "Fraction < -0.5"),
    )
    for layer, rows in by_layer.items():
        rows.sort(key=lambda row: (int(row["epoch"]), _phase_sort_key(str(row["phase"]))))
        x = [int(row["epoch"]) + (0.12 if str(row["phase"]) == "after" else -0.12) for row in rows]

        fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
        axes_flat = list(axes.flatten())
        colors = [AFTER_COLOR if str(row["phase"]) == "after" else BEFORE_COLOR for row in rows]
        for ax, (metric, label) in zip(axes_flat, metrics, strict=True):
            y = [float(row[metric]) for row in rows]
            ax.scatter(x, y, c=colors, s=24)
            ax.plot(x, y, color="#9aa3af", linewidth=1.0)
            ax.set_title(label)
            ax.grid(axis="y", color="#d7dce2", linewidth=0.8)
        for ax in axes_flat[2:]:
            ax.set_xlabel("Epoch")
        fig.suptitle(f"{layer} summary trends")
        _write_figure(fig, output_dir / "summary_trends" / _safe_layer_name(layer), written)


def _plot_aggregate_pdf(
    *,
    histograms: dict[tuple[str, int, str, str], list[dict[str, Any]]],
    output_dir: Path,
    written: list[Path],
) -> None:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for key, rows in histograms.items():
        run_id, epoch, phase, _layer = key
        grouped[(run_id, epoch, phase)].extend(rows)

    aggregate: dict[tuple[str, int, str], dict[float, float]] = {}
    for key, rows in grouped.items():
        density_by_center: dict[float, list[float]] = defaultdict(list)
        for row in rows:
            density_by_center[float(row["bin_center"])].append(float(row["density"]))
        aggregate[key] = {
            center: sum(values) / len(values)
            for center, values in density_by_center.items()
        }

    if not aggregate:
        return
    first_key = sorted(aggregate, key=lambda key: (key[1], _phase_sort_key(key[2])))[0]
    final_key = sorted(aggregate, key=lambda key: (key[1], _phase_sort_key(key[2])))[-1]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for key, color in ((first_key, BEFORE_COLOR), (final_key, AFTER_COLOR)):
        centers = sorted(aggregate[key])
        density = [aggregate[key][center] for center in centers]
        label = f"epoch {key[1]} {key[2]}"
        ax.plot(centers, density, linewidth=2.0, color=color, label=label)
    ax.set_title("Aggregate layer correlation PDF")
    ax.set_xlabel("Pearson correlation")
    ax.set_ylabel("Mean layer density")
    ax.set_xlim(-1.0, 1.0)
    ax.grid(axis="y", color="#d7dce2", linewidth=0.8)
    ax.legend(frameon=False, fontsize=8)
    _write_figure(fig, output_dir / "aggregate_model_pdf", written)


def _final_after_keys(
    histograms: dict[tuple[str, int, str, str], list[dict[str, Any]]],
) -> list[tuple[str, int, str, str]]:
    by_run_layer: dict[tuple[str, str], list[tuple[str, int, str, str]]] = defaultdict(list)
    for key in histograms:
        run_id, _epoch, _phase, layer = key
        by_run_layer[(run_id, layer)].append(key)
    keys: list[tuple[str, int, str, str]] = []
    for candidates in by_run_layer.values():
        sorted_candidates = sorted(candidates, key=lambda key: (key[1], _phase_sort_key(key[2])))
        keys.append(next((key for key in reversed(sorted_candidates) if key[2] == "after"), sorted_candidates[-1]))
    return keys


def _comparison_metadata(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("architecture", "")),
        str(row.get("dataset", "")),
        str(row.get("samplesize_image", "")),
        str(row.get("samplesize_patches", "")),
    )


def _plot_comparison_group(
    *,
    title: str,
    label_field: str,
    keys: list[tuple[str, int, str, str]],
    histograms: dict[tuple[str, int, str, str], list[dict[str, Any]]],
    output_base: Path,
    written: list[Path],
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for key in sorted(keys, key=lambda item: str(histograms[item][0].get(label_field, ""))):
        rows = histograms[key]
        label = str(rows[0].get(label_field, "")) or key[0]
        centers = [float(row["bin_center"]) for row in rows]
        density = [float(row["density"]) for row in rows]
        ax.plot(centers, density, linewidth=1.8, label=label)
    ax.set_title(title)
    ax.set_xlabel("Pearson correlation")
    ax.set_ylabel("Density")
    ax.set_xlim(-1.0, 1.0)
    ax.grid(axis="y", color="#d7dce2", linewidth=0.8)
    ax.legend(frameon=False, fontsize=8)
    _write_figure(fig, output_base, written)


def _plot_comparisons(
    *,
    histogram_rows: list[dict[str, Any]],
    output_root: Path,
    written: list[Path],
) -> None:
    histograms = _group_histograms(histogram_rows)
    final_keys = _final_after_keys(histograms)

    architecture_groups: dict[tuple[str, str, str, str], list[tuple[str, int, str, str]]] = defaultdict(list)
    dataset_groups: dict[tuple[str, str, str, str], list[tuple[str, int, str, str]]] = defaultdict(list)
    for key in final_keys:
        first_row = histograms[key][0]
        architecture, dataset, samplesize_image, samplesize_patches = _comparison_metadata(first_row)
        if architecture and dataset:
            architecture_groups[(dataset, samplesize_image, samplesize_patches, key[3])].append(key)
            dataset_groups[(architecture, samplesize_image, samplesize_patches, key[3])].append(key)

    for (dataset, samplesize_image, samplesize_patches, layer), keys in sorted(architecture_groups.items()):
        labels = {str(histograms[key][0].get("architecture", "")) for key in keys}
        if len(labels) < 2:
            continue
        sample_label = f"images_{samplesize_image}_patches_{samplesize_patches}"
        output_base = (
            output_root
            / "comparisons"
            / "architecture"
            / f"{dataset}_{sample_label}_{_safe_layer_name(layer)}"
        )
        _plot_comparison_group(
            title=f"{dataset} architecture comparison: {layer}",
            label_field="architecture",
            keys=keys,
            histograms=histograms,
            output_base=output_base,
            written=written,
        )

    for (architecture, samplesize_image, samplesize_patches, layer), keys in sorted(dataset_groups.items()):
        labels = {str(histograms[key][0].get("dataset", "")) for key in keys}
        if len(labels) < 2:
            continue
        sample_label = f"images_{samplesize_image}_patches_{samplesize_patches}"
        output_base = (
            output_root
            / "comparisons"
            / "dataset"
            / f"{architecture}_{sample_label}_{_safe_layer_name(layer)}"
        )
        _plot_comparison_group(
            title=f"{architecture} dataset comparison: {layer}",
            label_field="dataset",
            keys=keys,
            histograms=histograms,
            output_base=output_base,
            written=written,
        )


def plot_summary_root(
    *,
    summary_root: Path,
    output_root: Path,
    run_id: str | None = None,
) -> list[Path]:
    summary_root = summary_root.resolve()
    resolved_run_id = run_id or _infer_run_id(summary_root)
    hist_path = summary_root / "correlations.parquet"
    stats_path = summary_root / "layer_stats.csv"
    if not hist_path.exists() and summary_root.is_dir():
        child_roots = [child for child in sorted(summary_root.iterdir()) if child.is_dir() and _has_summary_files(child)]
        if child_roots:
            written: list[Path] = []
            all_histogram_rows: list[dict[str, Any]] = []
            for child in child_roots:
                written.extend(plot_summary_root(summary_root=child, output_root=output_root, run_id=child.name))
                all_histogram_rows.extend(_read_parquet(child / "correlations.parquet"))
            _plot_comparisons(
                histogram_rows=all_histogram_rows,
                output_root=output_root,
                written=written,
            )
            return written
    if not hist_path.exists():
        raise FileNotFoundError(f"Missing histogram rows: {hist_path}")
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing layer stats: {stats_path}")

    histogram_rows = _read_parquet(hist_path)
    stats_rows = _read_csv(stats_path)
    histograms = _group_histograms(histogram_rows)
    stats = _group_stats(stats_rows)
    output_dir = output_root / resolved_run_id
    written: list[Path] = []

    keys_by_layer: dict[str, list[tuple[str, int, str, str]]] = defaultdict(list)
    for key in histograms:
        keys_by_layer[key[3]].append(key)

    for layer, keys in sorted(keys_by_layer.items()):
        sorted_keys = sorted(keys, key=lambda key: (key[1], _phase_sort_key(key[2])))
        before_key = next((key for key in sorted_keys if key[1] == 0 and key[2] == "before"), sorted_keys[0])
        after_key = next((key for key in reversed(sorted_keys) if key[2] == "after"), sorted_keys[-1])
        _plot_layer_overlay(
            layer=layer,
            before_key=before_key,
            after_key=after_key,
            histograms=histograms,
            stats=stats,
            output_dir=output_dir,
            written=written,
        )
        _plot_layer_trajectory(
            layer=layer,
            keys=sorted_keys,
            histograms=histograms,
            output_dir=output_dir,
            written=written,
        )

    _plot_summary_trends(stats_rows=stats_rows, output_dir=output_dir, written=written)
    _plot_aggregate_pdf(histograms=histograms, output_dir=output_dir, written=written)
    return written
