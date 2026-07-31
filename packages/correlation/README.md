# Correlation analysis

## Conda dependencies

Create the environment used on the training/analysis server:

```bash
conda env create -f packages/correlation/environment.yaml
conda activate correlation
```

This package contains the post-training analysis tools for sampled activation
artifacts. Training is expected to save activation samples shaped as
`(samplesize_image, samplesize_patches, channels)` for every captured layer.

The analysis step computes pairwise Pearson correlations between channels in
each layer. Plotting then renders per-layer PDFs, before/after overlays,
epoch-trajectory heatmaps, and summary-statistic trends.

## Expected activation artifact

Each `.pt` activation artifact must be a dictionary with this shape:

```python
{
    "metadata": {
        "run_id": "example",
        "epoch": 1,
        "phase": "after",
        "architecture": "vgg16",
        "dataset": "cifar10",
        "samplesize_image": 256,
        "samplesize_patches": 64,
    },
    "activations": {
        "conv1": torch.randn(samplesize_image, samplesize_patches, channels),
    },
}
```

## Usage

```bash
python3 packages/correlation/scripts/summarize_correlations.py \
  --activation-root packages/correlation/outputs/activation_samples/example \
  --output-root packages/correlation/outputs/summaries

python3 packages/correlation/scripts/plot_summaries.py \
  --summary-root packages/correlation/outputs/summaries/example \
  --output-root packages/correlation/outputs/figures
```
