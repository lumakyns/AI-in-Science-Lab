# CNN weight patterns

Use the `ai-science` Conda environment from the repository root. Each training
process trains exactly one model, selected by `dataset` and `seed` in YAML.
The grid sweep selects CIFAR-10 and CIFAR-100, seed 0, and 10 epochs: two runs.
All runs use the stable W&B project `cnn1`.

Create the sweep, then start a standard W&B agent using the sweep ID it prints:

```bash
conda activate ai-science
wandb sweep src/cnn1/sweep.yaml
env -u LD_LIBRARY_PATH -u LD_PRELOAD wandb agent <entity>/cnn1/<sweep-id>
```

For a detached agent, use this instead of the foreground agent command:

```bash
mkdir -p checkpoints/cnn1
nohup setsid env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  python -m wandb agent <entity>/cnn1/<sweep-id> \
  >> checkpoints/cnn1/study.log 2>&1 < /dev/null &
echo $! > checkpoints/cnn1/study.pid
```

Check for an existing agent before starting another. Reuse its sweep ID to
continue an existing sweep; run `wandb sweep` only to create a new one.
There is no custom launcher. All CNN1 Python source lives in `src/cnn1/`.

The inherited system `LD_LIBRARY_PATH` in this host causes CUDA symbol errors.
The commands clear library overrides only for the launched process. PyTorch's
bundled CUDA libraries work without explicit library paths or preload.
For a single model, or the GPU/synthetic check:

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD python -m src.cnn1.training --config src/cnn1/config.yaml
env -u LD_LIBRARY_PATH -u LD_PRELOAD python -m src.cnn1.verify
```

The architecture is Conv(3→32, 3×3), ReLU, Conv(32→64, 20×20), ReLU,
Conv(64→128, 3×3), ReLU, global average pooling, and a linear classifier.
All convolutions preserve 32×32 resolution; the even kernel uses zero padding
of 9 pixels left/top and 10 right/bottom. No batch normalization or intermediate
pooling is used. AdamW, cosine scheduling, augmentation, loaders, seeds,
checkpoints, image scale, and online W&B are controlled by config.yaml.

Prepared train and validation partitions are merged into all 50,000 official
training images. Test loss and accuracy use all 10,000 test images after every
epoch, as requested. Training accuracy is measured on augmented batches during
updates. Accuracy is a fraction. Repeated test monitoring is part of this
exploratory run; no best model is selected using test accuracy.

For the 20×20 convolution only, save **one map for the whole layer**:
`weight.mean(dim=(0, 1))`, reducing `[out_channels, in_channels, 20, 20]` to
`[20, 20]`. These are signed raw weights, with no absolute value or activation
averaging. Positive and negative filters may cancel. Each epoch (plus epoch 0)
produces a PNG with a fixed symmetric color scale and an exact NPY array.
Map extrema, standard deviation and clipped fraction are logged alongside it.

Artifacts live under
`checkpoints/cnn1/runs/<UTC timestamp>_<dataset>_seed<seed>/<dataset>_seed<seed>/`:
`images/features.3_epoch_XXX.{png,npy}`, `metrics.jsonl`, `batch_metrics.jsonl`,
`config.yaml`, `checkpoints/last.pt`, `result.json`, and `logs/wandb/`.
`checkpoints/cnn1/latest` points to the newest run. Each run writes `COMPLETE`
only after its own model finishes. Tracking failures raise errors; explicit
offline mode is available in YAML for smoke checks.

Sweep parameters are passed as dotted YAML overrides using
`${args_no_hyphens}`. Add parameter values to `sweep.yaml` to expand the grid;
training contains no internal dataset or seed loop.
