# CNN weight patterns

Visualize existing full runs without training or W&B access:

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD conda run --no-capture-output -n ai-science python -m src.cnn2.visualize_weights
```

This writes a local `index.html` report, synchronized GIF (initialization and
every saved epoch), PNG frames, exact stacked mean arrays, and final-epoch
channel grids under `checkpoints/cnn2/visualizations/<UTC timestamp>/`.
The GIF uses the saved signed means over both channel axes, with one fixed
symmetric scale spanning all epochs and runs without clipping. Full per-epoch
weights were not retained, so individual channel histories cannot be recovered.
Final grids show 16 random individual input/output channel kernels (sample seed
0), and each output filter averaged over its input channels. The manifest
records source checkpoints, epochs, scales, and sampled zero-based channel
indices. Final saved means are checked against the full checkpoint.
Use `--runs <run-directory> ...` to select runs, `--samples` and `--sample-seed`
to change sampling, or `--output` to choose a fresh directory under
`checkpoints/cnn2/`. Selected runs must have matching complete epoch sequences.

Use the `ai-science` Conda environment from the repository root. Each training
process trains exactly one model, selected by `dataset` and `seed` in YAML.
The grid sweep selects CIFAR-10 and CIFAR-100, seed 0, and 10 epochs: two runs.
All runs use the stable W&B project `cnn2`.

Create the sweep, then start a standard W&B agent using the sweep ID it prints:

```bash
conda activate ai-science
wandb sweep src/cnn2/sweep.yaml
env -u LD_LIBRARY_PATH -u LD_PRELOAD wandb agent <entity>/cnn2/<sweep-id>
```

For a detached tmux agent, run from the repository root (substitute the sweep ID):

```bash
mkdir -p checkpoints/cnn2
tmux new-session -d -s cnn2 -c "$PWD" \
  'export PATH=/home/lmacadar/.conda/envs/ai-science/bin:$PATH; echo $$ > checkpoints/cnn2/study.pid; env -u LD_LIBRARY_PATH -u LD_PRELOAD /home/lmacadar/.conda/envs/ai-science/bin/python -m wandb agent --count 2 <entity>/cnn2/<sweep-id> >> checkpoints/cnn2/study.log 2>&1 < /dev/null; status=$?; exit $status'
tmux attach -t cnn2
```

The worker command uses `python` from PATH, so put the environment first in PATH
even when the agent uses an absolute interpreter. Keep the tmux shell alive
while the agent runs.

Check for an existing agent before starting another. Reuse its sweep ID to
continue an existing sweep; run `wandb sweep` only to create a new one.
There is no custom launcher. All CNN2 Python source lives in `src/cnn2/`.

The inherited system `LD_LIBRARY_PATH` in this host causes CUDA symbol errors.
The commands clear library overrides only for the launched process. PyTorch's
bundled CUDA libraries work without explicit library paths or preload.
For a single model, or the GPU/synthetic check:

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD python -m src.cnn2.training --config src/cnn2/config.yaml
env -u LD_LIBRARY_PATH -u LD_PRELOAD python -m src.cnn2.verify
```

The architecture is Conv(3→32, 3×3), ReLU, Conv(32→32, 20×20), ReLU,
Conv(32→32, 3×3), ReLU, global average pooling, and a linear classifier.
Hidden channel width is constant at 32 after the initial RGB-to-32 expansion.
The model rejects unequal channel widths; a different constant width can be set
through `model.channels` in YAML. All convolutions preserve 32×32 resolution; the even kernel uses zero padding
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
`checkpoints/cnn2/runs/<UTC timestamp>_<dataset>_seed<seed>/<dataset>_seed<seed>/`:
`images/features.3_epoch_XXX.{png,npy}`, `metrics.jsonl`, `batch_metrics.jsonl`,
`config.yaml`, `checkpoints/last.pt`, `result.json`, and `logs/wandb/`.
`checkpoints/cnn2/latest` points to the newest run. Each run writes `COMPLETE`
only after its own model finishes. Tracking failures raise errors; explicit
offline mode is available in YAML for smoke checks.

Sweep parameters are passed as dotted YAML overrides using
`${args_no_hyphens}`. Add parameter values to `sweep.yaml` to expand the grid;
training contains no internal dataset or seed loop.
