# Lifetime Ghost

At each block and spatial position (the lifetime unit), select the
ceil(0.25 × batch size) examples with the largest prediction errors. This is
across-example lifetime winner-take-all, rather than top-k patches within each
image. Spatial positions have equal batch selection frequency; harder images can
receive more attention than easier images. Maintain an EMA of the lowest winning
error per position (decay 0.95). Evaluation uses these stored training thresholds,
without batch statistics or updates; its observed budget can drift. A batch of
one during training selects every position, so use the configured batch of 128.
The last 72-example training batch also has exactly 25% selected. Compare with
`src/predictive_ghost`, which fixes the budget separately for each image.

## Shared architecture and hypotheses

These independent CIFAR-scale implementations follow the repository's NAT and
**Vision Longformer (ViL)** convention. ViL here means Vision Longformer.
Both use widths 64/128/256, depths 2/2/4, heads 2/4/8, spatial grids
16×16, 8×8, 4×4, and local windows of 3×3. NAT shifts boundary windows inward;
ViL clips them and retains one global token per stage, with separate global
Q/K/V projections. No surround-modulation operator is included.

Before attention, a center-excluding eight-neighbor average and a learned 1×1
convolution predict the normalized spatial input. Channel-mean squared error
against the detached observed input is the routing score. The training objective
is label-smoothed cross entropy plus 0.1 × mean prediction MSE across blocks.
This is a feedforward predictive-coding-inspired proxy, without iterative latent
inference or top-down feedback. Representation collapse and score quality should
be checked through prediction MSE, selected/skipped error, and task accuracy.

Only selected spatial queries undergo attention and its output projection.
K/V are projected for all tokens so every patch remains available as context.
Neighborhoods are gathered directly; no dense N×N local attention matrix is made.
ViL global queries still attend to all spatial and global tokens. All patches
retain their residual stream and receive the MLP, so a 75% spatial query reduction
is not a 75% FLOP reduction.

Scatter attended outputs into a sparse feature map. Generate skipped updates as
`channel_scale * (context_prediction + depthwise_5x5(anchors) / local_density)`.
Density is the local mean selection mask, floored at 1/25. The channel scale
starts at 0.1 and is learned. This can generate a prediction even where there is
no nearby attended anchor. Selected positions use their actual attention updates.
This spatial feature synthesis borrows the cheap-operation idea from GhostNet;
it generates feature updates, rather than RGB pixels. Classification trains the
generator; there is no dense teacher or reconstruction target for ghost outputs.

The hypothesis is that prediction-error routing beats random routing at the same
query budget, and lifetime allocation trades a variable per-image budget for
more even spatial use. Dense attention provides an accuracy/cost reference.
Falsify these using final test accuracy and measured validation throughput/memory,
while checking actual query fractions rather than assuming matched inference cost.

## Comparison and execution

The two folders own their own enums, data logic, and models. The dataset-agnostic
runner is `src/patch_training.py`. Train on prepared CIFAR-100: 45,000 training,
5,000 validation, and 10,000 test examples, with crop/flip augmentation only for
training. Defaults match the earlier study: 30 epochs, seed 0, batch 128,
AdamW 5e-4, weight decay 0.05 excluding biases/norms, three-epoch warmup, cosine
schedule, FP16 AMP, gradient clipping at 1, and smoothing 0.1. Final-epoch test
evaluation occurs once, with no best-validation checkpoint selection.
This single-seed study is exploratory; it does not establish statistical significance.

All training choices are in YAML. Launch from the repository root:

```bash
conda activate ai-science
env -u LD_LIBRARY_PATH -u LD_PRELOAD python -m scripts.verify_patch_ghost
env -u LD_LIBRARY_PATH -u LD_PRELOAD python -m scripts.run_patch_ghost_study
```

The last command detaches a supervisor, records its PID in both experiment
folders, snapshots YAML configs, and runs eight models serially on one GPU:
NAT dense/random/predictive, ViL dense/random/predictive, then NAT/ViL lifetime.
Controls live in predictive_ghost and serve both experiments. Every policy
constructs identical parameters with paired initialization; the dense control
also trains the predictor, but does not use ghost updates. Random selection is
stochastic at evaluation as well as training; seeds reproduce it within the same
execution order. The script stops if any run fails and writes study.json in both
checkpoint roots; comparison.json is written after all eight runs finish.
Do not launch the command again while its supervisor is active.

Single run (sole training entry point):

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD python -m src.lifetime_ghost.training --config src/lifetime_ghost/config.yaml
```

A W&B-compatible grid sweep is in `sweep.yaml`; dotted YAML overrides supplied by
`wandb agent` use the same config loader. Expand seeds and budgets there for later
studies. The study launcher reads its families/policies/seeds from config.yaml.

## Tracking and artifacts

Online W&B is required by default and initialization errors fail the run.
Verification explicitly uses offline tracking for synthetic smoke tests only.
Stable project: `lifetime_ghost`. Each training process creates one UTC timestamp used
for its run group and output directory. Resolved config records model parameters,
split counts, token grids, backend, library versions, and device metadata.

W&B and local metrics include train/validation/test loss (cross entropy), accuracy,
top-5 accuracy, objective, prediction loss, selected/skipped error, query fraction,
per-block attended fractions, per-image fraction variability, EMA spatial-use
variability, inactive spatial fraction, examples/sec, seconds, GPU peak allocated
memory, epoch, optimizer LR, and AMP scale. Lifetime-frequency statistics use an
EMA of batch masks with update weight 0.05, not an exact whole-study counter.
Validation/test loss has no smoothing; objective includes the auxiliary term.
Throughput includes loading and metric collection, with CUDA synchronization at
pass boundaries. Training and evaluation costs must be compared separately.

Under `checkpoints/lifetime_ghost/runs/<UTC>_<family>_<policy>_seed0/`:
resolved config.yaml, metrics.jsonl, batch_metrics.jsonl, W&B URL/state,
checkpoints/last.pt (model including lifetime thresholds, optimizer, AMP, epoch,
config), per-epoch images and routing arrays, result.json, and COMPLETE.
No automatic resume is implemented. Routing figures show the first example of
the final validation batch; NPZ files also save per-block frequency/threshold maps.
Study output is in study.log, study.pid, and study.json at each experiment root.

## Sources

- [Neighborhood Attention Transformer](https://github.com/SHI-Labs/Neighborhood-Attention-Transformer)
- [Vision Longformer](https://github.com/microsoft/vision-longformer)
- [Rao and Ballard, predictive coding](https://www.nature.com/articles/nn0199_79)
- [Winner-Take-All Autoencoders, lifetime sparsity](https://arxiv.org/abs/1409.2752)
- [GhostNet, cheap feature generation](https://openaccess.thecvf.com/content_CVPR_2020/html/Han_GhostNet_More_Features_From_Cheap_Operations_CVPR_2020_paper.html)
