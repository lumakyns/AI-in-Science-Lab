# Surround modulation in NAT and Vision Longformer

This experiment trains **10 CIFAR-100 models from scratch**: two hierarchical
attention backbones × five surround variants, 30 epochs each, seed 0.
All choices are in `config.yaml`, with the comparison defined in `sweep.yaml`.

| Variant | Initial kernel | Learned during training |
| --- | --- | --- |
| `baseline` | No surround operation | Backbone only |
| `dog_fixed` | Center-normalized DoG | Backbone only |
| `random_fixed` | Spatially scrambled DoG | Backbone only |
| `dog_trainable` | Center-normalized DoG | Backbone and independent kernel coefficients |
| `random_trainable` | Spatially scrambled DoG | Backbone and independent kernel coefficients |

The surround operator follows equations 1–3 of
[Hasani et al., NeurIPS 2019](https://proceedings.neurips.cc/paper_files/paper/2019/file/c535e3a7f97daf1c4b1eb03cc8e31623-Paper.pdf).
Construct two normalized continuous Gaussian samples with standard deviations
1.2 and 1.4, subtract them, then divide by the central DoG value. Do not normalize
the discrete Gaussian sums or force zero mean. The resulting center weight is 1.
Apply a 5×5, zero-padded depthwise convolution directly, without a new residual,
bias, or activation. This preserves the original center contribution.

The placement is an explicit transformer adaptation: after the first complete
attention/MLP block of the middle stage (stage 1, block 0, zero-based), on half
of its spatial channels. The other half and ViL global tokens bypass modulation.
This gives 64 independent kernels and 1,600 additional parameters when trainable.
Random controls independently permute the same 25 DoG entries in each channel,
exactly preserving empirical mean, variance, and norm while removing radial
organization. Fixed/trainable counterparts use identical starting kernels.
`dog_trainable` means **DoG initialization, freely trainable weights**, rather
than learning Gaussian widths under a DoG constraint.

## Backbones and comparison scope

These are compact CIFAR adaptations based on
[Neighborhood Attention Transformer](https://github.com/SHI-Labs/Neighborhood-Attention-Transformer)
and [Multi-Scale Vision Longformer](https://github.com/microsoft/vision-longformer).
They are not official ImageNet-size models, pretrained checkpoints, or a
reproduction of the original surround paper's CNN experiments.

Both use a two-convolution tokenizer (3×3 stride 1 then 3×3 stride 2),
three stages with token grids 16×16, 8×8, 4×4, widths 64/128/256, depths 2/2/4,
heads 2/4/8, MLP expansion 4, pre-norm residual blocks, relative position bias,
and stochastic depth up to 0.1. Stages downsample with 3×3 stride-2 convolutions.
NAT uses a 3×3 neighborhood shifted inward at boundaries to preserve nine
neighbors per query, and final spatial average pooling. ViL uses exact centered
3×3 neighborhoods clipped at boundaries, with one global token per stage.
Local queries also attend to globals; global queries attend to every token
using separate Q/K/V and output projections. Each stage initializes its own
global token, following the multiscale ViL design; classify the final global.

The local connectivity is implemented with dense masked PyTorch scaled-dot-product
attention, appropriate for these small grids. It preserves the intended attention
graphs but has quadratic computation/storage rather than the linear scaling of
NATTEN or sparse ViL implementations. No NATTEN, TVM, or custom CUDA extension is
required. Verification compares attention outputs and input gradients against
explicit per-query references. Base parameter counts are approximately 4.1M for
NAT and 5.3M for ViL; compare ablation effects within each family rather than
treating the backbones as parameter matched.

Backbone initializations, data order, augmentation seeds, and training settings
are paired within each architecture. Random kernels use a separate generator to
avoid altering backbone initialization. A single seed is an exploratory comparison;
it does not estimate variability across initializations.

## Data and training

Use native 32×32 images with CIFAR-100 mean/std normalization. Prepared splits
contain 45,000 training, 5,000 validation, and 10,000 test images. Training uses
random crop with four-pixel padding and horizontal flip. Validation/test are
deterministic. Validation is evaluated each epoch; the test set is evaluated
once after epoch 30, using the final model. No best-model selection or early
stopping is used. Train loss uses label smoothing 0.1; evaluation loss is ordinary
cross entropy, so their magnitudes differ by construction.

Defaults: AdamW at 5e-4, weight decay 0.05 on multidimensional parameters and no
decay on biases/norm scales, batch size 128, three-epoch linear warmup followed by
cosine decay to 1% of initial LR, clipping at gradient norm 1, FP16 AMP with a
gradient scaler, and four loader workers. Attention/MLP dropout is zero.

## Run and inspect

Run from the repository root in Conda environment `ai-science`:

```bash
conda activate ai-science
python data/download.py cifar100
env -u LD_LIBRARY_PATH -u LD_PRELOAD python -m src.surround_attention.verify
env -u LD_LIBRARY_PATH -u LD_PRELOAD wandb sweep src/surround_attention/sweep.yaml
env -u LD_LIBRARY_PATH -u LD_PRELOAD wandb agent --count 10 <entity>/surround_attention/<sweep-id>
```

The verification command performs actual GPU forward/backward checks for every
variant and configured batch-size AMP checks. It then exercises both families
with one epoch of synthetic data and **explicit offline W&B tracking**, under
`checkpoints/surround_attention/smoke/`. It does not launch the full comparison.
The sweep uses online tracking; tracking errors fail the run.
Clear inherited CUDA library overrides only in the launched process on this host.

Single configured run (the sole training entry point):

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD python -m src.surround_attention.training --config src/surround_attention/config.yaml
```

W&B sweep arguments use dotted YAML overrides through `${args_no_hyphens}`.
Change seeds or other choices in `sweep.yaml` to expand the study.
The standard W&B agent serializes the ten workers on the available GPU.
Check for a running agent before creating or launching another sweep.

Artifacts are stored under
`checkpoints/surround_attention/runs/<UTC timestamp>_<family>_<variant>_seed<seed>/`:

- `config.yaml`: resolved settings, model/device/data metadata.
- `metrics.jsonl`, `batch_metrics.jsonl`: epoch/batch progress.
- `images/epoch_XXX.png`: per-epoch training/validation plots and mean kernel.
- `images/surround_epoch_XXX.npy`: all surround kernels, including initialization.
  Baseline arrays are unused reference buffers, not applied convolutions.
- `checkpoints/last.pt`: atomically replaced model/optimizer/AMP checkpoint.
- `wandb_url.txt`, `logs/wandb/`: tracking link and local W&B state.
- `result.json`, `COMPLETE`: final metrics and marker after successful W&B finish.

The detached study writes `study.log`, `study.pid`, and `study.json` under
`checkpoints/surround_attention/`. All artifacts and data remain untracked.
