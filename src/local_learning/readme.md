# Local Learning Notes

## W&B Logging Reference

This file reflects the current `training.py` logging surface. The old
visualization stack is archived in `old_readme.md`.

### Model Config

- `architecture_type`: selects the model family, for example `vgg16`,
  `resnet18`, or `densenet121`.
- `weights`: selects initialization. Use `random` for random initialization or
  `default` for the torchvision default weights. `default` is currently
  supported for `vgg16`, `resnet18`, and `densenet121`.
- `frozen`: when `True`, no model parameters are trained and the model stays in
  eval mode during the epoch.
- Sparse WTA/GSA settings live in `K_BASE_CONFIG`: `k_spatial`, `k_population`,
  `k_lifetime`, and `k_population_alpha`.

### Core Metrics

- `train/loss`: total training objective for the current logged batch.
- `train/acc`: batch training accuracy for classification runs.
- `train/mse`: batch reconstruction MSE for reconstruction-only runs.
- `test/loss`: validation loss from the bounded evaluation pass.
- `test/acc`: validation accuracy for classification runs.
- `test/mse`: validation MSE for reconstruction-only runs.

### Loss Breakdown

- `losses/ce`: classification cross entropy.
- `losses/final_layer_ce`: same value as `losses/ce`.
- `losses/mse`: total reconstruction MSE tracked by the active criterion.
- `losses/reconstruction_mse/{layer}`: per-layer reconstruction MSE when local
  reconstruction outputs are active.
- `losses/autoencoder_mse/{layer}`: same value as `losses/reconstruction_mse/{layer}`.
- `losses/correlation_total`: total redundancy/correlation penalty when that
  loss is active.
- `losses/{layer}`: per-layer redundancy/correlation penalty when that loss is active.

### Model Diagnostics

- `weight-mean/{model-type}`: image line plot of `get_weights()` in network
  order. Each point is one named weight tensor, reduced to its scalar mean
  across all dimensions. For example, `architecture_type = "resnet18"` logs to
  `weight-mean/resnet18`.

This image is logged once before training starts and once after every epoch.
For example, with `epochs = 1`, each run logs an initial image and an
`epoch_0` image. If `frozen = True`, the model is not trained, and those weight
mean plots should remain unchanged.

### Sparse Model State

- `general/last_k`: current active channel count for models that expose
  `last_k`, such as population-sparsity WTA autoencoders.
