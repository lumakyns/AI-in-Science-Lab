# Local Learning Notes

## W&B Visualization Reference

This section documents the W&B metrics and media panels emitted by `training.py` and
the logger modules. Keep this updated whenever visualization logging changes.

### Core Training

- `train/loss`: total training objective for the current logged batch. For VGG
  pretrain phases this includes classification loss plus reconstruction loss.
- `train/acc`: batch training accuracy for classification runs.
- `train/mse`: batch reconstruction MSE for reconstruction-only runs.
- `test/loss`: validation loss from the bounded evaluation pass.
- `test/acc`: validation accuracy for classification runs.
- `test/mse`: validation MSE for reconstruction-only runs.
- `losses/ce`: classification cross entropy.
- `losses/final_layer_ce`: same classification CE value, named to emphasize the
  final classifier supervision signal.
- `losses/mse`: total reconstruction MSE tracked by the active criterion.
- `losses/correlation_total`: total redundancy/correlation penalty when that
  loss is active.
- `losses/{layer}`: per-layer redundancy/correlation penalty when that loss is active.

### VGG Deconv Reconstruction

- `losses/reconstruction_mse/{layer}`: per-layer reconstruction MSE for each VGG
  deconv autoencoder.
- `losses/autoencoder_mse/{layer}`: same value as `losses/reconstruction_mse`;
  kept as a legacy/alternate name.
- `train-first-layer-reconstruction/vgg16__conv1_1__reconstruction`: image
  grid for the first VGG deconv. Rows are target image, reconstruction, and
  absolute error. Normalized CIFAR/ImageNet inputs are denormalized for display.
- `train-first-layer-reconstruction/greedy_stacked_autoencoder__layer0__reconstruction`:
  same target/reconstruction/error image grid for the first greedy stacked
  autoencoder layer.

### Activation Diagnostics

- `test-activation-channel-pair-scatter/{layer}`: sampled channel-pair
  scatterplot grid. Each point is one image; axes are mean activations for two
  sampled channels. Pairs are randomly chosen once per layer and reused for the
  entire training run. Used to inspect channel collapse or correlation.
- `test-activation-geometric-mean-distance/{layer}`: average distance of
  encoder channels from the layer channel centroid. This is only logged for
  encoder/conv feature maps, not reconstruction, pooling, or classifier outputs.
  Lower values can suggest channels are becoming more similar.

### Feature Map Distribution

These are compact scalar summaries per layer. For a feature map shaped like
`[image, channel, spatial...]`, we first compute each image/channel spatial mean,
then summarize across images and channels. Decoder/reconstruction feature maps
are not logged here.

- `train-feature-map-distribution/mean_mean/{layer}`: average, across channels,
  of per-image spatial mean activations.
- `train-feature-map-distribution/mean_stddev/{layer}`: channel spread of
  those per-image spatial means.
- `train-feature-map-distribution/stddev_mean/{layer}`: average, across
  channels, of image-to-image variation.
- `train-feature-map-distribution/stddev_stddev/{layer}`: channel spread of
  image-to-image variation.

### Weights And Gradients

- `test-filter-grid/{layer}_{conv|deconv}`: visual grid of the first filters
  in a conv/deconv layer.
- `train-encoder-weight-distribution/{stat}/{layer}`: compact scalar
  distribution summaries for encoder conv weights. `{stat}` is one of
  `mean_mean`, `mean_stddev`, `stddev_mean`, or `stddev_stddev`.
- `train-gradient-distribution/{stat}/{layer}`: same four scalar summaries
  for layer gradients.
- `train-filter-norm-kde-progress/{layer}`: image of KDE overlays showing how
  filter norm distributions move over training.

### Compute

- `test-flops/{layer}`: estimated per-layer FLOPs for one sample under the
  currently active forward mode. For scheduled VGG pretrain epochs, this includes
  decoder/deconv FLOPs; after the switch to full training, those decoder FLOPs
  disappear.
- `test-flops/total`: estimated total FLOPs for one sample under the active
  forward mode.
