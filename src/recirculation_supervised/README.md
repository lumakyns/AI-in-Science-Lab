# Does a calibrated recurrence train a classifier?

**Question:** Can an output-to-input recurrence deliver useful supervised credit assignment, and does useful error geometry explain its behavior relative to BP?

Train MNIST labels, not pixels. Every condition uses the same **784 → 128 → 64 → 32 → 10** bias-free linear classifier, half coordinate-summed squared loss against one-hot labels, batch size 1, and SGD. Keeping the network linear first isolates the feedback mechanism from activation derivatives. Its classification capacity is that of a factorized linear classifier; this is a mechanism experiment rather than an accuracy benchmark.

## Where the classifier is

The classifier is [`RecirculationClassifier` in `models/linear.py`](models/linear.py). Its four learned linear transformations form the entire prediction network; the last transformation, **32 → 10**, is the classification head:

```text
MNIST image x (784) → W1 → h1 (128) → W2 → h2 (64) → W3 → h3 (32) → W4 → y (10)
```

Using column vectors, the prediction is `y = W4 W3 W2 W1 x`. Coordinate `y[i]` is the real-valued score for digit `i`; the predicted digit is `argmax(y)`. The target `t` is a 10-dimensional one-hot label. All methods minimize `loss = 0.5 * ||y - t||²`.

The feedback bridge **R is a separate training-time map**, not the classifier. It maps the output residual `e = t - y` from 10 dimensions back to a 784-dimensional input teaching perturbation. Passing this perturbation through W1, W2, and W3 supplies approximate hidden-layer teaching signals. W4 always receives the exact output-layer gradient.

For `calibrated`, 16 input probes measure the current classifier's output responses, and a damped least-squares solve constructs R to approximately translate desired output corrections into input perturbations. For `scrambled`, we use `R O e` instead of `R e`, with fixed orthogonal O: the same feedback spectrum, but rotated error coordinates. Ordinary prediction always uses the original image and the forward classifier alone.

| Condition | How hidden layers learn |
|---|---|
| `bp` | Exact backpropagation. |
| `random` | Forward-propagate the output residual through one fixed random output-to-input bridge and the existing forward weights. |
| `calibrated` | Calibrate that bridge from measured forward responses, then use the same local updates. |
| `scrambled` | Calibrate identically, then apply a fixed orthogonal rotation to output-error coordinates. At identical forward weights this preserves the bridge's rank, norm, and singular values but changes its directional meaning. |

The scrambled condition tests **whether rank alone explains success**. Calibration improves the error-to-output loop; it does not optimize rank. Forward weights always learn the supervised classification objective. Initialization and shuffled-example order match across conditions at a given seed. Feedback randomness uses its own seeded generator.

For column-vector notation, take 16 orthonormal input probes V, measure Y=FV by evaluating the current linear classifier on the probe directions, and construct R=V Y^T(YY^T+lambda I)^-1. The identity loop is approached on well-conditioned output modes. With rows/batches, inject `error @ R.T` and propagate it through the hidden layers. Local gradients use original cached presynaptic activities; the final layer uses the exact classification residual. No backward pass is used by the three approximate update modes. The audit computes analytical BP signals only for measurement.

Every 2,000 updates and at each epoch end, freeze the weights and evaluate 256 held-out examples for diagnostics. Log per-layer numerical rank (relative threshold 1e-6), singular-value entropy effective rank, cosine, BP/recirculation norm ratio, and sign agreement, plus hidden-only aggregate parameter-gradient cosine and inner product. These are matrices across examples; a single batch-size-1 vector cannot reveal error rank. The `bp_*` statistics describe exact BP signals and `used_*` statistics describe the signals used by the selected training rule; in the BP baseline these coincide. Bridge-only statistics in the BP baseline refer to an unused fixed-random reference bridge.

Classification loss/MSE/accuracy log every 100 updates. Validation loss/accuracy log every epoch; test metrics log at completion. Probe singular values and loop minimum symmetric eigenvalue log at calibration. Elapsed time and probe-vector counts expose calibration cost; all methods incur the same audit schedule. Also report compute to reach a fixed validation loss, not just final accuracy. Audits never update weights or choose hyperparameters from the test split.

**Decision:** if calibrated recurrence improves held-out loss/accuracy over random and rank-matched scrambled recurrence across the learning-rate grid, with better true parameter-gradient alignment, the recurrence geometry matters beyond rank alone. If scrambled performs similarly, the proposed explanation fails. If calibration restores rank but not accuracy, investigate conditioning, alignment, and drift instead of calling it a rank success. Scrambling preserves spectra at matched weights; trajectories subsequently differ, so periodic paired frozen-checkpoint comparisons remain the clean causal control.

Run from the repository root using its ML environment:

```bash
python data/download.py mnist
python -m src.recirculation_supervised.training --config src/recirculation_supervised/config.yaml
wandb sweep --project ai-science src/recirculation_supervised/sweep.yaml
# Then execute the wandb agent command printed above.
python -m unittest src.recirculation_supervised.test_learning
```

The first sweep has **24 runs**: four methods × two learning rates (0.001, 0.0001) × three seeds. Each run defaults to four epochs. W&B, PyTorch, NumPy, and PyYAML are required; use `wandb.mode: offline` explicitly for offline runs. `training.max_batches` supports smoke checks. Set logging/audit/refresh cadence in YAML. The constant scheduler is configured as `none`; checkpoints/config/metrics go under `checkpoints/recirculation_supervised/<UTC timestamp>_<run ID>/`.

The existing `src/recirculation/` reconstruction experiment and its sweep are unchanged. This experiment owns its own dataset code as required by the repository contract.

## Initial pilot (not the full sweep)

One seed (0), SGD learning rate 0.001, 2,000 single-image updates, and the full prepared validation split:

| Method | Validation accuracy | Validation loss |
|---|---:|---:|
| BP | 58.81% | 0.37677 |
| Random recurrence | 16.05% | 0.46555 |
| Calibrated recurrence | 72.25% | 0.28348 |
| Scrambled calibrated recurrence | Diverged at update 458 | Nonfinite training loss |

The completed methods all had bridge rank 10. Random recurrence actually had higher first-hidden-layer error effective rank (9.48) than calibrated recurrence (8.72) at the terminal frozen audit, while its validation accuracy was much worse. This pilot does not support a rank-only explanation. It motivates measuring orientation and conditioning together with rank. Calibrated recurrence's aggregate hidden-gradient cosine was 0.354, versus -0.069 for random recurrence at that audit.

These runs used explicit offline W&B tracking and left the existing reconstruction experiment untouched. Their resolved configs and metrics are under `checkpoints/recirculation_supervised/pilot/`; the failed control has no completed-epoch metrics. Per-run compute was approximately three seconds excluding startup/data preparation; tiny timings are descriptive, not a throughput benchmark. One seed, one learning rate, and 2,000 updates do not establish superiority to BP. Complete the matched seed/LR sweep before drawing that conclusion.
