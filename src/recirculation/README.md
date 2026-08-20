# Recirculation on MNIST

One circular dense autoencoder: **784 → 20 → 784 → 20 → 784**, with bias-free linear layers and no activation functions. It reconstructs unnormalized MNIST pixels in [0, 1]; labels are unused.

`learning.algorithm` selects regular backpropagation (`bp`) or approximate PR recirculation (`recirculation`). Both use SGD with the same settings. Recirculation assigns local outer-product gradients using the forward-propagated residual for hidden layers and the original residual for the output; it does not use autograd. For this linear, bias-free model, this is exactly the original-minus-recirculated-activity rule.

Log `ratio_A/B/C`, `cossim_A/B/C`, reconstruction `train/loss` and `train/mse`, epoch loss, learning rate, validation loss/MSE, and test loss/MSE. `logging.every_steps` controls reconstruction logging and `logging.diagnostics_every_steps` controls both computation and logging of the six signal diagnostics (default: every 100 updates, including update 0). Samples are direct values, never averages of the diagnostics. Diagnostic steps also log reconstruction error. Set both intervals to 1 to log every update. Zero-norm undefined diagnostics remain NaN.

Training retains batch size 1. Evaluation uses independent batches of 512 and one forward pass per batch; CPU execution uses one thread by default to reduce overhead for these small matrix operations. Set `training.cpu_threads` to tune this. All training examples still contribute to epoch loss, and full validation/test splits are evaluated.

```bash
python data/download.py mnist
python -m src.recirculation.training --config src/recirculation/config.yaml
python -m unittest src.recirculation.test_learning
wandb sweep src/recirculation/sweep.yaml
# Then launch: wandb agent <sweep-id>
```

Use the repository's `biai_env` environment (PyTorch, NumPy, PyYAML, W&B, and Matplotlib). `config.yaml` controls training, SGD, the optional step scheduler, data loading, checkpoints, plots, and W&B. Set `wandb.mode: offline` explicitly for offline runs. Set `training.max_batches` for a short smoke run.

Log half coordinate-summed squared error and per-pixel MSE on the prepared validation/test splits. Save periodic checkpoints, learning curves, reconstruction examples, metrics, and resolved config under `checkpoints/recirculation/`. Existing artifacts from the previous configurable implementation are historical and its checkpoints are incompatible with this simplified model.

The current sweep runs 12 trials: tied recirculation with learning rates 0.001, 0.0005, 0.0001, and 0.00001, each with seeds 0, 1, and 2. Both learning rules and independent weights remain available in config. Dotted W&B parameters override the base config; resolved configuration is validated and recorded. SGD settings, batch size, architecture, and logging frequency stay matched between methods. Larger training batches are a further speed option, but change the single-example experiment. Approximate learning can be less stable; a nonfinite loss or gradient fails the run explicitly.

`model.tied_weights: true` enforces W2 = W1^T and W4 = W3^T throughout training. Each pair shares one trainable matrix. BP accumulates gradients through both uses automatically; recirculation adds the encoder local gradient to the transposed decoder local gradient. The optimizer updates each shared matrix once. Pairwise tying alone does not guarantee all full-network signal ratios or cosines equal 1.

At each diagnostic step, `sign_agreement_A`, `sign_agreement_B`, and `sign_agreement_C` log the proportion of components for which sign(LHS) equals sign(RHS), for the same signal pairs as the ratios/cosines. Values range from 0 to 1, pooling batch and coordinate entries at that step without averaging across steps. Matching zeros count as agreement; zero versus nonzero counts as disagreement.
