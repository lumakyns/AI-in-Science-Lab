# ToMe ViT on CIFAR-100

Run from the repository root with the `ai-science` Conda environment:

```bash
python -m src.tome.training --config src/tome/config.yaml
```

This is a CIFAR-scale ViT trained from scratch, using the algorithm described in
[Token Merging: Your ViT but Faster](https://github.com/facebookresearch/ToMe).
It uses 4x4 patches, width 192, nine transformer blocks, three attention heads,
four merges per block, weighted token averaging, a protected class token and
proportional attention. The 64 spatial patches reduce to 28 spatial tokens.
Default training is 100 epochs, batch size 128, AdamW, five warmup epochs and
cosine learning-rate decay. Configured seed, architecture, loader, training,
checkpoint and W&B options are all in `config.yaml`.

Prepare missing splits with `python data/download.py cifar100`. Dataset code is
owned by this experiment. Prepared train/validation/test splits are separate;
validation chooses the best checkpoint, and testing runs once after training.

Outputs are under `checkpoints/tome/runs/<UTC timestamp>/`:

- `images/patch_merges/epoch_NNN_sample_SS.png`: fixed validation images and
  every layer's groups. Matching colors and numbers within a panel identify
  original patches represented by one surviving token. Epoch zero is untrained.
- `images/patch_merges/epoch_NNN_groups.npz`: exact per-layer group IDs, original
  images, labels and predictions. IDs are local to each sample and layer.
- `images/epoch_NNN.png`: loss and accuracy curves after every epoch.
- `checkpoints/{last,best}.pt`: model, optimizer, scaler, RNG states and resolved
  config. Best is selected by validation accuracy; automatic resume is not implemented.
- `metrics.jsonl`, `batch_metrics.jsonl`, `config.yaml`, `wandb_url.txt`.
- `result.json` and `COMPLETE` appear only after training and testing finish.

W&B project is `tome`; mode defaults to online and errors do not silently switch
to offline. The run group and output directory share one startup UTC timestamp.
Patch figures, losses, accuracy, top-5 accuracy, learning rate, throughput and
peak GPU memory are logged. `latest_run.txt` points to the latest initialized run.
A detached launch uses `subprocess.Popen` with `start_new_session=True`,
`stdin=subprocess.DEVNULL`, and explicit environment interpreter, recording its
PID to `checkpoints/tome/study.pid` and output to `checkpoints/tome/study.log`.

Create a sweep with `wandb sweep --project tome src/tome/sweep.yaml`. Dotted
`key=value` arguments override the same YAML configuration (no separate defaults).
The sweep compares reductions including r=0 as an ordinary ViT baseline.
