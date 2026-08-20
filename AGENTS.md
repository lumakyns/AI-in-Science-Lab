# Research Repository Conventions

This repository is organized around independent experiments. Treat every
first-level directory under `src/` as an experiment and give each experiment
the same structure and execution contract.

## Required layout

```text
data/
  download.py
  <dataset>/.gitkeep
checkpoints/
  <experiment>/
src/
  <experiment>/
    README.md
    notes/
      experiments.txt
    __init__.py
    config.yaml
    datasets.py
    training.py
    models/
      __init__.py
      <model>.py
```

- Do not create a `latex/` directory for experiment notes. Keep a very short,
  technical description of the current experimental design in
  `src/<experiment>/notes/experiments.txt`; it must be no more than two short
  paragraphs and should be updated whenever the experiment design changes.

- Put reusable model definitions only in `src/<experiment>/models/`. Do not put
  models in notebooks or directly in the experiment directory.
- Keep one model family per file and export supported models from
  `models/__init__.py`.
- Keep dataset definitions and utilities in `src/<experiment>/datasets.py`.
  Dataset code belongs to the experiment, but is separate from its model
  definitions.
- Shared utilities directly under `src/` must be genuinely dataset-agnostic.
  Dataset enums, metadata, loaders, transforms, splits, and preprocessing are
  never shared between experiments.
- Use `data/download.py` to download and prepare missing datasets. Write
  checkpoints under `checkpoints/<experiment>/`. Do not commit downloaded
  datasets, checkpoints, W&B files, generated plots, or other run artifacts.
  Dataset directories contain only a tracked `.gitkeep` placeholder.

## Dataset contract

- Hardcode a string-valued `Dataset` enum in each experiment's
  `src/<experiment>/datasets.py`. It must have exactly one member for
  every first-level directory in `data/`; do not use a shared global enum.
- Keep every dataset utility used by an experiment in that experiment's own
  `datasets.py`: the enum, shape/class metadata, loaders, transforms,
  preprocessing, and split helpers. Do not place dataset logic in model files,
  `training.py`, a root `src/` utility, or a shared package.
- Never import dataset code from another experiment. Each experiment must own
  its complete dataset implementation, even when that requires intentional
  duplication. Cross-experiment dataset abstractions are prohibited.
- Whenever a dataset is added or removed, update `data/<dataset>/.gitkeep` and
  the `Dataset` enum in every experiment's `datasets.py` in the same change.
- Every model constructor must accept a keyword parameter named `dataset` with
  type `Dataset`, even when the architecture is dataset-independent. Do not
  accept a raw dataset string at the model boundary.
- Use `dataset` to select input channels, input dimensions, and output classes
  where necessary. Hardcode this metadata beside the local `Dataset` enum so
  models within an experiment do not duplicate dataset-specific constants.
- Reject an unsupported dataset or incompatible input shape early with a clear
  error; do not silently choose a default architecture.

## Training contract

- `src/<experiment>/training.py` is the sole training entry point for an
  experiment and must run from the repository root as
  `python -m src.<experiment>.training --config src/<experiment>/config.yaml`.
- The YAML config is the source of truth for dataset, model, optimizer,
  scheduler, seed, epoch count, batch size, data-loader settings, checkpoint
  policy, and W&B settings. CLI arguments may select a config file but should
  not grow into a second configuration system. Never put credentials in config.
- Keep W&B sweep definitions in a separate `src/<experiment>/sweep.yaml`, not
  nested inside `config.yaml`. The sweep file must be directly usable with
  `wandb sweep src/<experiment>/sweep.yaml`, specify the experiment's module
  training command, and override values from the experiment config using
  dotted parameter keys where needed.
- When practical, structure the config so it can also serve as the basis for a
  W&B sweep. Keep important experimental choices sweep-ready, such as dataset,
  model variant, seed, learning rates, optimizer settings, batch size, epoch or
  step counts, loss-specific hyperparameters, and W&B mode.
- Seed Python, NumPy, and the ML framework; select the compute device explicitly;
  and record the fully resolved config in W&B.
- Initialize W&B with a project name of
  `<experiment-folder>_<YYYYMMDDTHHMMSSZ>`, using a UTC timestamp created once
  at process startup. Use the same timestamp for all artifacts from that run.
- Log `train/loss`, epoch, optimizer learning rate, and validation loss when a
  validation split exists. Log task-appropriate quality metrics and useful
  model/data metadata as well. Use consistent namespaced metric keys.
- Fail loudly if required config, data, or W&B initialization is invalid. A
  deliberate offline W&B mode may be configured explicitly; do not silently
  disable tracking.

## Quality bar for agents

- Preserve this layout when creating or refactoring experiments; do not invent
  a one-off structure for a new folder under `src/`.
- Keep imports package-safe and avoid runtime `sys.path` manipulation.
- Before handing off a change, at least check that the experiment imports and
  its config can be loaded. When practical, run a short smoke check that does
  not require downloading a full dataset.
