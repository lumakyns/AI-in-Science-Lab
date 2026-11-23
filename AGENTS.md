# Research repository conventions

Each first-level directory under `src/` is an independent experiment.

## Layout and ownership

```text
data/download.py
data/<dataset>/.gitkeep
checkpoints/<experiment>/
src/<experiment>/
  __init__.py
  README.md
  config.yaml
  datasets.py
  training.py
  notes/experiments.txt
  models/
    __init__.py
    <model_family>.py
```

- Keep design notes in `notes/experiments.txt`: at most two short technical
  paragraphs, updated when the design changes. Do not create `latex/` notes.
- Put reusable models only in `models/`, one family per file, and export
  supported models from `models/__init__.py`. Keep imports package-safe;
  do not modify `sys.path` at runtime.
- Each experiment owns all its dataset logic in its own `datasets.py`:
  enums, metadata, loaders, transforms, preprocessing, and splits. Never
  import dataset code across experiments; duplicate it when needed.
  Shared utilities under `src/` must be dataset-agnostic.
- Hardcode a local string-valued `Dataset` enum with one member per dataset
  directory in `data/`. Adding/removing a dataset requires updating its
  `.gitkeep` and every experiment's enum in the same change.
- Every model constructor accepts keyword `dataset: Dataset`, not a raw
  string. Use metadata beside the local enum for channels, dimensions, and
  classes; reject unsupported datasets and incompatible shapes early.
- Prepare missing data through `data/download.py`. Only `.gitkeep` placeholders
  are tracked in dataset directories. Keep checkpoints, images, and other run
  outputs under `checkpoints/<experiment>/`; never commit data or run artifacts,
  including W&B files.

## Training and tracking

- The sole training entry point runs from the repository root:
  `python -m src.<experiment>.training --config src/<experiment>/config.yaml`.
- YAML is the source of truth for dataset, model, optimizer, scheduler, seeds,
  duration, batch size, loaders, checkpoints, and W&B settings. CLI selects
  the config; do not build a second configuration system or store credentials.
- Keep important experimental choices sweep-ready. Put sweep definitions in
  `src/<experiment>/sweep.yaml`, usable with `wandb sweep`, with the module
  training command and dotted parameter overrides where needed.
- Seed Python, NumPy, and the ML framework; select the device explicitly.
  Record the fully resolved config and useful model/data metadata in W&B.
- Create one UTC timestamp at process startup (`YYYYMMDDTHHMMSSZ`). Use a stable W&B
  project `<experiment-folder>`; use the timestamp for run artifacts and run grouping.
- Log `train/loss`, epoch, optimizer learning rate, validation loss when
  applicable, and task metrics such as accuracy with consistent namespaced
  keys. Save per-epoch figures in the run's `images/` directory.
- Fail clearly on invalid config, data, or W&B initialization. Offline tracking
  must be explicitly configured; never silently disable W&B.

## Environment and long runs

- Use a Conda environment named `ai-science` for future work. Activate it or
  use `conda run --no-capture-output -n ai-science ...`; install dependencies
  with that environment's `python -m pip`. `scripts/setup.sh` defaults to this
  name. Prefer a self-contained environment over dependent virtualenvs.
- Before GPU training, check `nvidia-smi`, `torch.version.cuda`, and
  `torch.cuda.is_available()`, then run a small forward/backward check on the
  actual device. Install a compatible CUDA-enabled PyTorch/torchvision pair
  when needed; a visible GPU does not imply the Python environment supports it.
  Apply any required library-path fixes to the launched process only.
- Verify experiment imports and config loading before handoff. When practical,
  run a small synthetic smoke test covering training, outputs, and tracking
  before launching a full study.
- Launch authorized long studies as detached processes (for example,
  `subprocess.Popen(..., start_new_session=True)` with stdin disconnected).
  Redirect stdout/stderr to `checkpoints/<experiment>/study.log` and record
  the PID in `study.pid`. Use an explicit environment interpreter or Conda run.
- Check that the process is alive, training progresses, artifacts appear, and
  W&B syncs. Report the config, output paths, W&B link, and current status;
  distinguish launched work from completed results.
- Preserve environments and dependencies used by running jobs. Do not migrate,
  remove, or modify them mid-run; use `ai-science` for subsequent jobs.
