# Repository conventions

## Layout

- `src/<experiment>/`: independent experiments containing `README.md`,
  `config.yaml`, `datasets.py`, `training.py`, `models/`, and
  `notes/experiments.txt`; add `sweep.yaml` for sweeps.
- `data/`: dataset placeholders and `download.py`. Track only `.gitkeep` in
  dataset directories; prepare missing data with `python data/download.py <dataset>`.
- `checkpoints/<experiment>/`: all generated weights, figures, logs, resolved
  configs, and W&B files. Never commit run artifacts.
- `archive/`: historical code, research notes, and disabled automation.
  `checkpoints/archive/` holds historical local run artifacts.
- `setup.sh`: environment setup, run from the repository root.

## Experiment code

- Each experiment owns all dataset code in `datasets.py`; never import dataset
  logic across experiments. Shared utilities must be dataset-agnostic.
- Define a local string-valued `Dataset` enum matching the directories in
  `data/`. Dataset additions/removals update placeholders and every experiment's
  enum together. Keep channels, dimensions, and classes beside the enum.
- Model constructors accept keyword `dataset: Dataset`. Reject unsupported
  datasets and incompatible shapes early.
- Put reusable model families in separate files under `models/`, exporting
  supported models from `models/__init__.py`. Use package-safe imports;
  never modify `sys.path` at runtime.
- Keep design notes to at most two short technical paragraphs in
  `notes/experiments.txt`, updating them when the design changes. No `latex/` notes.

## Training

- Run from the root:
  `python -m src.<experiment>.training --config src/<experiment>/config.yaml`.
- YAML controls data, models, optimization, seeds, duration, loaders,
  checkpoints, and tracking. CLI selects the config; dotted overrides may
  support sweeps. Do not add another configuration system or store credentials.
- Keep experimental choices sweep-ready. `sweep.yaml` must work with
  `wandb sweep` and invoke the module training command.
- Seed Python, NumPy, and the ML framework; explicitly select the device.
- Use one startup UTC timestamp (`YYYYMMDDTHHMMSSZ`) for run directories and
  W&B grouping, with stable W&B project `<experiment>`.
- Record resolved config and useful model/data metadata. Log `train/loss`,
  epoch, optimizer learning rate, validation loss when applicable, and
  namespaced task metrics. Save per-epoch figures in the run's `images/`.
- Keep W&B files inside the run directory. Fail clearly on invalid config,
  missing/invalid data, or W&B initialization failure. Offline tracking must
  be explicit; never silently disable tracking.

## Environment and execution

- Use the self-contained Conda environment `ai-science`, activated or through
  `conda run --no-capture-output -n ai-science ...`. Install dependencies with
  that environment's `python -m pip`.
- Before GPU training, check `nvidia-smi`, `torch.version.cuda`, and
  `torch.cuda.is_available()`, then verify a small forward/backward pass on the
  actual device. Use a compatible CUDA PyTorch/torchvision pair; any library-path
  fixes apply only to the launched process.
- Verify imports and config loading; where practical, smoke-test training,
  outputs, and tracking before a full study.
- Launch authorized long studies detached, with stdin disconnected and an
  explicit environment interpreter. Write stdout/stderr to
  `checkpoints/<experiment>/study.log` and the PID to `study.pid`.
- Verify progress, artifacts, and W&B sync. Report config, output paths,
  W&B link, and whether work is running or complete.
- Never change environments or dependencies used by running jobs.
