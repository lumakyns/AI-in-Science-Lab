#!/usr/bin/env bash

set -Eeuo pipefail

ENVIRONMENT_NAME="${AI_SCIENCE_ENV_NAME:-ai-science}"
PYTHON_VERSION="${AI_SCIENCE_PYTHON_VERSION:-3.11}"
MINIFORGE_DIRECTORY="${AI_SCIENCE_MINIFORGE_DIR:-${HOME}/miniforge3}"
PYTORCH_INDEX_URL="${AI_SCIENCE_PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
export PIP_NO_CACHE_DIR=1

fail() {
  echo "setup error: $*" >&2
  exit 1
}

[[ "$(uname -s)" == "Linux" ]] || fail "this setup script supports Linux only"
[[ -f "src/separation/config.yaml" && -f "src/separation/sweep.yaml" && \
  -f "data/download.py" ]] || \
  fail "run this script from the AI-in-Science-Lab repository root"
command -v nvidia-smi >/dev/null 2>&1 || \
  fail "nvidia-smi was not found; install the NVIDIA driver before running setup"
nvidia-smi >/dev/null || fail "the NVIDIA driver is installed but no usable GPU was detected"

case "$(uname -m)" in
  x86_64) MINIFORGE_ARCH="x86_64" ;;
  aarch64|arm64) MINIFORGE_ARCH="aarch64" ;;
  *) fail "unsupported CPU architecture: $(uname -m)" ;;
esac

if command -v conda >/dev/null 2>&1; then
  CONDA_EXECUTABLE="$(command -v conda)"
elif [[ -x "${MINIFORGE_DIRECTORY}/bin/conda" ]]; then
  CONDA_EXECUTABLE="${MINIFORGE_DIRECTORY}/bin/conda"
else
  INSTALLER="$(mktemp --suffix=.sh)"
  trap 'rm -f "${INSTALLER:-}"' EXIT
  INSTALLER_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${MINIFORGE_ARCH}.sh"

  echo "Downloading Miniforge..."
  if command -v curl >/dev/null 2>&1; then
    curl --fail --location --retry 3 --output "${INSTALLER}" "${INSTALLER_URL}"
  elif command -v wget >/dev/null 2>&1; then
    wget --tries=3 --output-document="${INSTALLER}" "${INSTALLER_URL}"
  else
    fail "curl or wget is required to download Miniforge"
  fi

  echo "Installing Miniforge in ${MINIFORGE_DIRECTORY}..."
  bash "${INSTALLER}" -b -p "${MINIFORGE_DIRECTORY}"
  CONDA_EXECUTABLE="${MINIFORGE_DIRECTORY}/bin/conda"
  rm -f "${INSTALLER}"
  trap - EXIT
fi

# Load Conda in this non-interactive shell. `conda init` prepares future shells.
eval "$("${CONDA_EXECUTABLE}" shell.bash hook)"
conda init bash >/dev/null

clean_package_caches() {
  conda clean --all --yes >/dev/null 2>&1 || true
  PIP_NO_CACHE_DIR=false python -m pip cache purge >/dev/null 2>&1 || true
}

# CUDA wheels are large. Avoid keeping a second copy in package caches, and
# repeat this cleanup on all exits so an interrupted installation stays small.
echo "Clearing Conda and pip package caches to minimize disk usage..."
clean_package_caches
trap clean_package_caches EXIT

if conda env list | awk '{print $1}' | grep -Fxq "${ENVIRONMENT_NAME}"; then
  echo "Updating existing Conda environment: ${ENVIRONMENT_NAME}"
  conda install --name "${ENVIRONMENT_NAME}" --yes \
    "python=${PYTHON_VERSION}" pip
else
  echo "Creating Conda environment: ${ENVIRONMENT_NAME}"
  conda create --name "${ENVIRONMENT_NAME}" --yes \
    "python=${PYTHON_VERSION}" pip
fi

conda activate "${ENVIRONMENT_NAME}"
python -m pip install --upgrade pip

echo "Installing CUDA-enabled PyTorch from ${PYTORCH_INDEX_URL}..."
python -m pip install --no-cache-dir --upgrade torch torchvision \
  --index-url "${PYTORCH_INDEX_URL}"
python -m pip install --no-cache-dir --upgrade numpy pyyaml wandb

echo "Preparing MNIST and CIFAR-10..."
python data/download.py mnist cifar10

echo "Verifying the environment and NVIDIA GPU..."
python - <<'PY'
import torch
import torchvision
import yaml
import wandb

from src.separation.training import load_yaml

if not torch.cuda.is_available():
    raise SystemExit(
        "PyTorch installed, but CUDA is unavailable. Check the NVIDIA driver "
        "and AI_SCIENCE_PYTORCH_INDEX_URL."
    )

configuration = load_yaml("src/separation/config.yaml")
device_name = torch.cuda.get_device_name(0)
print(f"PyTorch: {torch.__version__}")
print(f"torchvision: {torchvision.__version__}")
print(f"CUDA runtime: {torch.version.cuda}")
print(f"GPU: {device_name}")
print(f"Configured dataset: {configuration['dataset']}")
print("Environment verification passed.")
PY

# torchvision retains downloaded archives and extracted source files even
# after data/download.py creates the .pt files used by this repository.
if [[ "${AI_SCIENCE_KEEP_RAW_DATA:-0}" != "1" ]]; then
  echo "Removing redundant raw dataset downloads..."
  rm -f data/cifar10/cifar-10-python.tar.gz
  rm -rf data/cifar10/cifar-10-batches-py
  rm -rf data/mnist/MNIST
fi

clean_package_caches
trap - EXIT

echo
echo "Setup complete. Open a new shell or run:"
echo "  conda activate ${ENVIRONMENT_NAME}"
echo "Then start training with:"
echo "  python -m src.separation.training --config src/separation/config.yaml"
echo "Or create a W&B sweep with:"
echo "  wandb sweep --project separation src/separation/sweep.yaml"
echo "  wandb agent <entity>/separation/<sweep-id>"
