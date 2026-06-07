#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/venv"
DOWNLOAD_SCRIPT="${ROOT_DIR}/download.sh"
DEFAULT_LOCAL_MODEL="models/Qwen__Qwen-Image-Edit-2511"

MODEL_PATH="${1:-}"

if [[ -f "${VENV_DIR}/bin/activate" ]]; then
  source "${VENV_DIR}/bin/activate"
else
  echo "Virtual environment not found at ${VENV_DIR}" >&2
  echo "Please run install.sh first." >&2
  exit 1
fi

if [[ -z "${MODEL_PATH}" ]]; then
  if [[ -d "${DEFAULT_LOCAL_MODEL}" ]]; then
    MODEL_PATH="${DEFAULT_LOCAL_MODEL}"
    echo "No model supplied; using local model at: ${MODEL_PATH}"
  else
    if [[ -x "${DOWNLOAD_SCRIPT}" ]]; then
      echo "Local model not found. Running download.sh first..."
      "${DOWNLOAD_SCRIPT}" --model Qwen/Qwen-Image-Edit-2511
    else
      echo "Local model not found and download.sh is missing or not executable." >&2
      echo "Expected:" >&2
      echo "  ${DOWNLOAD_SCRIPT}" >&2
      echo "" >&2
      echo "Please run:" >&2
      echo "  chmod +x download.sh" >&2
      echo "  ./download.sh --model Qwen/Qwen-Image-Edit-2511" >&2
      exit 1
    fi

    if [[ -d "${DEFAULT_LOCAL_MODEL}" ]]; then
      MODEL_PATH="${DEFAULT_LOCAL_MODEL}"
    else
      echo "Download finished, but the model still was not found at:" >&2
      echo "  ${DEFAULT_LOCAL_MODEL}" >&2
      exit 1
    fi
  fi
fi

LAST_GPU_INDEX="${LAST_GPU_INDEX:-7}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${LAST_GPU_INDEX}}"

echo "============================================================"
echo " Qwen-Image-Edit-2511 launcher"
echo " Using last available GPU: ${CUDA_VISIBLE_DEVICES}"
echo " Model: ${MODEL_PATH}"
echo "============================================================"

exec vllm serve "${MODEL_PATH}" \
  --host "${HOST:-127.0.0.1}" \
  --port "${PORT:-8000}" \
  --tensor-parallel-size 1 \
  --trust-remote-code
