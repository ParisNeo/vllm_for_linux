#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/venv"
DEFAULT_LOCAL_MODEL="${ROOT_DIR}/models/Qwen__Qwen3-Embedding-8B"
DOWNLOAD_SCRIPT="${ROOT_DIR}/download.sh"

MODEL_PATH="${1:-}"

if [[ -f "${VENV_DIR}/bin/activate" ]]; then
  source "${VENV_DIR}/bin/activate"
else
  echo "Virtual environment not found at ${VENV_DIR}" >&2
  echo "Create it first:" >&2
  echo "  python -m venv .venv" >&2
  echo "  source .venv/bin/activate" >&2
  echo "  pip install -r requirements.txt" >&2
  exit 1
fi

if [[ -z "${MODEL_PATH}" ]]; then
  if [[ -d "${DEFAULT_LOCAL_MODEL}" ]]; then
    MODEL_PATH="${DEFAULT_LOCAL_MODEL}"
    echo "No model supplied; using local model at: ${MODEL_PATH}"
  else
    if [[ -x "${DOWNLOAD_SCRIPT}" ]]; then
      echo "No model supplied and local model not found."
      echo "Running download.sh to fetch the embedding model..."
      "${DOWNLOAD_SCRIPT}"
    else
      echo "No model supplied and local model was not found at:" >&2
      echo "  ${DEFAULT_LOCAL_MODEL}" >&2
      echo "" >&2
      echo "download.sh was not found or is not executable:" >&2
      echo "  ${DOWNLOAD_SCRIPT}" >&2
      echo "" >&2
      echo "Please create or fix download.sh, then run it first:" >&2
      echo "  chmod +x download.sh" >&2
      echo "  ./download.sh" >&2
      exit 1
    fi

    if [[ -d "${DEFAULT_LOCAL_MODEL}" ]]; then
      MODEL_PATH="${DEFAULT_LOCAL_MODEL}"
    else
      echo "Download finished, but the model directory is still missing:" >&2
      echo "  ${DEFAULT_LOCAL_MODEL}" >&2
      exit 1
    fi
  fi
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VLLM_RPC_TIMEOUT="${VLLM_RPC_TIMEOUT:-300}"

echo "============================================================"
echo " Qwen embedding launcher"
echo " Optimized for GPU ${CUDA_VISIBLE_DEVICES} on an 8x H100 setup"
echo " Model: ${MODEL_PATH}"
echo "============================================================"

exec vllm serve "${MODEL_PATH}" \
  --runner pooling \
  --host "${HOST:-127.0.0.1}" \
  --port "${PORT:-8005}" \
  --tensor-parallel-size "${TP_SIZE:-1}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.92}" \
  --max-model-len "${MAX_MODEL_LEN:-8192}" \
  --trust-remote-code
