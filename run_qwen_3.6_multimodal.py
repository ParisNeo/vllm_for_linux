#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
DEFAULT_LOCAL_MODEL="${ROOT_DIR}/models/Qwen__Qwen3.6-27B"

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
    echo "No model supplied and local model was not found at:" >&2
    echo "  ${DEFAULT_LOCAL_MODEL}" >&2
    echo "" >&2
    echo "Please download it first by running:" >&2
    echo "  chmod +x download.sh" >&2
    echo "  ./download.sh" >&2
    echo "" >&2
    echo "Then re-run:" >&2
    echo "  ./serve_qwen36_mm.sh" >&2
    exit 1
  fi
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VLLM_RPC_TIMEOUT="${VLLM_RPC_TIMEOUT:-600}"

echo "============================================================"
echo " Qwen3.6 multimodal vLLM launcher"
echo " Optimized for 8x H100 80GB"
echo " Default max_model_len: ${MAX_MODEL_LEN:-262144}"
echo " Model: ${MODEL_PATH}"
echo "============================================================"

exec vllm serve "${MODEL_PATH}" \
  --host "${HOST:-127.0.0.1}" \
  --port "${PORT:-8000}" \
  --tensor-parallel-size "${TP_SIZE:-4}" \
  --max-model-len "${MAX_MODEL_LEN:-262144}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.92}" \
  --reasoning-parser qwen3 \
  --mm-encoder-tp-mode data \
  --mm-processor-cache-type shm \
  --no-enable-prefix-caching
