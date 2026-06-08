#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/venv"
DEFAULT_LOCAL_MODEL="models/Qwen__Qwen3.6-27B"

MODEL_PATH="${1:-}"
TP_SIZE="${TP_SIZE:-2}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.94}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"

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
    echo "Local Qwen3.6 model not found at:" >&2
    echo "  ${DEFAULT_LOCAL_MODEL}" >&2
    echo "" >&2
    echo "Please download it first by running:" >&2
    echo "  ./download.sh --model Qwen/Qwen3.6-27B" >&2
    echo "" >&2
    echo "Then re-run:" >&2
    echo "  ./serve_qwen36.sh" >&2
    exit 1
  fi
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VLLM_RPC_TIMEOUT="${VLLM_RPC_TIMEOUT:-600}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

echo "============================================================"
echo " Qwen3.6 text-only vLLM launcher"
echo " Optimized for 4x A100 40GB"
echo " Default max_model_len: ${MAX_MODEL_LEN}"
echo " Model: ${MODEL_PATH}"
echo " TP_SIZE: ${TP_SIZE}"
echo " GPU_MEM_UTIL: ${GPU_MEM_UTIL}"
echo "============================================================"

exec vllm serve "${MODEL_PATH}" \
  --host "${HOST:-127.0.0.1}" \
  --port "${PORT:-8000}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  # --reasoning-parser qwen3 \
  --chat-template-kwargs '{"enable_thinking":false}'
  --language-model-only \
  --enable-prefix-caching \
  --disable-custom-all-reduce
