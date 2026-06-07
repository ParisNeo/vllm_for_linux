#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/venv"
DEFAULT_LOCAL_MODEL="${ROOT_DIR}/models/Qwen__Qwen3.5-397B-A17B-GPTQ-Int4"
DOWNLOAD_SCRIPT="${ROOT_DIR}/download.sh"

MODEL_PATH="${1:-}"

if [[ -f "${VENV_DIR}/bin/activate" ]]; then
  source "${VENV_DIR}/bin/activate"
else
  echo "Virtual environment not found at ${VENV_DIR}" >&2
  echo "please run install.sh first" >&2
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
    echo "  chmod +x download.sh --model Qwen/Qwen3.5-397B-A17B-GPTQ-Int4" >&2
    echo "  ./download.sh" >&2
    echo "" >&2
    echo "Then re-run:" >&2
    echo "  ./serve_qwen.sh" >&2
    exit 1
  fi
fi

# export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# export VLLM_RPC_TIMEOUT="${VLLM_RPC_TIMEOUT:-600}"

echo "============================================================"
echo " Qwen3.5-397B-A17B vLLM launcher"
echo " Optimized for 8x H100 80GB"
echo " Default max_model_len: ${MAX_MODEL_LEN:-260000}"
echo " Profile: ${PROFILE:-text}"
echo " Model: ${MODEL_PATH}"
echo "============================================================"

COMMON_ARGS=(
  --host "${HOST:-127.0.0.1}"
  --port "${PORT:-8000}"
  --tensor-parallel-size "${TP_SIZE:-4}"
  --max-model-len "${MAX_MODEL_LEN:-260000}"
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.92}"
  --quantization moe_wna16
  --language-model-only
)
printf 'CMD: '
printf '%q ' vllm serve "${MODEL_PATH}" "${COMMON_ARGS[@]}"
echo
set -x
vllm serve "${MODEL_PATH}" "${COMMON_ARGS[@]}"
