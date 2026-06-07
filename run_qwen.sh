#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:-}"
MODE="${MODE:-multimodal}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
DP_SIZE="${DP_SIZE:-8}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MM_ENCODER_TP_MODE="${MM_ENCODER_TP_MODE:-data}"
MM_PROCESSOR_CACHE_TYPE="${MM_PROCESSOR_CACHE_TYPE:-shm}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
ENABLE_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL:-1}"
ENABLE_TOOLS="${ENABLE_TOOLS:-1}"
DISABLE_THINKING="${DISABLE_THINKING:-0}"

if [[ -z "$MODEL_PATH" ]]; then
  echo "Usage: $0 <local-model-path>"
  exit 1
fi

if [[ -f "venv/bin/activate" ]]; then
  source venv/bin/activate
elif [[ -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
fi

export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

MODEL_PATH="$(realpath "$MODEL_PATH")"

ARGS=(
  --model "$MODEL_PATH"
  --host "$HOST"
  --port "$PORT"
  -dp "$DP_SIZE"
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  --max-model-len "$MAX_MODEL_LEN"
)

if [[ "$ENABLE_PREFIX_CACHING" == "1" ]]; then
  ARGS+=(--enable-prefix-caching)
fi

if [[ "$ENABLE_EXPERT_PARALLEL" == "1" ]]; then
  ARGS+=(--enable-expert-parallel)
fi

case "$MODE" in
  multimodal|mm)
    ARGS+=(--reasoning-parser qwen3)
    ARGS+=(--mm-encoder-tp-mode "$MM_ENCODER_TP_MODE")
    ARGS+=(--mm-processor-cache-type "$MM_PROCESSOR_CACHE_TYPE")
    ;;
  text|text-only)
    ARGS+=(--language-model-only)
    ;;
  *)
    echo "Invalid MODE: $MODE"
    exit 1
    ;;
esac

if [[ "$ENABLE_TOOLS" == "1" ]]; then
  ARGS+=(--enable-auto-tool-choice --tool-call-parser qwen3_coder)
fi

if [[ "$DISABLE_THINKING" == "1" ]]; then
  ARGS+=(--default-chat-template-kwargs '{"enable_thinking": false}')
fi

python -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
