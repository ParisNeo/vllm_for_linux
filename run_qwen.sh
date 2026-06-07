#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:-}"
MODE="${MODE:-multimodal}"          # multimodal | text
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
DP_SIZE="${DP_SIZE:-8}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
ENABLE_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL:-1}"
ENABLE_TOOLS="${ENABLE_TOOLS:-0}"
DISABLE_THINKING="${DISABLE_THINKING:-0}"
MM_ENCODER_TP_MODE="${MM_ENCODER_TP_MODE:-data}"
MM_PROCESSOR_CACHE_TYPE="${MM_PROCESSOR_CACHE_TYPE:-shm}"
MM_PROCESSOR_KWARGS="${MM_PROCESSOR_KWARGS:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [[ -z "$MODEL_PATH" ]]; then
  echo "Usage: $0 <local-model-path-or-hf-model-id>"
  echo
  echo "Examples:"
  echo "  $0 /data/vllm_for_linux/models/Qwen__Qwen3.5-397B-A17B-FP8"
  echo "  MODE=multimodal DP_SIZE=8 $0 Qwen/Qwen3.5-397B-A17B-FP8"
  echo "  MODE=text DP_SIZE=8 $0 /data/vllm_for_linux/models/Qwen__Qwen3.5-397B-A17B-FP8"
  exit 1
fi

if [[ -f "venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source venv/bin/activate
elif [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

# If local model path, force offline mode.
if [[ -d "$MODEL_PATH" || -f "$MODEL_PATH" ]]; then
  MODEL_PATH="$(realpath "$MODEL_PATH")"
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
else
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
fi

ARGS=(
  --model "$MODEL_PATH"
  --host "$HOST"
  --port "$PORT"
  -dp "$DP_SIZE"
  --reasoning-parser qwen3
  --gpu-memory-utilization "$GPU_MEM_UTIL"
)

if [[ "$ENABLE_EXPERT_PARALLEL" == "1" ]]; then
  ARGS+=(--enable-expert-parallel)
fi

if [[ "$ENABLE_PREFIX_CACHING" == "1" ]]; then
  ARGS+=(--enable-prefix-caching)
fi

case "$MODE" in
  text|text-only|lm|language-model-only)
    ARGS+=(--language-model-only)
    ;;
  multimodal|mm)
    ARGS+=(--mm-encoder-tp-mode "$MM_ENCODER_TP_MODE")
    ARGS+=(--mm-processor-cache-type "$MM_PROCESSOR_CACHE_TYPE")
    if [[ -n "$MM_PROCESSOR_KWARGS" ]]; then
      ARGS+=(--mm-processor-kwargs "$MM_PROCESSOR_KWARGS")
    fi
    ;;
  *)
    echo "Invalid MODE: $MODE"
    echo "Allowed values: multimodal, text"
    exit 1
    ;;
esac

if [[ "$ENABLE_TOOLS" == "1" ]]; then
  ARGS+=(--enable-auto-tool-choice --tool-call-parser qwen3_coder)
fi

if [[ "$DISABLE_THINKING" == "1" ]]; then
  ARGS+=(--default-chat-template-kwargs '{"enable_thinking": false}')
fi

if [[ -n "$MAX_MODEL_LEN" ]]; then
  ARGS+=(--max-model-len "$MAX_MODEL_LEN")
fi

if [[ -n "$EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  EXTRA_SPLIT=( $EXTRA_ARGS )
  ARGS+=("${EXTRA_SPLIT[@]}")
fi

echo "Starting Qwen with configuration:"
echo "  model: $MODEL_PATH"
echo "  mode: $MODE"
echo "  host: $HOST"
echo "  port: $PORT"
echo "  dp_size: $DP_SIZE"
echo "  visible_gpus: $CUDA_VISIBLE_DEVICES"
echo "  offline: HF_HUB_OFFLINE=$HF_HUB_OFFLINE TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE"

echo "Command:"
printf 'python -m vllm.entrypoints.openai.api_server '
printf '%q ' "${ARGS[@]}"
echo

python -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
