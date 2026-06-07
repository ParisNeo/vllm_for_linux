#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:-models/Qwen__Qwen3.5-397B-A17B-GPTQ-Int4}"
PROFILE="${PROFILE:-text}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-260000}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VLLM_RPC_TIMEOUT="${VLLM_RPC_TIMEOUT:-600}"

echo "============================================================"
echo " Qwen3.5-397B-A17B vLLM launcher"
echo " Optimized for 8x H100 80GB"
echo " Default max_model_len: ${MAX_MODEL_LEN}"
echo " Profile: ${PROFILE}"
echo " Model: ${MODEL_PATH}"
echo "============================================================"

COMMON_ARGS=(
  --host "$HOST"
  --port "$PORT"
  --tensor-parallel-size "$TP_SIZE"
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  --quantization moe_wna16
  --reasoning-parser qwen3
)

case "$PROFILE" in
  text)
    exec vllm serve "$MODEL_PATH" \
      "${COMMON_ARGS[@]}" \
      --language-model-only
    ;;
  multimodal)
    exec vllm serve "$MODEL_PATH" \
      "${COMMON_ARGS[@]}" \
      --no-enable-prefix-caching
    ;;
  *)
    echo "Unknown PROFILE: $PROFILE"
    echo "Use PROFILE=text or PROFILE=multimodal"
    exit 1
    ;;
esac
