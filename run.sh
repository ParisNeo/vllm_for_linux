#!/usr/bin/env bash
set -euo pipefail

MODEL_INPUT="${1:-}"
PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
TP_SIZE="${TP_SIZE:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-}"

if [[ -z "$MODEL_INPUT" ]]; then
  echo "Usage: $0 <model-path-or-hf-repo>"
  exit 1
fi

source venv/bin/activate

export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"

MODEL="$MODEL_INPUT"
EXTRA_ARGS=()

if [[ -d "$MODEL_INPUT" || -f "$MODEL_INPUT" ]]; then
  MODEL="$(realpath "$MODEL_INPUT")"
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
else
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
fi

if [[ -n "$SERVED_MODEL_NAME" ]]; then
  EXTRA_ARGS+=(--served-model-name "$SERVED_MODEL_NAME")
fi

if [[ -n "$MAX_MODEL_LEN" ]]; then
  EXTRA_ARGS+=(--max-model-len "$MAX_MODEL_LEN")
fi

MODEL_LC="$(echo "$MODEL" | tr '[:upper:]' '[:lower:]')"

case "$MODEL_LC" in
  *qwen3* )
    EXTRA_ARGS+=(--reasoning-parser qwen3)
    ;;

  *deepseek-r1*|*deepseek*r1* )
    EXTRA_ARGS+=(--reasoning-parser deepseek_r1)
    ;;

  *glm-4.5*|*glm45* )
    EXTRA_ARGS+=(--reasoning-parser glm45)
    ;;

  *gemma-4* )
    EXTRA_ARGS+=(--reasoning-parser gemma4)
    ;;

  *mistral*|*mistralai*|*ministral*|*magistral* )
    EXTRA_ARGS+=(
      --tokenizer_mode mistral
      --config_format mistral
      --load_format mistral
    )

    case "$MODEL_LC" in
      *reasoning*|*magistral*|*think* )
        EXTRA_ARGS+=(--reasoning-parser mistral)
        ;;
    esac
    ;;
esac

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --tensor-parallel-size "$TP_SIZE" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  "${EXTRA_ARGS[@]}"
