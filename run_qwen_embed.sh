#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:-Qwen/Qwen3-Embedding-8B}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8005}"
GPU_ID="${GPU_ID:-4}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

echo "============================================================"
echo " Qwen embedding launcher"
echo " Optimized for GPU ${GPU_ID} on an 8x H100 setup"
echo " Model: ${MODEL_PATH}"
echo "============================================================"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VLLM_RPC_TIMEOUT="${VLLM_RPC_TIMEOUT:-300}"

exec vllm serve "$MODEL_PATH" \
  --runner pooling \
  --host "$HOST" \
  --port "$PORT" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --trust-remote-code
