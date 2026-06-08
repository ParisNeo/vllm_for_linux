#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/venv"
DEFAULT_LOCAL_MODEL="models/Qwen__Qwen3.5-122B-A10B-GPTQ-Int4"

MODEL_PATH="${1:-}"
TP_SIZE="${TP_SIZE:-4}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.60}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-128000}"
# Mamba architecture requires 1 cache block per sequence
# At 60% GPU utilization: ~161 blocks available, using 128 for safety margin
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"

# NCCL stability settings for multi-GPU communication
export NCCL_ALGO="Ring"
export NCCL_NET_GDR_LEVEL="2"
export NCCL_P2P_LEVEL="2"
export NCCL_SOCKET_IFNAME="lo"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

# Cleanup stale vLLM processes from previous crashes
echo "Cleaning up any stale vLLM processes..."
pkill -f "vllm serve" 2>/dev/null || true
pkill -f "EngineCore" 2>/dev/null || true
pkill -f "Worker_TP" 2>/dev/null || true
sleep 2

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
    echo "Local Qwen3.5 model not found at:" >&2
    echo "  ${DEFAULT_LOCAL_MODEL}" >&2
    echo "" >&2
    echo "Please download it first." >&2
    exit 1
  fi
fi

# Check if other GPU processes are running and warn
echo ""
echo "Checking for other GPU processes..."
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null || true
echo ""

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VLLM_RPC_TIMEOUT="${VLLM_RPC_TIMEOUT:-600}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

echo "============================================================"
echo " Qwen3.5 122B MoE vLLM launcher"
echo " Optimized for 4x A100 40GB"
echo " Default max_model_len: ${MAX_MODEL_LEN}"
echo " Model: ${MODEL_PATH}"
echo " TP_SIZE: ${TP_SIZE}"
echo " GPU_MEM_UTIL: ${GPU_MEM_UTIL}"
echo " MAX_NUM_SEQS: ${MAX_NUM_SEQS} (limited by Mamba cache blocks)"
echo "============================================================"
echo ""
echo "⚠️  NOTE: GPU_MEM_UTIL set to ${GPU_MEM_UTIL} for optimal KV cache allocation"
echo "   vLLM recommends minimum 0.56 when CUDA graph profiling is enabled."
echo "   If you encounter OOM errors, kill other GPU processes or reduce this value."
echo ""

# Check available GPU memory before launching
echo ""
echo "Checking GPU memory availability..."
python3 -c "
import torch
free_mem = []
for i in range(torch.cuda.device_count()):
    torch.cuda.set_device(i)
    free, total = torch.cuda.mem_get_info()
    free_gb = free / 1024**3
    total_gb = total / 1024**3
    free_mem.append(free_gb)
    print(f'  GPU {i}: {free_gb:.2f} GiB free / {total_gb:.2f} GiB total')

min_free = min(free_mem)
required = ${GPU_MEM_UTIL} * 40  # Approximate for A100 40GB
if min_free < required:
    print(f'')
    print(f'⚠️  WARNING: Minimum free memory ({min_free:.2f} GiB) is less than requested ({required:.2f} GiB)')
    print(f'   Consider reducing GPU_MEM_UTIL or killing other GPU processes.')
"
echo ""

exec vllm serve "${MODEL_PATH}" \
  --host "${HOST:-127.0.0.1}" \
  --port "${PORT:-8000}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --language-model-only \
  --enable-prefix-caching \
  --disable-custom-all-reduce \
  --distributed-executor-backend mp \
  --max-num-batched-tokens 8192 \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --scheduling-policy "fcfs"
