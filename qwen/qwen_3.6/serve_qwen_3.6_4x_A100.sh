#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/../../venv"

SERVE_HOST="${HOST:-127.0.0.1}"
SERVE_PORT="${PORT:-8000}"
MODEL_PATH=""
DEFAULT_MODEL="${ROOT_DIR}/models/Qwen__Qwen3.6-27B"

usage() {
  cat <<EOF
Usage: $(basename "${BASH_SOURCE[0]}") [MODEL_PATH] [OPTIONS]
Tested Architecture: 2x A100 (40GB) - Frees GPUs 2 and 3 for other workloads

Options:
  --host HOST        Host/interface (default: ${SERVE_HOST})
  --port PORT        Port to listen on (default: ${SERVE_PORT})
  -h, --help         Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)   SERVE_HOST="$2"; shift 2 ;;
    --host=*) SERVE_HOST="${1#*=}"; shift ;;
    --port)   SERVE_PORT="$2"; shift 2 ;;
    --port=*) SERVE_PORT="${1#*=}"; shift ;;
    -h|--help) usage; exit 0 ;;
    -*)        echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
    *)
      if [[ -z "${MODEL_PATH}" ]]; then MODEL_PATH="$1"; else
        echo "Unexpected argument: $1" >&2; usage >&2; exit 1
      fi
      shift
      ;;
  esac
done

MODEL_PATH="${MODEL_PATH:-$DEFAULT_MODEL}"

if [[ -f "${VENV_DIR}/bin/activate" ]]; then source "${VENV_DIR}/bin/activate"; else
  echo "Virtual environment not found at ${VENV_DIR}" >&2; exit 1
fi

# Lock vLLM to GPUs 0 and 1 (leaving GPU 2 available and GPU 3 dedicated to Image Editing)
export CUDA_VISIBLE_DEVICES="0,1"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export FLASHINFER_DISABLE_VERSION_CHECK=1
export VLLM_RPC_TIMEOUT=600

echo "============================================================"
echo " ▶️ vLLM High-Throughput Launcher: Qwen 3.6"
echo " Target Arch: 2x A100 40GB (Tensor Parallel on GPUs 0,1)"
echo " Model:       ${MODEL_PATH}"
echo " Endpoint:    ${SERVE_HOST}:${SERVE_PORT}"
echo "============================================================"

echo "[PRE-FLIGHT] Checking GPU memory availability on physical GPUs 0,1..."
GPU_MEM_FREE_0=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | tr -d '[:space:]')
GPU_MEM_FREE_1=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1 | tr -d '[:space:]')

if [[ -z "${GPU_MEM_FREE_0}" || -z "${GPU_MEM_FREE_1}" ]]; then
  echo "[PRE-FLIGHT][WARN] Could not query free memory for GPUs 0,1. Proceeding anyway."
elif [[ "${GPU_MEM_FREE_0}" -lt 15360 || "${GPU_MEM_FREE_1}" -lt 15360 ]]; then
  echo "[PRE-FLIGHT][ERROR] Insufficient free memory on target GPUs."
  echo "                 GPU 0: ${GPU_MEM_FREE_0}MiB free / GPU 1: ${GPU_MEM_FREE_1}MiB free"
  echo "                 At least 15360MiB is required per GPU."
  exit 1
else
  echo "[PRE-FLIGHT][OK] GPU 0: ${GPU_MEM_FREE_0}MiB free | GPU 1: ${GPU_MEM_FREE_1}MiB free"
fi

echo "[PRE-FLIGHT] Clearing PyTorch distributed and CUDA cache..."
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

exec vllm serve "${MODEL_PATH}" \
  --host "${SERVE_HOST}" \
  --port "${SERVE_PORT}" \
  --tensor-parallel-size 2 \
  --max-model-len 32768 \
  --max-num-seqs 256 \
  --gpu-memory-utilization 0.93 \
  --kv-cache-dtype fp8 \
  --enable-chunked-prefill \
  --max-num-batched-tokens 4096 \
  --async-output-proc \
  --enable-prefix-caching \
  --trust-remote-code \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --limit-mm-per-prompt '{"image": 4}'
