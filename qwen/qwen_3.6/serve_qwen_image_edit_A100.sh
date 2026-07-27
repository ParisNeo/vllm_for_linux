#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/../../venv"

SERVE_HOST="${HOST:-127.0.0.1}"
SERVE_PORT="${PORT:-8001}"
MODEL_PATH=""
DEFAULT_MODEL="${ROOT_DIR}/models/Qwen__Qwen-Image-Edit-2511"

usage() {
  cat <<EOF
Usage: $(basename "${BASH_SOURCE[0]}") [MODEL_PATH] [OPTIONS]
Tested Architecture: A100 (40GB) - Isolates on 1 single GPU

Options:
  --host HOST       Host/interface (default: ${SERVE_HOST})
  --port PORT       Port to listen on (default: ${SERVE_PORT})
  -h, --help        Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)    SERVE_HOST="$2"; shift 2 ;;
    --host=*)  SERVE_HOST="${1#*=}"; shift ;;
    --port)    SERVE_PORT="$2"; shift 2 ;;
    --port=*)  SERVE_PORT="${1#*=}"; shift ;;
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

export CUDA_VISIBLE_DEVICES="3"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export FLASHINFER_DISABLE_VERSION_CHECK=1
export VLLM_RPC_TIMEOUT=600

echo "============================================================"
echo " ▶️ vLLM Launcher: Qwen Image Editing"
echo " Target Arch: 1x A100 40GB (Isolating on GPU 3 alongside Qwen 3.6 TP)"
echo " Model:       ${MODEL_PATH}"
echo " Endpoint:    ${SERVE_HOST}:${SERVE_PORT}"
echo "============================================================"

echo "[PRE-FLIGHT] Checking GPU memory availability on physical GPU 3..."
GPU_MEM_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 3 | tr -d '[:space:]')
if [[ -z "${GPU_MEM_FREE}" ]]; then
  echo "[PRE-FLIGHT][WARN] Could not query free memory for GPU 3. Proceeding anyway."
elif [[ "${GPU_MEM_FREE}" -lt 15360 ]]; then
  echo "[PRE-FLIGHT][ERROR] GPU 3 has only ${GPU_MEM_FREE}MiB free."
  echo "                 At least 15360MiB is required for safe diffusion pipeline initialization."
  echo "                 Run 'nvidia-smi' to identify the process and kill it before retrying."
  exit 1
else
  echo "[PRE-FLIGHT][OK] GPU 3 has ${GPU_MEM_FREE}MiB free."
fi

echo "[PRE-FLIGHT] Clearing PyTorch distributed and CUDA cache to prevent fragmentation locks..."
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

exec vllm serve "${MODEL_PATH}" \
  --host "${SERVE_HOST}" \
  --port "${SERVE_PORT}" \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --omni \
  --diffusion-load-format diffusers \
  --diffusion-load-kwargs '{"dtype": "bfloat16"}' \
  --vae-use-slicing \
  --vae-use-tiling
