#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE}")" && pwd)"
VENV_DIR="${ROOT_DIR}/../../../../venv"

SERVE_HOST="${HOST:-127.0.0.1}"
SERVE_PORT="${PORT:-8001}"
MODEL_PATH=""
DEFAULT_MODEL="${ROOT_DIR}/models/Qwen__Qwen-Image-Edit-2511"

usage() {
  cat <<EOF
Usage: $(basename "${BASH_SOURCE}") [MODEL_PATH] [OPTIONS]
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
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

echo "============================================================"
echo " ▶️ vLLM Launcher: Qwen Image Editing"
echo " Target Arch: 1x A100 40GB (Isolating on GPU 3)"
echo " Model:       ${MODEL_PATH}"
echo " Endpoint:    ${SERVE_HOST}:${SERVE_PORT}"
echo "============================================================"

exec vllm serve "${MODEL_PATH}" \
  --host "${SERVE_HOST}" \
  --port "${SERVE_PORT}" \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.95 \
  --vae-use-tiling \
  --vae-use-slicing
