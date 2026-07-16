#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/venv"
DEFAULT_LOCAL_MODEL="models/Qwen__Qwen3.6-27B"

MODEL_PATH=""
SERVE_HOST="${HOST:-127.0.0.1}"
SERVE_PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-2}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.94}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-128000}"

# ===== ARGUMENT PARSING =====
# Supports: ./serve_qwen36.sh [MODEL_PATH] [--host HOST] [--port PORT] [--model MODEL_PATH]
usage() {
  cat <<EOF
Usage: $(basename "${BASH_SOURCE[0]}") [MODEL_PATH] [OPTIONS]

Options:
  --host HOST       Host/interface to bind to (default: ${SERVE_HOST})
  --port PORT       Port to listen on (default: ${SERVE_PORT})
  --model PATH      Path to the model (alternative to positional arg)
  -h, --help        Show this help message

Environment variables HOST and PORT are also honored as defaults.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      SERVE_HOST="$2"
      shift 2
      ;;
    --host=*)
      SERVE_HOST="${1#*=}"
      shift
      ;;
    --port)
      SERVE_PORT="$2"
      shift 2
      ;;
    --port=*)
      SERVE_PORT="${1#*=}"
      shift
      ;;
    --model)
      MODEL_PATH="$2"
      shift 2
      ;;
    --model=*)
      MODEL_PATH="${1#*=}"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
    *)
      if [[ -z "${MODEL_PATH}" ]]; then
        MODEL_PATH="$1"
      else
        echo "Unexpected argument: $1" >&2
        usage >&2
        exit 1
      fi
      shift
      ;;
  esac
done

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
    echo "Local Qwen3.6 model not found at:" >&2
    echo "  ${DEFAULT_LOCAL_MODEL}" >&2
    echo "" >&2
    echo "Please download it first by running:" >&2
    echo "  ./download.sh --model Qwen/Qwen3.6-27B" >&2
    echo "" >&2
    echo "Then re-run:" >&2
    echo "  ./serve_qwen36.sh" >&2
    exit 1
  fi
fi
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VLLM_RPC_TIMEOUT="${VLLM_RPC_TIMEOUT:-600}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
echo "============================================================"
echo " Qwen3.6 text-only vLLM launcher"
echo " Optimized for 4x A100 40GB"
echo " Default max_model_len: ${MAX_MODEL_LEN}"
echo " Model: ${MODEL_PATH}"
echo " Host: ${SERVE_HOST}"
echo " Port: ${SERVE_PORT}"
echo " TP_SIZE: ${TP_SIZE}"
echo " GPU_MEM_UTIL: ${GPU_MEM_UTIL}"
echo "============================================================"
exec vllm serve "${MODEL_PATH}" \
  --host "${SERVE_HOST}" \
  --port "${SERVE_PORT}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --chat-template-kwargs '{"enable_thinking":false}' \
  --language-model-only \
  --enable-prefix-caching \
  --disable-custom-all-reduce
