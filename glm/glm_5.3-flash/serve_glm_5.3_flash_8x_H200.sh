#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/../../venv"

MODEL_PATH=""
DEFAULT_MODEL="${ROOT_DIR}/models/unsloth__GLM-5.3-Flash-FP8"

SERVE_HOST="${HOST:-0.0.0.0}"
SERVE_PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-8}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-auto}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"

# H200 (Hopper) supports FP8 KV cache for GLM-5.3-Flash when using vLLM >= 0.29.0
# FP8 KV cache significantly reduces memory footprint and boosts throughput.
# If you encounter numerical instability or errors, fallback to bfloat16.
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
DTYPE="${DTYPE:-bfloat16}"

SPEC_METHOD="${SPEC_METHOD:-none}"
SPEC_NUM_TOKENS="${SPEC_NUM_TOKENS:-0}"

CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

usage() {
  cat <<EOF
Usage: $(basename "${BASH_SOURCE[0]}") [MODEL_PATH] [OPTIONS]
Tested Architecture: 8x H200 - Full Tensor Parallel for GLM-5.3-Flash

REQUIRES: vLLM >= 0.29.0

Options:
  --host HOST       Host/interface to bind to (default: ${SERVE_HOST})
  --port PORT       Port to listen on (default: ${SERVE_PORT})
  --model PATH      Path to the model (alternative to positional arg)
  -h, --help        Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)    SERVE_HOST="$2"; shift 2 ;;
    --host=*)  SERVE_HOST="${1#*=}"; shift ;;
    --port)    SERVE_PORT="$2"; shift 2 ;;
    --port=*)  SERVE_PORT="${1#*=}"; shift ;;
    --model)   MODEL_PATH="$2"; shift 2 ;;
    --model=*) MODEL_PATH="${1#*=}"; shift ;;
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

if [[ -f "${VENV_DIR}/bin/activate" ]]; then
  source "${VENV_DIR}/bin/activate"
else
  echo "Virtual environment not found at ${VENV_DIR}" >&2
  exit 1
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Model path does not exist: ${MODEL_PATH}" >&2
  exit 1
fi

# Verify vLLM version
VLLM_VERSION=$(vllm --version 2>/dev/null | grep -oP '\d+\.\d+\.\d+' | head -n 1)
if [[ -z "${VLLM_VERSION}" ]]; then
  echo "⚠️  Could not determine vLLM version. Ensure vLLM is installed in the virtual environment."
else
  # Simple check: must be 0.29.0 or higher
  if [[ "${VLLM_VERSION}" != "0.29."* && "${VLLM_VERSION}" != "0.3"* && "${VLLM_VERSION}" != "1."* ]]; then
    echo "⚠️  WARNING: vLLM version ${VLLM_VERSION} may not support GLM-5.3-Flash."
    echo "    Please upgrade to vLLM >= 0.29.0: pip install -U vllm"
    echo "    Continuing anyway, but startup may fail with weight loading errors."
  fi
fi

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VLLM_RPC_TIMEOUT="${VLLM_RPC_TIMEOUT:-600}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export FLASHINFER_DISABLE_VERSION_CHECK="${FLASHINFER_DISABLE_VERSION_CHECK:-1}"

# GLM-5.3 uses specific hybrid attention; disabling standard CUDA graph estimation
# may help if issues arise, but V1 engine handles it.
export VLLM_DISABLE_CUDA_GRAPH="${VLLM_DISABLE_CUDA_GRAPH:-0}"

echo "============================================================"
echo " GLM-5.3-Flash vLLM Launcher"
echo " Optimized for 8x H200 (verified configuration)"
echo " Environment: Script-relative 'models/' deployment"
echo "============================================================"
echo " Model:           ${MODEL_PATH}"
echo " Host:            ${SERVE_HOST}"
echo " Port:            ${SERVE_PORT}"
echo " Tensor Parallel: ${TP_SIZE}"
echo " Expert Parallel: ENABLED (--enable-expert-parallel)"
echo " GPU Memory Util: ${GPU_MEM_UTIL}"
echo " Max Model Len:   ${MAX_MODEL_LEN}"
echo " Max Num Seqs:    ${MAX_NUM_SEQS}"
echo " KV Cache Dtype:  ${KV_CACHE_DTYPE} (BF16 required for Hopper)"
echo " Model Dtype:     ${DTYPE}"
echo " Speculative:     ${SPEC_METHOD} (${SPEC_NUM_TOKENS} tokens)"
echo " CUDA Devices:    ${CUDA_VISIBLE_DEVICES}"
echo "============================================================"

echo "Starting vLLM server..."

exec vllm serve "${MODEL_PATH}" \
  --host "${SERVE_HOST}" \
  --port "${SERVE_PORT}" \
  --served-model-name GLM-5.3-Flash \
  --trust-remote-code \
  --dtype "${DTYPE}" \
  --kv-cache-dtype "${KV_CACHE_DTYPE}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --enable-expert-parallel \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --disable-uvicorn-access-log