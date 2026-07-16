#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/venv"
DEFAULT_LOCAL_MODEL="models/QuantTrio__GLM-5.2-Int4-Int8Mix"

# ===== CONFIGURABLE PARAMETERS =====
MODEL_PATH=""
SERVE_HOST="${HOST:-0.0.0.0}"
SERVE_PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-8}"                    # GLM-5.2 verified on TP=8
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"       # 0.90 as per model card
MAX_MODEL_LEN="${MAX_MODEL_LEN:-auto}"     # auto or specific value
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"         # As per model card
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"    # FP8 KV cache
DTYPE="${DTYPE:-bfloat16}"                 # Model dtype
QUANTIZATION="${QUANTIZATION:-compressed-tensors}"

# ===== SPECULATIVE DECODING (MTP) =====
# DISABLED - causes issues with CUDA graph capture on some systems
SPEC_METHOD="${SPEC_METHOD:-none}"
SPEC_NUM_TOKENS="${SPEC_NUM_TOKENS:-0}"

# ===== CUDA DEVICE SELECTION =====
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

# ===== ARGUMENT PARSING =====
# Supports: ./serve_glm52.sh [MODEL_PATH] [--host HOST] [--port PORT] [--model MODEL_PATH]
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
      echo "❌ Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
    *)
      if [[ -z "${MODEL_PATH}" ]]; then
        MODEL_PATH="$1"
      else
        echo "❌ Unexpected argument: $1" >&2
        usage >&2
        exit 1
      fi
      shift
      ;;
  esac
done

# ===== CUDA GRAPH SETTINGS =====
# Disable CUDA graphs via environment variable (compatible with vLLM 0.22.x and 0.23.x)
VLLM_DISABLE_CUDA_GRAPH="${VLLM_DISABLE_CUDA_GRAPH:-1}"
VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS="${VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:-0}"

# ===== ACTIVATE VIRTUAL ENVIRONMENT =====
if [[ -f "${VENV_DIR}/bin/activate" ]]; then
  source "${VENV_DIR}/bin/activate"
else
  echo "❌ Virtual environment not found at ${VENV_DIR}" >&2
  echo "Please run install.sh first:" >&2
  echo "  python3.12 -m venv venv" >&2
  echo "  source venv/bin/activate" >&2
  echo "  pip install vllm==0.23.0 transformers==5.12.1" >&2
  exit 1
fi

# ===== RESOLVE MODEL PATH =====
if [[ -z "${MODEL_PATH}" ]]; then
  if [[ -d "${DEFAULT_LOCAL_MODEL}" ]]; then
    MODEL_PATH="${DEFAULT_LOCAL_MODEL}"
    echo "ℹ️  No model supplied; using local model at: ${MODEL_PATH}"
  else
    echo "❌ Local GLM-5.2 model not found at:" >&2
    echo "  ${DEFAULT_LOCAL_MODEL}" >&2
    echo "" >&2
    echo "Please download it first by running:" >&2
    echo "  python3 -c \"from huggingface_hub import snapshot_download; snapshot_download('QuantTrio/GLM-5.2-Int4-Int8Mix', cache_dir='models')\"" >&2
    echo "" >&2
    echo "Then re-run:" >&2
    echo "  ./serve_glm52.sh" >&2
    exit 1
  fi
fi

# ===== VERIFY MODEL DIRECTORY =====
if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "❌ Model path does not exist: ${MODEL_PATH}" >&2
  exit 1
fi

# ===== ENVIRONMENT VARIABLES =====
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VLLM_RPC_TIMEOUT="${VLLM_RPC_TIMEOUT:-600}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"  # Enable vLLM v1 engine for GLM-5.2

# ===== PRINT CONFIGURATION =====
echo "============================================================"
echo " GLM-5.2 vLLM Launcher"
echo " Optimized for 8x H200 (verified configuration)"
echo "============================================================"
echo " Model:           ${MODEL_PATH}"
echo " Host:            ${SERVE_HOST}"
echo " Port:            ${SERVE_PORT}"
echo " Tensor Parallel: ${TP_SIZE}"
echo " Expert Parallel: ENABLED (--enable-expert-parallel)"
echo " GPU Memory Util: ${GPU_MEM_UTIL}"
echo " Max Model Len:   ${MAX_MODEL_LEN}"
echo " Max Num Seqs:    ${MAX_NUM_SEQS}"
echo " KV Cache Dtype:  ${KV_CACHE_DTYPE}"
echo " Model Dtype:     ${DTYPE}"
echo " Quantization:    ${QUANTIZATION}"
echo " Speculative:     ${SPEC_METHOD} (${SPEC_NUM_TOKENS} tokens)"
echo " CUDA Devices:    ${CUDA_VISIBLE_DEVICES}"
echo "============================================================"

# ===== LAUNCH VLLM SERVER =====
echo "🚀 Starting vLLM server..."
echo ""

exec vllm serve "${MODEL_PATH}" \
  --host "${SERVE_HOST}" \
  --port "${SERVE_PORT}" \
  --served-model-name GLM-5.2 \
  --trust-remote-code \
  --dtype "${DTYPE}" \
  --quantization "${QUANTIZATION}" \
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
