#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/../../venv"

MODEL_PATH=""
DEFAULT_MODEL="${ROOT_DIR}/models/QuantTrio__GLM-5.2-Int4-Int8Mix"

SERVE_HOST="${HOST:-0.0.0.0}"
SERVE_PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-8}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-auto}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
DTYPE="${DTYPE:-bfloat16}"
QUANTIZATION="${QUANTIZATION:-compressed-tensors}"

SPEC_METHOD="${SPEC_METHOD:-none}"
SPEC_NUM_TOKENS="${SPEC_NUM_TOKENS:-0}"

CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

usage() {
  cat <<EOF
Usage: $(basename "${BASH_SOURCE[0]}") [MODEL_PATH] [OPTIONS]
Tested Architecture: 8x H200 - Full Tensor Parallel for GLM-5.2

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

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VLLM_RPC_TIMEOUT="${VLLM_RPC_TIMEOUT:-600}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export FLASHINFER_DISABLE_VERSION_CHECK="${FLASHINFER_DISABLE_VERSION_CHECK:-1}"

# In vLLM 0.28.x, the custom_all_reduce kernel fails during CUDA graph capture with 'invalid argument'.
# We must disable custom all-reduce (falling back to NCCL) and force eager mode to bypass the crash.
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
DISABLE_CUSTOM_ALL_REDUCE="${DISABLE_CUSTOM_ALL_REDUCE:-1}"

echo "============================================================"
echo " GLM-5.2 vLLM Launcher"
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
echo " KV Cache Dtype:  ${KV_CACHE_DTYPE}"
echo " Model Dtype:     ${DTYPE}"
echo " Quantization:    ${QUANTIZATION}"
echo " Speculative:     ${SPEC_METHOD} (${SPEC_NUM_TOKENS} tokens)"
echo " CUDA Devices:    ${CUDA_VISIBLE_DEVICES}"
echo "============================================================"

IFS=',' read -ra GPU_ARRAY <<< "${CUDA_DEVICES}"
echo "[PRE-FLIGHT] Checking GPU memory availability on physical GPUs ${CUDA_DEVICES}..."
INSUFFICIENT_MEMORY=0
for GPU_ID in "${GPU_ARRAY[@]}"; do
  GPU_MEM_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${GPU_ID}" 2>/dev/null | tr -d '[:space:]')
  if [[ -z "${GPU_MEM_FREE}" ]]; then
    echo "[PRE-FLIGHT][WARN] Could not query free memory for GPU ${GPU_ID}. Proceeding anyway."
  elif [[ "${GPU_MEM_FREE}" -lt 15360 ]]; then
    echo "[PRE-FLIGHT][ERROR] GPU ${GPU_ID} has only ${GPU_MEM_FREE}MiB free (15360MiB required)."
    INSUFFICIENT_MEMORY=1
  else
    echo "[PRE-FLIGHT][OK] GPU ${GPU_ID}: ${GPU_MEM_FREE}MiB free"
  fi
done

if [[ "${INSUFFICIENT_MEMORY}" -eq 1 ]]; then
  echo "[PRE-FLIGHT][ERROR] Insufficient free memory on one or more target GPUs."
  echo "                 Run 'nvidia-smi' to identify the process and kill it before retrying."
  exit 1
fi

echo "[PRE-FLIGHT] Clearing PyTorch distributed and CUDA cache..."
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

echo "Starting vLLM server..."

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
  --disable-uvicorn-access-log \
  $(if [[ "${DISABLE_CUSTOM_ALL_REDUCE}" == "1" ]]; then echo "--disable-custom-all-reduce"; fi) \
  $(if [[ "${ENFORCE_EAGER}" == "1" ]]; then echo "--enforce-eager"; fi)