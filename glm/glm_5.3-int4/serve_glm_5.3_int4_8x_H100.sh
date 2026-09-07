#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/../../venv"

MODEL_PATH=""
DEFAULT_MODEL="${ROOT_DIR}/models/Tech2wild__GLM-5.3-Int4-Int8Mix"

SERVE_HOST="${HOST:-127.0.0.1}"
SERVE_PORT="${PORT:-8000}"

TP_SIZE="${TP_SIZE:-8}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.97}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-524288}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"

KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
DTYPE="${DTYPE:-bfloat16}"
QUANTIZATION="${QUANTIZATION:-compressed-tensors}"

SPEC_METHOD="${SPEC_METHOD:-mtp}"
SPEC_NUM_TOKENS="${SPEC_NUM_TOKENS:-5}"

DCP_SIZE="${DCP_SIZE:-0}"

CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
MIN_FREE_MB="${MIN_FREE_MB:-55000}"

DISABLE_CUSTOM_ALL_REDUCE="${DISABLE_CUSTOM_ALL_REDUCE:-1}"
ENABLE_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL:-1}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"

usage() {
  cat <<EOF
Usage: $(basename "${BASH_SOURCE[0]}") [MODEL_PATH] [OPTIONS]

GLM-5.3 Int4-Int8Mix vLLM launcher for 8x H100.

Options:
  --host HOST              Host/interface to bind to
  --port PORT              Port to listen on
  --model PATH             Path to the model
  --max-model-len TOKENS   Maximum context length
  --max-num-seqs N         Maximum concurrent sequences
  --gpu-memory-utilization N
                           GPU memory fraction
  --kv-cache-dtype TYPE    KV cache dtype
  --max-num-batched-tokens N
                           Maximum batched tokens
  --dcp-size N             Decode context parallel size
  -h, --help               Show this help message

Examples:

  MAX_MODEL_LEN=262144 ./serve.sh

  MAX_MODEL_LEN=524288 MAX_NUM_SEQS=4 ./serve.sh

  ./serve.sh --max-model-len 524288 --max-num-seqs 4

Environment variables:

  MAX_MODEL_LEN
  MAX_NUM_SEQS
  MAX_NUM_BATCHED_TOKENS
  GPU_MEM_UTIL
  KV_CACHE_DTYPE
  DCP_SIZE
  SPEC_METHOD
  SPEC_NUM_TOKENS
  CUDA_VISIBLE_DEVICES
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
    --max-model-len)
      MAX_MODEL_LEN="$2"
      shift 2
      ;;
    --max-model-len=*)
      MAX_MODEL_LEN="${1#*=}"
      shift
      ;;
    --max-num-seqs)
      MAX_NUM_SEQS="$2"
      shift 2
      ;;
    --max-num-seqs=*)
      MAX_NUM_SEQS="${1#*=}"
      shift
      ;;
    --gpu-memory-utilization)
      GPU_MEM_UTIL="$2"
      shift 2
      ;;
    --gpu-memory-utilization=*)
      GPU_MEM_UTIL="${1#*=}"
      shift
      ;;
    --kv-cache-dtype)
      KV_CACHE_DTYPE="$2"
      shift 2
      ;;
    --kv-cache-dtype=*)
      KV_CACHE_DTYPE="${1#*=}"
      shift
      ;;
    --max-num-batched-tokens)
      MAX_NUM_BATCHED_TOKENS="$2"
      shift 2
      ;;
    --max-num-batched-tokens=*)
      MAX_NUM_BATCHED_TOKENS="${1#*=}"
      shift
      ;;
    --dcp-size)
      DCP_SIZE="$2"
      shift 2
      ;;
    --dcp-size=*)
      DCP_SIZE="${1#*=}"
      shift
      ;;
    -h|--help)
      usage
      exit 0
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

MODEL_PATH="${MODEL_PATH:-$DEFAULT_MODEL}"

if [[ -f "${VENV_DIR}/bin/activate" ]]; then
  source "${VENV_DIR}/bin/activate"
else
  echo "Virtual environment not found at ${VENV_DIR}" >&2
  exit 1
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Model path does not exist: ${MODEL_PATH}" >&2
  echo "Download it first from this directory:" >&2
  echo "  ./download_hf.sh tonyd2wild/GLM-5.3-Int4-Int8Mix" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export FLASHINFER_DISABLE_VERSION_CHECK="${FLASHINFER_DISABLE_VERSION_CHECK:-1}"
export VLLM_RPC_TIMEOUT="${VLLM_RPC_TIMEOUT:-600}"

echo "============================================================"
echo " GLM-5.3 Int4-Int8Mix vLLM Launcher"
echo " Long-context configuration"
echo "============================================================"
echo " Model:                 ${MODEL_PATH}"
echo " Host:                  ${SERVE_HOST}"
echo " Port:                  ${SERVE_PORT}"
echo " Tensor Parallel:       ${TP_SIZE}"
echo " Expert Parallel:       ${ENABLE_EXPERT_PARALLEL}"
echo " GPU Memory Util:       ${GPU_MEM_UTIL}"
echo " Max Model Len:         ${MAX_MODEL_LEN}"
echo " Max Num Seqs:           ${MAX_NUM_SEQS}"
echo " Max Batched Tokens:    ${MAX_NUM_BATCHED_TOKENS}"
echo " KV Cache Dtype:        ${KV_CACHE_DTYPE}"
echo " Model Dtype:           ${DTYPE}"
echo " Quantization:          ${QUANTIZATION}"
echo " Speculative Method:    ${SPEC_METHOD}"
echo " Speculative Tokens:    ${SPEC_NUM_TOKENS}"
echo " Decode CP:              ${DCP_SIZE}"
echo " CUDA Devices:           ${CUDA_VISIBLE_DEVICES}"
echo "============================================================"

IFS=',' read -ra GPU_ARRAY <<< "${CUDA_DEVICES}"

echo "[PRE-FLIGHT] Checking GPU memory availability..."

INSUFFICIENT_MEMORY=0

for GPU_ID in "${GPU_ARRAY[@]}"; do
  GPU_MEM_FREE="$(
    nvidia-smi \
      --query-gpu=memory.free \
      --format=csv,noheader,nounits \
      -i "${GPU_ID}" \
      2>/dev/null |
      tr -d '[:space:]'
  )"

  if [[ ! "${GPU_MEM_FREE}" =~ ^[0-9]+$ ]]; then
    echo "[PRE-FLIGHT][WARN] Could not query GPU ${GPU_ID}."
    continue
  fi

  if [[ "${GPU_MEM_FREE}" -lt "${MIN_FREE_MB}" ]]; then
    echo "[PRE-FLIGHT][ERROR] GPU ${GPU_ID}: ${GPU_MEM_FREE} MiB free."
    echo "[PRE-FLIGHT][ERROR] Required minimum: ${MIN_FREE_MB} MiB."
    INSUFFICIENT_MEMORY=1
  else
    echo "[PRE-FLIGHT][OK] GPU ${GPU_ID}: ${GPU_MEM_FREE} MiB free"
  fi
done

if [[ "${INSUFFICIENT_MEMORY}" -eq 1 ]]; then
  echo "[PRE-FLIGHT][ERROR] Insufficient free GPU memory."
  exit 1
fi

echo "[PRE-FLIGHT] Clearing CUDA cache..."

python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

echo "[PRE-FLIGHT] Starting vLLM..."

EXEC_ARGS=(
  vllm serve "${MODEL_PATH}"
  --host "${SERVE_HOST}"
  --port "${SERVE_PORT}"
  --served-model-name GLM-5.3
  --trust-remote-code
  --dtype "${DTYPE}"
  --quantization "${QUANTIZATION}"
  --kv-cache-dtype "${KV_CACHE_DTYPE}"
  --tensor-parallel-size "${TP_SIZE}"
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_MEM_UTIL}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
  --enable-auto-tool-choice
  --tool-call-parser glm47
  --reasoning-parser glm45
  --disable-uvicorn-access-log
)

if [[ -n "${SPEC_METHOD}" && "${SPEC_METHOD}" != "none" ]]; then
  EXEC_ARGS+=(
    --speculative-config.method "${SPEC_METHOD}"
    --speculative-config.num_speculative_tokens "${SPEC_NUM_TOKENS}"
  )
fi

if [[ "${ENABLE_EXPERT_PARALLEL}" == "1" ]]; then
  EXEC_ARGS+=(--enable-expert-parallel)
fi

if [[ "${DISABLE_CUSTOM_ALL_REDUCE}" == "1" ]]; then
  EXEC_ARGS+=(--disable-custom-all-reduce)
fi

if [[ "${ENFORCE_EAGER}" == "1" ]]; then
  EXEC_ARGS+=(--enforce-eager)
fi

if [[ -n "${DCP_SIZE}" && "${DCP_SIZE}" != "0" ]]; then
  EXEC_ARGS+=(--decode-context-parallel-size "${DCP_SIZE}")
fi

printf '[COMMAND]'
printf ' %q' "${EXEC_ARGS[@]}"
printf '\n'

exec "${EXEC_ARGS[@]}"
