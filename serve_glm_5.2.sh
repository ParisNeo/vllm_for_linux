#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/venv"
DEFAULT_LOCAL_MODEL="models/QuantTrio__GLM-5.2-Int4-Int8Mix"

# ===== CONFIGURABLE PARAMETERS =====
MODEL_PATH="${1:-}"
TP_SIZE="${TP_SIZE:-8}"                    # GLM-5.2 verified on TP=8
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"       # Slightly higher for more KV cache
MAX_MODEL_LEN="${MAX_MODEL_LEN:-auto}"     # auto or specific value
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"         # As per model card

# ===== KV CACHE =====
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"    # FP8 KV cache
DTYPE="${DTYPE:-bfloat16}"                 # Model dtype
QUANTIZATION="${QUANTIZATION:-compressed-tensors}"

# ===== SPECULATIVE DECODING (MTP) =====
# Re-enabled for performance — the model card specifies MTP with 1 token
SPEC_METHOD="${SPEC_METHOD:-mtp}"
SPEC_NUM_TOKENS="${SPEC_NUM_TOKENS:-1}"

# ===== CUDA DEVICE SELECTION =====
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

# ===== PERFORMANCE TUNING =====
# Enable CUDA graphs for faster decode (now that vLLM 0.23.0 works correctly)
VLLM_DISABLE_CUDA_GRAPH="${VLLM_DISABLE_CUDA_GRAPH:-0}"
VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS="${VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:-1}"

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

# ===== BUILD SPECULATIVE ARGS =====
SPEC_ARGS=""
if [[ "${SPEC_METHOD}" != "none" && "${SPEC_NUM_TOKENS}" -gt 0 ]]; then
  SPEC_ARGS=""#"--speculative-config method=${SPEC_METHOD} num_speculative_tokens=${SPEC_NUM_TOKENS}"
  SPEC_DISPLAY=""#"${SPEC_METHOD} (${SPEC_NUM_TOKENS} tokens)"
else
  SPEC_DISPLAY="disabled"
fi

# ===== BUILD CUDA GRAPH ARGS =====
CUDA_GRAPH_STATUS="ENABLED"
if [[ "${VLLM_DISABLE_CUDA_GRAPH}" == "1" ]]; then
  CUDA_GRAPH_STATUS="DISABLED"
fi

# ===== PRINT CONFIGURATION =====
echo "============================================================"
echo " GLM-5.2 vLLM Launcher (Optimized)"
echo " Optimized for 8x H200 (verified configuration)"
echo "============================================================"
echo " Model:           ${MODEL_PATH}"
echo " Tensor Parallel: ${TP_SIZE}"
echo " Expert Parallel: ENABLED (--enable-expert-parallel)"
echo " GPU Memory Util: ${GPU_MEM_UTIL}"
echo " Max Model Len:   ${MAX_MODEL_LEN}"
echo " Max Num Seqs:    ${MAX_NUM_SEQS}"
echo " KV Cache Dtype:  ${KV_CACHE_DTYPE}"
echo " Model Dtype:     ${DTYPE}"
echo " Quantization:    ${QUANTIZATION}"
echo " Speculative:     ${SPEC_DISPLAY}"
echo " CUDA Graphs:     ${CUDA_GRAPH_STATUS}"
echo " Prefix Caching:  ENABLED"
echo " Generation Config: vllm (prevents temp=1.0 override)"
echo " CUDA Devices:    ${CUDA_VISIBLE_DEVICES}"
echo "============================================================"

# ===== LAUNCH VLLM SERVER =====
echo "🚀 Starting vLLM server..."
echo ""

exec vllm serve "${MODEL_PATH}" \
  --host "${HOST:-0.0.0.0}" \
  --port "${PORT:-8000}" \
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
  --generation-config vllm \
  --enable-prefix-caching \
  ${SPEC_ARGS} \
  --disable-uvicorn-access-log
