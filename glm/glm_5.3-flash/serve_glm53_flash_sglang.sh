#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/../../venv"

MODEL_PATH=""
DEFAULT_MODEL="${ROOT_DIR}/models/unsloth__GLM-5.3-Flash-FP8"

SERVE_HOST="${HOST:-0.0.0.0}"
SERVE_PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-8}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-}"          # SGLang auto-derives if unset
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-32}"

# H200 FP8: SGLang's default attention/KV path for GLM-5-family FP8 checkpoints.
# These checkpoints do not ship pre-calibrated KV-cache scales, so FP8 KV cache
# is NOT enabled by default (SGLang docs warn of accuracy loss on reasoning
# tasks if you force --kv-cache-dtype fp8_e4m3 manually). Leave at auto/bf16
# unless you've validated accuracy for your workload.
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"

# Enable EAGLE speculative decoding (recommended for interactive/low-latency use).
ENABLE_SPECULATIVE="${ENABLE_SPECULATIVE:-1}"
SPEC_ALGO="${SPEC_ALGO:-EAGLE}"
SPEC_NUM_STEPS="${SPEC_NUM_STEPS:-3}"
SPEC_EAGLE_TOPK="${SPEC_EAGLE_TOPK:-1}"
SPEC_NUM_DRAFT_TOKENS="${SPEC_NUM_DRAFT_TOKENS:-4}"

# DP Attention trades low-concurrency latency for high-concurrency throughput.
# Disabled by default to match vLLM script's latency-oriented defaults.
ENABLE_DP_ATTENTION="${ENABLE_DP_ATTENTION:-0}"

CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

usage() {
  cat <<EOF
Usage: $(basename "${BASH_SOURCE[0]}") [MODEL_PATH] [OPTIONS]
Tested Architecture: 8x H200 - Full Tensor Parallel for GLM-5.3-Flash (FP8)

REQUIRES: SGLang (installed via 'uv pip install --prerelease=allow sglang')

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

# Verify SGLang is installed
SGLANG_VERSION=$(python3 -c "import sglang; print(sglang.__version__)" 2>/dev/null || true)
if [[ -z "${SGLANG_VERSION}" ]]; then
  echo "⚠️  Could not import sglang. Install it with:" >&2
  echo "    uv pip install --prerelease=allow sglang" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export FLASHINFER_DISABLE_VERSION_CHECK="${FLASHINFER_DISABLE_VERSION_CHECK:-1}"

echo "============================================================"
echo " GLM-5.3-Flash SGLang Launcher"
echo " Optimized for 8x H200 (FP8, day-0 SGLang support)"
echo " SGLang version:  ${SGLANG_VERSION}"
echo "============================================================"
echo " Model:              ${MODEL_PATH}"
echo " Host:                ${SERVE_HOST}"
echo " Port:                ${SERVE_PORT}"
echo " Tensor Parallel:     ${TP_SIZE}"
echo " DP Attention:        ${ENABLE_DP_ATTENTION}"
echo " Mem Fraction Static: ${MEM_FRACTION_STATIC}"
echo " Max Running Reqs:    ${MAX_RUNNING_REQUESTS}"
echo " KV Cache Dtype:      ${KV_CACHE_DTYPE}"
echo " Speculative:         ${SPEC_ALGO} (enabled=${ENABLE_SPECULATIVE}, steps=${SPEC_NUM_STEPS})"
echo " CUDA Devices:        ${CUDA_VISIBLE_DEVICES}"
echo "============================================================"

CMD=(
  sglang serve
  --model-path "${MODEL_PATH}"
  --served-model-name GLM-5.3-Flash
  --trust-remote-code
  --host "${SERVE_HOST}"
  --port "${SERVE_PORT}"
  --tp "${TP_SIZE}"
  --mem-fraction-static "${MEM_FRACTION_STATIC}"
  --max-running-requests "${MAX_RUNNING_REQUESTS}"
  --tool-call-parser glm47
  --reasoning-parser glm45
  --enable-flashinfer-allreduce-fusion
)

if [[ -n "${MAX_TOTAL_TOKENS}" ]]; then
  CMD+=(--context-length "${MAX_TOTAL_TOKENS}")
fi

if [[ "${KV_CACHE_DTYPE}" != "auto" ]]; then
  CMD+=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
fi

if [[ "${ENABLE_DP_ATTENTION}" == "1" ]]; then
  CMD+=(--dp "${TP_SIZE}" --enable-dp-attention)
fi

if [[ "${ENABLE_SPECULATIVE}" == "1" ]]; then
  CMD+=(
    --speculative-algorithm "${SPEC_ALGO}"
    --speculative-num-steps "${SPEC_NUM_STEPS}"
    --speculative-eagle-topk "${SPEC_EAGLE_TOPK}"
    --speculative-num-draft-tokens "${SPEC_NUM_DRAFT_TOKENS}"
  )
fi

echo "Starting SGLang server..."
echo "Command: ${CMD[*]}"

exec "${CMD[@]}"
