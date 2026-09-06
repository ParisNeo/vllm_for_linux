#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/../../venv"

MODEL_PATH=""
DEFAULT_MODEL="${ROOT_DIR}/models/tonyd2wild__GLM-5.3-Int4-Int8Mix"

SERVE_HOST="${HOST:-0.0.0.0}"
SERVE_PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-8}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8_ds_mla}"
DTYPE="${DTYPE:-bfloat16}"
QUANTIZATION="${QUANTIZATION:-compressed-tensors}"

SPEC_METHOD="${SPEC_METHOD:-mtp}"
SPEC_NUM_TOKENS="${SPEC_NUM_TOKENS:-5}"
DCP_SIZE="${DCP_SIZE:-0}"

CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
MIN_FREE_MB="${MIN_FREE_MB:-55000}"

usage() {
  cat <<EOF
Usage: $(basename "${BASH_SOURCE[0]}") [MODEL_PATH] [OPTIONS]
Tested Architecture: 8x H200 - Full Tensor Parallel for GLM-5.3 Int4-Int8Mix

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
    --model)   MODEL_PATH="$2"; shift 2