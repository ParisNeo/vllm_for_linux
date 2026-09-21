#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/../../venv"

SERVE_HOST="${HOST:-127.0.0.1}"
SERVE_PORT="${PORT:-8001}"
MODEL_PATH=""
DEFAULT_MODEL="${ROOT_DIR}/models/Qwen__Qwen-Image-2.1"

GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"

# Opt-in: FP8 prefix KV cache ("fp8_v" quantizes V only ~41dB PSNR, "fp8" also K ~35dB).
# A quantized cache is not CUDA-graph capturable, so those requests fall back to eager decode.
PREFIX_KV_CACHE_DTYPE="${PREFIX_KV_CACHE_DTYPE:-}"

# Opt-in: step-level continuous batching. Admission is all-or-nothing per denoising wave.
ENABLE_STEP_EXECUTION="${ENABLE_STEP_EXECUTION:-0}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"

# Force eager decode (disables CUDA graph capture of fixed-shape decode steps).
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"

TARGET_GPU="${TARGET_GPU:-3}"
MIN_FREE_MB="${MIN_FREE_MB:-15360}"

usage() {
  cat <<EOF
Usage: $(basename "${BASH_SOURCE[0]}") [MODEL_PATH] [OPTIONS]
Tested Architecture: A100 (40GB) - Isolates on 1 single GPU

Serves unified text-to-image and image editing via vLLM-Omni (QwenImage21Pipeline).
Requires vLLM-Omni nightly with Qwen-Image-2.1 support (vllm-project/vllm-omni#7759).

Options:
  --host HOST       Host/interface (default: ${SERVE_HOST})
  --port PORT       Port to listen on (default: ${SERVE_PORT})
  --gpu GPU_ID      Physical GPU to isolate on (default: ${TARGET_GPU})
  -h, --help        Show this help message

Environment overrides: GPU_MEM_UTIL, MAX_MODEL_LEN, PREFIX_KV_CACHE_DTYPE,
ENABLE_STEP_EXECUTION, MAX_NUM_SEQS, ENFORCE_EAGER, TARGET_GPU, MIN_FREE_MB

Per-request contract (server defaults do NOT match this checkpoint):
  num_inference_steps: 40   (server default is 50)
  true_cfg_scale:      1.0  (server default is 4.0; only engages with a negative_prompt)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)    SERVE_HOST="$2"; shift 2 ;;
    --host=*)  SERVE_HOST="${1#*=}"; shift ;;
    --port)    SERVE_PORT="$2"; shift 2 ;;
    --port=*)  SERVE_PORT="${1#*=}"; shift ;;
    --gpu)     TARGET_GPU="$2"; shift 2 ;;
    --gpu=*)   TARGET_GPU="${1#*=}"; shift ;;
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

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Model path does not exist: ${MODEL_PATH}" >&2
  echo "Download it first from this directory:" >&2
  echo "  ./download_hf.sh Qwen/Qwen-Image-2.1" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${TARGET_GPU}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export FLASHINFER_DISABLE_VERSION_CHECK=1
export VLLM_RPC_TIMEOUT="${VLLM_RPC_TIMEOUT:-600}"

echo "============================================================"
echo " ▶️ vLLM-Omni Launcher: Qwen-Image-2.1"
echo " Target Arch: 1x A100 40GB (Isolating on GPU ${TARGET_GPU})"
echo " Model:       ${MODEL_PATH}"
echo " Endpoint:    ${SERVE_HOST}:${SERVE_PORT}"
echo " Mem Util:    ${GPU_MEM_UTIL} (BF16 peak ~34GB at 1024x1024/40 steps)"
echo "============================================================"
echo " ⚠️  Send per request: num_inference_steps=40, true_cfg_scale=1.0"
echo "    (server defaults 50 steps / CFG 4.0 do NOT match this checkpoint)"
echo "============================================================"

echo "[PRE-FLIGHT] Checking GPU memory availability on physical GPU ${TARGET_GPU}..."
GPU_MEM_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${TARGET_GPU}" 2>/dev/null | tr -d '[:space:]')
if [[ ! "${GPU_MEM_FREE}" =~ ^[0-9]+$ ]]; then
  echo "[PRE-FLIGHT][WARN] Could not query free memory for GPU ${TARGET_GPU}. Proceeding anyway."
elif [[ "${GPU_MEM_FREE}" -lt "${MIN_FREE_MB}" ]]; then
  echo "[PRE-FLIGHT][ERROR] GPU ${TARGET_GPU} has only ${GPU_MEM_FREE}MiB free."
  echo "                 At least ${MIN_FREE_MB}MiB is required for safe diffusion pipeline initialization."
  echo "                 Run 'nvidia-smi' to identify the process and kill it before retrying."
  exit 1
else
  echo "[PRE-FLIGHT][OK] GPU ${TARGET_GPU} has ${GPU_MEM_FREE}MiB free."
fi

echo "[PRE-FLIGHT] Clearing PyTorch distributed and CUDA cache to prevent fragmentation locks..."
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

EXEC_ARGS=(
  vllm serve "${MODEL_PATH}"
  --host "${SERVE_HOST}"
  --port "${SERVE_PORT}"
  --tensor-parallel-size 1
  --omni
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_MEM_UTIL}"
  --vae-use-slicing
  --vae-use-tiling
)

if [[ -n "${PREFIX_KV_CACHE_DTYPE}" ]]; then
  EXEC_ARGS+=(--stage-overrides "{\"0\":{\"extras\":{\"prefix_kv_cache_dtype\":\"${PREFIX_KV_CACHE_DTYPE}\"}}}")
fi

if [[ "${ENABLE_STEP_EXECUTION}" == "1" ]]; then
  EXEC_ARGS+=(--step-execution --max-num-seqs "${MAX_NUM_SEQS}")
fi

if [[ "${ENFORCE_EAGER}" == "1" ]]; then
  EXEC_ARGS+=(--enforce-eager)
fi

echo "Starting vLLM-Omni server..."
exec "${EXEC_ARGS[@]}"