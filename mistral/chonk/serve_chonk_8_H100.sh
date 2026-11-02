#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/../../venv"

SERVE_HOST="${HOST:-127.0.0.1}"
SERVE_PORT="${PORT:-8000}"
MODEL_PATH=""
DOWNLOAD_ONLY=false

# Official Hugging Face repository and local target directory
DEFAULT_MODEL="${ROOT_DIR}/models/mistralai__Mistral-Large-4-1T-A52B-NVFP4"
DEFAULT_REPO_ID="mistralai/Mistral-Large-4-1T-A52B-NVFP4"

usage() {
  cat <<EOF
Usage: $(basename "${BASH_SOURCE[0]}") [MODEL_PATH] [OPTIONS]
Target Architecture: 8x NVIDIA H100 (80GB) - Official NVFP4 MoE Inference

Options:
  --host HOST         Host/interface (default: ${SERVE_HOST})
  --port PORT         Port to listen on (default: ${SERVE_PORT})
  -d, --download-only Only download/materialize the model and exit without running vLLM
  -h, --help          Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)   SERVE_HOST="$2"; shift 2 ;;
    --host=*) SERVE_HOST="${1#*=}"; shift ;;
    --port)   SERVE_PORT="$2"; shift 2 ;;
    --port=*) SERVE_PORT="${1#*=}"; shift ;;
    -d|--download-only) DOWNLOAD_ONLY=true; shift ;;
    -h|--help) usage; exit 0 ;;
    -*)       echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
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
  echo "Virtual environment not found at ${VENV_DIR}" >&2; exit 1
fi

# Ensure model files exist and download via Hugging Face Hub if missing
ensure_model_local() {
  local model_path="$1"

  if [[ -f "${model_path}/config.json" ]] && [[ -f "${model_path}/params.json" ]]; then
    echo "[AUTO-DOWNLOAD][OK] Verified local model files at ${model_path}"
    return 0
  fi

  echo "[AUTO-DOWNLOAD] Model weights or configuration missing at '${model_path}'."
  echo "[AUTO-DOWNLOAD] Pulling fresh official release for ${DEFAULT_REPO_ID}..."
  mkdir -p "${model_path}"

  python3 - "${DEFAULT_REPO_ID}" "${model_path}" <<'PY'
import sys
from huggingface_hub import snapshot_download

repo_id, local_dir = sys.argv[1], sys.argv[2]
print(f"[*] Downloading '{repo_id}' into '{local_dir}'...")
snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    ignore_patterns=["*.msgpack", "*.h5"],
)
PY

  if [[ -f "${model_path}/config.json" ]]; then
    echo "[AUTO-DOWNLOAD][OK] Model successfully materialized at ${model_path}"
    return 0
  else
    echo "[AUTO-DOWNLOAD][ERROR] Failed to obtain valid model configuration at ${model_path}" >&2
    return 1
  fi
}

if ! ensure_model_local "${MODEL_PATH}"; then
  exit 1
fi

if [[ "${DOWNLOAD_ONLY}" == true ]]; then
  echo "[DOWNLOAD-ONLY] Model is fully verified at ${MODEL_PATH}. Exiting."
  exit 0
fi

# Hardware topology & runtime environment locks for 8x H100
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export FLASHINFER_DISABLE_VERSION_CHECK=1
export VLLM_ENGINE_READY_TIMEOUT_S=7200
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "============================================================"
echo " ▶️ vLLM Cluster Launcher: Mistral Large 4 (Le Chonk)"
echo " Target Arch: 8x H100 80GB (Tensor Parallel size: 8)"
echo " Format:      Native NVFP4 Quantization"
echo " Model:       ${MODEL_PATH}"
echo " Endpoint:    ${SERVE_HOST}:${SERVE_PORT}"
echo "============================================================"

echo "[PRE-FLIGHT] Verifying aggregate memory across 8x H100 nodes..."
python3 - <<EOF
import torch
if torch.cuda.device_count() < 8:
    print(f"[ERROR] Expected 8 GPUs, but torch detected only {torch.cuda.device_count()}.")
    exit(1)
for i in range(8):
    free_mem = torch.cuda.mem_get_info(i)[0] / (1024**3)
    print(f"  - GPU {i}: {free_mem:.2f} GB free VRAM")
EOF

echo "[PRE-FLIGHT] Clearing CUDA memory caches..."
python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# Production vLLM serve invocation with official Mistral parsing flags
exec vllm serve "${MODEL_PATH}" \
  --host "${SERVE_HOST}" \
  --port "${SERVE_PORT}" \
  --tensor-parallel-size 8 \
  --quantization nvfp4 \
  --max-model-len 32000 \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.97 \
  --enable-prefix-caching \
  --tool-call-parser mistral \
  --reasoning-parser mistral \
  --enable-auto-tool-choice \
  --trust-remote-code