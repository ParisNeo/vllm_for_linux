#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/../../venv"

SERVE_HOST="${HOST:-127.0.0.1}"
SERVE_PORT="${PORT:-8000}"
MODEL_PATH=""
# Target: DeepSeek-V4.1-Flash-1017 INT4 quantized
DEFAULT_MODEL="${ROOT_DIR}/models/cyankiwi__DeepSeek-V4.1-Flash-1017-AWQ"

usage() {
  cat <<EOF
Usage: $(basename "${BASH_SOURCE[0]}") [MODEL_PATH] [OPTIONS]
Target Architecture: 2x A100 (40GB) - Leaves GPUs 2 and 3 unallocated

Options:
  --host HOST         Host/interface (default: ${SERVE_HOST})
  --port PORT         Port to listen on (default: ${SERVE_PORT})
  -h, --help          Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)   SERVE_HOST="$2"; shift 2 ;;
    --host=*) SERVE_HOST="${1#*=}"; shift ;;
    --port)   SERVE_PORT="$2"; shift 2 ;;
    --port=*) SERVE_PORT="${1#*=}"; shift ;;
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
  echo "Virtual environment not found at ${VENV_DIR}" >&2; exit 1
fi

# Lock vLLM to GPUs 0 and 1
export CUDA_VISIBLE_DEVICES="0,1"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export FLASHINFER_DISABLE_VERSION_CHECK=1

echo "============================================================"
echo " ▶️ vLLM Launcher: DeepSeek V4.1 Flash (1017) INT4"
echo " Target Arch: 2x A100 40GB (Tensor Parallel on GPUs 0,1)"
echo " Model:       ${MODEL_PATH}"
echo " Endpoint:    ${SERVE_HOST}:${SERVE_PORT}"
echo "============================================================"

# Auto-download: Check if MODEL_PATH points to an existing directory
if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "[AUTO-DOWNLOAD] Path '${MODEL_PATH}' not found on disk."
  
  # Determine Hugging Face Repository Name
  # If user supplied a directory path like 'models/cyankiwi__DeepSeek-V4.1-Flash-1017-AWQ',
  # map it back to standard Hugging Face ID 'cyankiwi/DeepSeek-V4.1-Flash-1017-AWQ'
  HF_REPO="${MODEL_PATH}"
  if [[ "${MODEL_PATH}" == *"/models/cyankiwi__"* ]]; then
    HF_REPO="cyankiwi/DeepSeek-V4.1-Flash-1017-AWQ"
  fi

  echo "[AUTO-DOWNLOAD] Downloading '${HF_REPO}' via huggingface_hub..."
  python3 - <<EOF
import sys
from huggingface_hub import snapshot_download

model_id = "${HF_REPO}"
print(f"[*] Fetching repository: {model_id}")
try:
    local_path = snapshot_download(
        repo_id=model_id,
        ignore_patterns=["*.msgpack", "*.h5", "*.ot"]
    )
    print(f"[+] Download finished. Cache location: {local_path}")
    with open("/tmp/resolved_ds1017_path.txt", "w") as f:
        f.write(local_path)
except Exception as e:
    print(f"[-] Download failed: {e}", file=sys.stderr)
    sys.exit(1)
EOF

  if [[ -f /tmp/resolved_ds1017_path.txt ]]; then
    MODEL_PATH="$(cat /tmp/resolved_ds1017_path.txt)"
    rm -f /tmp/resolved_ds1017_path.txt
    echo "[AUTO-DOWNLOAD][OK] Set MODEL_PATH to cached snapshot: ${MODEL_PATH}"
  fi
else
  echo "[AUTO-DOWNLOAD][OK] Local model directory verified at ${MODEL_PATH}"
fi

echo "[PRE-FLIGHT] Checking GPU memory availability on physical GPUs 0,1..."
GPU_MEM_FREE_0=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | tr -d '[:space:]')
GPU_MEM_FREE_1=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1 | tr -d '[:space:]')

if [[ -z "${GPU_MEM_FREE_0}" || -z "${GPU_MEM_FREE_1}" ]]; then
  echo "[PRE-FLIGHT][WARN] Could not query free memory for GPUs 0,1. Proceeding anyway."
elif [[ "${GPU_MEM_FREE_0}" -lt 18432 || "${GPU_MEM_FREE_1}" -lt 18432 ]]; then
  echo "[PRE-FLIGHT][ERROR] Insufficient free memory on target GPUs."
  echo "                 GPU 0: ${GPU_MEM_FREE_0}MiB free / GPU 1: ${GPU_MEM_FREE_1}MiB free"
  echo "                 At least 18432MiB (~18GB) required per GPU."
  exit 1
else
  echo "[PRE-FLIGHT][OK] GPU 0: ${GPU_MEM_FREE_0}MiB free | GPU 1: ${GPU_MEM_FREE_1}MiB free"
fi

echo "[PRE-FLIGHT] Clearing PyTorch distributed and CUDA cache..."
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

exec vllm serve "${MODEL_PATH}" \
  --host "${SERVE_HOST}" \
  --port "${SERVE_PORT}" \
  --tensor-parallel-size 2 \
  --disable-custom-all-reduce \
  --quantization awq \
  --max-model-len 131072 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.92 \
  --kv-cache-dtype auto \
  --enable-prefix-caching \
  --trust-remote-code \
  --reasoning-parser deepseek_r1 \
  --default-chat-template-kwargs '{"enable_thinking": true}'