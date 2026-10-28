#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/../../venv"

SERVE_HOST="${HOST:-127.0.0.1}"
SERVE_PORT="${PORT:-8000}"
MODEL_PATH=""
# Official Hugging Face repository for DeepSeek-V4.1-Flash-1017
DEFAULT_MODEL="${ROOT_DIR}/models/deepseek-ai__DeepSeek-V4.1-Flash-1017"

usage() {
  cat <<EOF
Usage: $(basename "${BASH_SOURCE[0]}") [MODEL_PATH] [OPTIONS]
Target Architecture: 8x NVIDIA H100 (80GB) - Full Tensor Parallelism

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

# Lock vLLM across all 8 H100 GPUs
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export FLASHINFER_DISABLE_VERSION_CHECK=1
# Extended timeout for large multi-GPU weight initialization and validation
export VLLM_ENGINE_READY_TIMEOUT_S=3600

echo "============================================================"
echo " ▶️ vLLM Cluster Launcher: DeepSeek-V4.1-Flash-1017"
echo " Target Arch: 8x H100 80GB (Tensor Parallel size: 8)"
echo " Model:       ${MODEL_PATH}"
echo " Endpoint:    ${SERVE_HOST}:${SERVE_PORT}"
echo "============================================================"

# Auto-download Check & HF Resolution
if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "[AUTO-DOWNLOAD] Path '${MODEL_PATH}' not found locally."
  
  HF_REPO="${MODEL_PATH}"
  if [[ "${MODEL_PATH}" == *"/models/deepseek-ai__"* ]]; then
    HF_REPO="deepseek-ai/DeepSeek-V4.1-Flash-1017"
  fi

  echo "[AUTO-DOWNLOAD] Pulling '${HF_REPO}' via huggingface_hub..."
  python3 - <<EOF
import sys
from huggingface_hub import snapshot_download

model_id = "${HF_REPO}"
print(f"[*] Downloading repository: {model_id} (this may take a while for large model sizes)...")
try:
    local_path = snapshot_download(
        repo_id=model_id,
        ignore_patterns=["*.msgpack", "*.h5"]
    )
    print(f"[+] Download complete. Snapshot cache: {local_path}")
    with open("/tmp/resolved_ds_v41_path.txt", "w") as f:
        f.write(local_path)
except Exception as e:
    print(f"[-] Download failed: {e}", file=sys.stderr)
    sys.exit(1)
EOF

  if [[ -f /tmp/resolved_ds_v41_path.txt ]]; then
    MODEL_PATH="$(cat /tmp/resolved_ds_v41_path.txt)"
    rm -f /tmp/resolved_ds_v41_path.txt
    echo "[AUTO-DOWNLOAD][OK] Set MODEL_PATH to: ${MODEL_PATH}"
  fi
else
  echo "[AUTO-DOWNLOAD][OK] Verified local model files at ${MODEL_PATH}"
fi

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

exec vllm serve "${MODEL_PATH}" \
  --host "${SERVE_HOST}" \
  --port "${SERVE_PORT}" \
  --tensor-parallel-size 8 \
  --tokenizer-mode deepseek_v41 \
  --max-model-len 131072 \
  --max-num-seqs 128 \
  --gpu-memory-utilization 0.94 \
  --enable-prefix-caching \
  --trust-remote-code \
  --reasoning-parser deepseek_r1 \
  --default-chat-template-kwargs '{"enable_thinking": true}'