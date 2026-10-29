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

# Derive the HF repo id from a local models/<org>__<repo> path, or pass through.
resolve_repo_id() {
  local path_or_repo="$1"
  local models_prefix="${ROOT_DIR}/models/"
  if [[ "${path_or_repo}" == "${models_prefix}"* ]]; then
    local base
    base="$(basename "${path_or_repo}")"
    echo "${base/__//}"
  else
    echo "${path_or_repo}"
  fi
}

# Ensure the model lives at MODEL_PATH (flat layout, config.json at root).
# 1) Already present -> no-op.
# 2) HF cache snapshot exists -> materialize via snapshot_download(local_dir=...)
#    (hardlinks from cache, no network) with cp -a -L fallback for offline salvage.
# 3) Otherwise -> download straight into MODEL_PATH.
ensure_model_local() {
  local model_path="$1"
  local repo_id
  repo_id="$(resolve_repo_id "${model_path}")"

  if [[ -f "${model_path}/config.json" ]]; then
    echo "[AUTO-DOWNLOAD][OK] Verified local model files at ${model_path}"
    return 0
  fi

  echo "[AUTO-DOWNLOAD] Model not found at '${model_path}' (repo: ${repo_id})."
  mkdir -p "${model_path}"

  # Primary path: snapshot_download with local_dir -> flat layout, reuses cache blobs.
  if python3 - "${repo_id}" "${model_path}" <<'PY'
import sys
from huggingface_hub import snapshot_download

repo_id, local_dir = sys.argv[1], sys.argv[2]
print(f"[*] Materializing '{repo_id}' into '{local_dir}' (reuses HF cache when available)...")
snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    ignore_patterns=["*.msgpack", "*.h5"],
    resume_download=True,
)
PY
  then
    if [[ -f "${model_path}/config.json" ]]; then
      echo "[AUTO-DOWNLOAD][OK] Model materialized at ${model_path}"
      return 0
    fi
  else
    echo "[AUTO-DOWNLOAD][WARN] snapshot_download failed; attempting offline cache salvage..." >&2
  fi

  # Fallback: salvage an existing HF cache snapshot by dereferencing symlinks.
  local hf_home="${HF_HOME:-${HOME}/.cache/huggingface}"
  local cache_dir="${hf_home}/hub/models--${repo_id//\//--}"
  local snapshot_dir=""
  if [[ -d "${cache_dir}/snapshots" ]]; then
    snapshot_dir="$(find "${cache_dir}/snapshots" -mindepth 1 -maxdepth 1 -type d | head -n 1 || true)"
  fi

  if [[ -n "${snapshot_dir}" && -d "${snapshot_dir}" ]]; then
    echo "[AUTO-DOWNLOAD] Salvaging cache snapshot: ${snapshot_dir}"
    if cp -a -L "${snapshot_dir}/." "${model_path}/"; then
      if [[ -f "${model_path}/config.json" ]]; then
        echo "[AUTO-DOWNLOAD][OK] Salvaged model into ${model_path}"
        echo "[AUTO-DOWNLOAD] Old cache dir can be removed: ${cache_dir}"
        return 0
      fi
    fi
    echo "[AUTO-DOWNLOAD][ERROR] Salvage copy did not produce config.json" >&2
  fi

  echo "[AUTO-DOWNLOAD][ERROR] Unable to obtain model at ${model_path}" >&2
  return 1
}

if ! ensure_model_local "${MODEL_PATH}"; then
  exit 1
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