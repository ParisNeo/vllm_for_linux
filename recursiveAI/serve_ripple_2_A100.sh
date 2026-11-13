#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/../../venv"

SERVE_HOST="${HOST:-127.0.0.1}"
SERVE_PORT="${PORT:-8000}"
MODEL_PATH=""
DEFAULT_MODEL="${ROOT_DIR}/models/ricursiveAI__Ripple-65B-Preview-NVFP4"

usage() {
  cat <<EOF
Usage: $(basename "${BASH_SOURCE[0]}") [MODEL_PATH] [OPTIONS]
Target Architecture: 2x NVIDIA A100 (80GB) - Tensor Parallelism

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
        return 0
      fi
    fi
  fi

  echo "[AUTO-DOWNLOAD][ERROR] Unable to obtain model at ${model_path}" >&2
  return 1
}

if ! ensure_model_local "${MODEL_PATH}"; then
  exit 1
fi

export CUDA_VISIBLE_DEVICES="0,1"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export FLASHINFER_DISABLE_VERSION_CHECK=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600

echo "============================================================"
echo " ▶️ vLLM Cluster Launcher: Ripple-65B-Preview-NVFP4 (2x A100)"
echo " Target Arch: 2x A100 80GB (Tensor Parallel size: 2)"
echo " Model:       ${MODEL_PATH}"
echo " Endpoint:    ${SERVE_HOST}:${SERVE_PORT}"
echo "============================================================"

python3 - <<EOF
import torch
if torch.cuda.device_count() < 2:
    print(f"[ERROR] Expected at least 2 GPUs, but torch detected only {torch.cuda.device_count()}.")
    exit(1)
for i in range(2):
    free_mem = torch.cuda.mem_get_info(i)[0] / (1024**3)
    print(f"  - GPU {i}: {free_mem:.2f} GB free VRAM")
EOF

python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

exec vllm serve "${MODEL_PATH}" \
  --host "${SERVE_HOST}" \
  --port "${SERVE_PORT}" \
  --tensor-parallel-size 2 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --trust-remote-code \
  --reasoning-parser ripple \
  --enable-auto-tool-choice \
  --tool-call-parser ripple