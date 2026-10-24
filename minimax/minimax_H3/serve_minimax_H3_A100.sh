#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/../../venv"

SERVE_HOST="${HOST:-127.0.0.1}"
SERVE_PORT="${PORT:-8002}"
MODEL_PATH=""
FORCE_DOWNLOAD=false
CONTINUE_DOWNLOAD=false

# Configuration pour MiniMax-H3 (Vidéo & Audio natif)
HF_REPO="MiniMaxAI/MiniMax-H3"
DEFAULT_MODEL="${ROOT_DIR}/models/MiniMaxAI__MiniMax-H3"

usage() {
  cat <<EOF
Usage: $(basename "${BASH_SOURCE[0]}") [MODEL_PATH] [OPTIONS]
Tested Architecture: A100 (40GB) - Isolates on 1 single GPU with Aggressive Offloading

Options:
  --host HOST             Host/interface (default: ${SERVE_HOST})
  --port PORT             Port to listen on (default: ${SERVE_PORT})
  --download              Force a fresh download of the model to the local folder
  --continue-download     Resume a broken or incomplete download
  -h, --help              Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)              SERVE_HOST="$2"; shift 2 ;;
    --host=*)            SERVE_HOST="${1#*=}"; shift ;;
    --port)              SERVE_PORT="$2"; shift 2 ;;
    --port=*)            SERVE_PORT="${1#*=}"; shift ;;
    --download)          FORCE_DOWNLOAD=true; shift ;;
    --continue-download) CONTINUE_DOWNLOAD=true; shift ;;
    -h|--help)           usage; exit 0 ;;
    -*)                  echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
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

# --- SECTION TÉLÉCHARGEMENT EXTENSIBLE (hf) ---
if [[ "${MODEL_PATH}" == "${DEFAULT_MODEL}" && ( ! -d "${DEFAULT_MODEL}" || "${FORCE_DOWNLOAD}" == true || "${CONTINUE_DOWNLOAD}" == true ) ]]; then
  echo "============================================================"
  if [[ "${FORCE_DOWNLOAD}" == true ]]; then
    echo " 📥 [FORCÉ] Re-téléchargement complet depuis Hugging Face..."
  elif [[ "${CONTINUE_DOWNLOAD}" == true ]]; then
    echo " 🔄 [REPRISE] Continuité d'un téléchargement interrompu..."
  else
    echo " 📥 Modèle local introuvable. Téléchargement initial depuis Hugging Face..."
  fi
  echo " Repo:    ${HF_REPO}"
  echo " Cible:   ${DEFAULT_MODEL}"
  echo " Utility: hf (New CLI)"
  echo "============================================================"
  
  if ! command -v hf &> /dev/null; then
    echo "[DOWNLOAD][ERROR] La commande 'hf' n'a pas été trouvée dans votre PATH."
    exit 1
  fi
  
  mkdir -p "${DEFAULT_MODEL}"
  
  hf download "${HF_REPO}" --local-dir "${DEFAULT_MODEL}"
  echo "[DOWNLOAD][OK] Téléchargement terminé ou vérifié avec succès !"
fi
# ------------------------------------------------

export CUDA_VISIBLE_DEVICES="2"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export FLASHINFER_DISABLE_VERSION_CHECK=1
export VLLM_RPC_TIMEOUT=1200 

echo "============================================================"
echo " ▶️ vLLM Launcher: MiniMax-H3 Omni (Video + Audio)"
echo " Target Arch: 1x A100 40GB (Isolating on GPU 2)"
echo " Model:       ${MODEL_PATH}"
echo " Endpoint:    ${SERVE_HOST}:${SERVE_PORT}"
echo " Quantization: FP8 (Online) + Model-level CPU Offload"
echo "============================================================"

echo "[PRE-FLIGHT] Checking GPU memory availability on physical GPU 2..."
GPU_MEM_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 2 | tr -d '[:space:]')
if [[ -z "${GPU_MEM_FREE}" ]]; then
  echo "[PRE-FLIGHT][WARN] Could not query free memory for GPU 2. Proceeding anyway."
elif [[ "${GPU_MEM_FREE}" -lt 32000 ]]; then
  echo "[PRE-FLIGHT][WARN] GPU 2 has only ${GPU_MEM_FREE}MiB free. MiniMax-H3 requires extreme VRAM allocation."
  echo "                 If an OOM occurs, ensure no other heavy process runs on GPU 2."
else
  echo "[PRE-FLIGHT][OK] GPU 2 has ${GPU_MEM_FREE}MiB free."
fi

echo "[PRE-FLIGHT] Clearing PyTorch distributed and CUDA cache to prevent fragmentation locks..."
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# Execution sur le port 8002 avec optimisations VRAM agressives pour MiniMax-H3
exec vllm serve "${MODEL_PATH}" \
  --host "${SERVE_HOST}" \
  --port "${SERVE_PORT}" \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.95 \
  --omni \
  --quantization fp8 \
  --vae-use-slicing \
  --vae-use-tiling \
  --enable-layerwise-offload \
  --cache-backend cache_dit
