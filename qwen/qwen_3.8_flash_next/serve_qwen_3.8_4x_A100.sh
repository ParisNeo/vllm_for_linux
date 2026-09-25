#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/../../venv"

SERVE_HOST="${HOST:-127.0.0.1}"
SERVE_PORT="${PORT:-8000}"
MODEL_PATH=""
DEFAULT_MODEL="${ROOT_DIR}/models/Qwen__Qwen3.8-Flash-Next-FP8"

usage() {
  cat <<EOF
Usage: $(basename "${BASH_SOURCE[0]}") [MODEL_PATH] [OPTIONS]
Tested Architecture: 4x A100 (40GB) - Full node allocation for large context MoE

Options:
  --host HOST        Host/interface (default: ${SERVE_HOST})
  --port PORT        Port to listen on (default: ${SERVE_PORT})
  -h, --help         Show this help message
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

if [[ -f "${VENV_DIR}/bin/activate" ]]; then source "${VENV_DIR}/bin/activate"; else
  echo "Virtual environment not found at ${VENV_DIR}" >&2; exit 1
fi

# Allocation complète des 4 GPU de calcul
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export FLASHINFER_DISABLE_VERSION_CHECK=1
export VLLM_RPC_TIMEOUT=600

echo "============================================================"
echo " ▶️ vLLM High-Throughput Launcher: Qwen 3.8 Flash Next"
echo " Target Arch: 4x A100 40GB (Tensor Parallel on GPUs 0,1,2,3)"
echo " Model:       ${MODEL_PATH}"
echo " Endpoint:    ${SERVE_HOST}:${SERVE_PORT}"
echo "============================================================"

echo "[PRE-FLIGHT] Checking GPU memory availability on physical GPUs 0,1,2,3..."
GPU_MEM_FREE_0=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | tr -d '[:space:]')
GPU_MEM_FREE_1=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1 | tr -d '[:space:]')
GPU_MEM_FREE_2=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 2 | tr -d '[:space:]')
GPU_MEM_FREE_3=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 3 | tr -d '[:space:]')

if [[ -z "${GPU_MEM_FREE_0}" || -z "${GPU_MEM_FREE_1}" || -z "${GPU_MEM_FREE_2}" || -z "${GPU_MEM_FREE_3}" ]]; then
  echo "[PRE-FLIGHT][WARN] Could not query free memory for all 4 GPUs. Proceeding anyway."
elif [[ "${GPU_MEM_FREE_0}" -lt 15360 || "${GPU_MEM_FREE_1}" -lt 15360 || "${GPU_MEM_FREE_2}" -lt 15360 || "${GPU_MEM_FREE_3}" -lt 15360 ]]; then
  echo "[PRE-FLIGHT][ERROR] Insufficient free memory on target GPUs."
  echo "                 GPU 0: ${GPU_MEM_FREE_0}MiB | GPU 1: ${GPU_MEM_FREE_1}MiB"
  echo "                 GPU 2: ${GPU_MEM_FREE_2}MiB | GPU 3: ${GPU_MEM_FREE_3}MiB"
  echo "                 At least 15360MiB is required per GPU."
  exit 1
else
  echo "[PRE-FLIGHT][OK] Memory clear on all 4 targets (0,1,2,3)."
fi

echo "[PRE-FLIGHT] Clearing PyTorch distributed and CUDA cache..."
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# Lancement propre en contournant la détection FP8 du modèle
# Au lieu de 'vllm serve', on lance via un script Python en ligne qui nettoie la config à la volée
exec python -c "
import json
import os
import sys
from transformers import AutoConfig
from vllm.entrypoints.openai.api_server import main

# 1. Charger et patcher la config en mémoire pour supprimer le bloc FP8 encombrant
model_path = '${MODEL_PATH}'
config_file = os.path.join(model_path, 'config.json')

if os.path.exists(config_file):
    with open(config_file, 'r') as f:
        config_data = json.load(f)
    
    if 'quantization_config' in config_data:
        print('[PATCH] Suppression dynamique de quantization_config en mémoire...')
        del config_data['quantization_config']
        
        # On force transformers/vllm à lire notre version modifiée en surchargeant la méthode de cache
        original_from_pretrained = AutoConfig.from_pretrained
        def patched_from_pretrained(pretrained_model_name_or_path, **kwargs):
            if pretrained_model_name_or_path == model_path:
                return AutoConfig.from_dict(config_data)
            return original_from_pretrained(pretrained_model_name_or_path, **kwargs)
        AutoConfig.from_pretrained = patched_from_pretrained

# 2. Reconstruire les arguments pour le serveur vLLM
sys.argv = [
    'vllm', 'serve', model_path,
    '--host', '${SERVE_HOST}',
    '--port', '${SERVE_PORT}',
    '--tensor-parallel-size', '4',
    '--disable-custom-all-reduce',
    '--quantization', 'unquantized',
    '--dtype', 'bfloat16',
    '--kv-cache-dtype', 'auto',
    '--max-model-len', '262144',
    '--max-num-seqs', '128',
    '--gpu-memory-utilization', '0.88',
    '--enable-prefix-caching',
    '--trust-remote-code',
    '--reasoning-parser', 'qwen3',
    '--default-chat-template-kwargs', '{\"enable_thinking\": false}',
    '--limit-mm-per-prompt', '{\"image\": 4}'
]

# 3. Lancer le serveur vLLM standard
main()
"
