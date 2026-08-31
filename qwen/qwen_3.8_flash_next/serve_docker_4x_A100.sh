#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SERVE_HOST="${HOST:-0.0.0.0}"
SERVE_PORT="${PORT:-8000}"
MODEL_PATH=""
DEFAULT_MODEL="${ROOT_DIR}/models/Qwen__Qwen3.8-Flash-Next-FP8"

DOCKER_IMAGE="${DOCKER_IMAGE:-vllm/vllm-openai:qwen38-flash-next}"
DOCKER_DEVICES="${DOCKER_DEVICES:-0,1,2,3}"

usage() {
  cat <<EOF
Usage: $(basename "${BASH_SOURCE[0]}") [MODEL_PATH] [OPTIONS]
Tested Architecture: 4x A100 (40GB) via Docker container

Options:
  --host HOST        Host/interface (default: ${SERVE_HOST})
  --port PORT        Port to listen on (default: ${SERVE_PORT})
  --image IMAGE      Docker image to use (default: ${DOCKER_IMAGE})
  --gpus DEVICES     Comma-separated GPU device IDs (default: ${DOCKER_DEVICES})
  -h, --help         Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)    SERVE_HOST="$2"; shift 2 ;;
    --host=*)  SERVE_HOST="${1#*=}"; shift ;;
    --port)    SERVE_PORT="$2"; shift 2 ;;
    --port=*)  SERVE_PORT="${1#*=}"; shift ;;
    --image)   DOCKER_IMAGE="$2"; shift 2 ;;
    --image=*) DOCKER_IMAGE="${1#*=}"; shift ;;
    --gpus)    DOCKER_DEVICES="$2"; shift 2 ;;
    --gpus=*)  DOCKER_DEVICES="${1#*=}"; shift ;;
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

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Model path does not exist: ${MODEL_PATH}" >&2
  exit 1
fi

if ! command -v docker &>/dev/null; then
  echo "[ERROR] Docker is not installed or not in PATH." >&2
  exit 1
fi

IFS=',' read -ra GPU_ARRAY <<< "${DOCKER_DEVICES}"
echo "============================================================"
echo " ▶️ vLLM Docker Launcher: Qwen 3.8 Flash Next (FP8)"
echo " Target Arch: 4x A100 40GB (Docker isolated)"
echo " Image:       ${DOCKER_IMAGE}"
echo " Model:       ${MODEL_PATH}"
echo " Endpoint:    ${SERVE_HOST}:${SERVE_PORT}"
echo "============================================================"

echo "[PRE-FLIGHT] Checking GPU memory availability on physical GPUs ${DOCKER_DEVICES}..."
INSUFFICIENT_MEMORY=0
for GPU_ID in "${GPU_ARRAY[@]}"; do
  GPU_MEM_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${GPU_ID}" 2>/dev/null | tr -d '[:space:]')
  if [[ -z "${GPU_MEM_FREE}" ]]; then
    echo "[PRE-FLIGHT][WARN] Could not query free memory for GPU ${GPU_ID}. Proceeding anyway."
  elif [[ "${GPU_MEM_FREE}" -lt 15360 ]]; then
    echo "[PRE-FLIGHT][ERROR] GPU ${GPU_ID} has only ${GPU_MEM_FREE}MiB free (15360MiB required)."
    INSUFFICIENT_MEMORY=1
  else
    echo "[PRE-FLIGHT][OK] GPU ${GPU_ID}: ${GPU_MEM_FREE}MiB free"
  fi
done

if [[ "${INSUFFICIENT_MEMORY}" -eq 1 ]]; then
  echo "[PRE-FLIGHT][ERROR] Insufficient free memory on one or more target GPUs."
  echo "                 Run 'nvidia-smi' to identify the process and kill it before retrying."
  exit 1
fi

TP_SIZE=${#GPU_ARRAY[@]}

echo "Starting vLLM Docker container (TP=${TP_SIZE})..."

CONTAINER_USER="$(id -u):$(id -g)"

exec docker run --rm \
  --user "${CONTAINER_USER}" \
  -e HOME=/tmp \
  --gpus "\"device=${DOCKER_DEVICES}\"" \
  --ipc=host \
  --network=host \
  -v "${MODEL_PATH}:/data/model:ro" \
  "${DOCKER_IMAGE}" \
  --model /data/model \
  --host "${SERVE_HOST}" \
  --port "${SERVE_PORT}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --quantization fp8 \
  --kv-cache-dtype fp8 \
  --max-model-len 262144 \
  --max-num-seqs 128 \
  --gpu-memory-utilization 0.92 \
  --enable-prefix-caching \
  --trust-remote-code \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --limit-mm-per-prompt '{"image": 4}'