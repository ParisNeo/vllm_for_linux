#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SERVE_HOST="${HOST:-0.0.0.0}"
SERVE_PORT="${PORT:-8000}"
MODEL_PATH=""
DEFAULT_MODEL="${ROOT_DIR}/models/zai-org__GLM-5.3-Flash-FP8"

DOCKER_IMAGE="${DOCKER_IMAGE:-vllm/vllm-openai:glm53-flash}"
DOCKER_DEVICES="${DOCKER_DEVICES:-0,1,2,3,4,5,6,7}"
SHM_SIZE="${SHM_SIZE:-32g}"

usage() {
  cat <<EOF
Usage: $(basename "${BASH_SOURCE[0]}") [MODEL_PATH] [OPTIONS]
Tested Architecture: 8x H200 via Docker container

Options:
  --host HOST        Host/interface (default: ${SERVE_HOST})
  --port PORT        Port to listen on (default: ${SERVE_PORT})
  --image IMAGE      Docker image to use (default: ${DOCKER_IMAGE})
  --gpus DEVICES     Comma-separated GPU device IDs (default: ${DOCKER_DEVICES})
  --shm-size SIZE    Docker shared memory size (default: ${SHM_SIZE})
  -h, --help         Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)        SERVE_HOST="$2"; shift 2 ;;
    --host=*)      SERVE_HOST="${1#*=}"; shift ;;
    --port)        SERVE_PORT="$2"; shift 2 ;;
    --port=*)      SERVE_PORT="${1#*=}"; shift ;;
    --image)       DOCKER_IMAGE="$2"; shift 2 ;;
    --image=*)     DOCKER_IMAGE="${1#*=}"; shift ;;
    --gpus)        DOCKER_DEVICES="$2"; shift 2 ;;
    --gpus=*)      DOCKER_DEVICES="${1#*=}"; shift ;;
    --shm-size)    SHM_SIZE="$2"; shift 2 ;;
    --shm-size=*)  SHM_SIZE="${1#*=}"; shift ;;
    -h|--help)     usage; exit 0 ;;
    -*)            echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
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

CONTAINER_MODEL_PATH="/data/model"
BYPASS_VOLUME_NAME="vllm-snap-bypass-$$"

cleanup() {
  docker rm -f "${BYPASS_VOLUME_NAME}" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

docker create --name "${BYPASS_VOLUME_NAME}" -v "${MODEL_PATH}:${CONTAINER_MODEL_PATH}:ro" busybox:latest >/dev/null 2>&1 || {
  echo "[ERROR] Failed to create bind mount bypass container." >&2
  exit 1
}

IFS=',' read -ra GPU_ARRAY <<< "${DOCKER_DEVICES}"
echo "============================================================"
echo " ▶️ vLLM Docker Launcher: GLM-5.3 Flash (FP8)"
echo " Target Arch: 8x H200 (Docker isolated)"
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
  --shm-size "${SHM_SIZE}" \
  --ipc=host \
  -p "${SERVE_HOST}:${SERVE_PORT}:8000" \
  --volumes-from "${BYPASS_VOLUME_NAME}" \
  "${DOCKER_IMAGE}" \
  --model "${CONTAINER_MODEL_PATH}" \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size "${TP_SIZE}" \
  --quantization fp8 \
  --kv-cache-dtype fp8_e4m3 \
  --max-num-seqs 128 \
  --tool-call-parser glm47 \
  --trust-remote-code