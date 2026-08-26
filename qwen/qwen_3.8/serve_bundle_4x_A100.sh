#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# amd/Qwen3.8-27B-Quark-AWQ-INT4-W4A16
MODEL_TEXT_VISION="${ROOT_DIR}/models/amd__Qwen3.8-27B-Quark-AWQ-INT4-W4A16/"
MODEL_IMAGE_EDIT="${ROOT_DIR}/models/Qwen__Qwen-Image-Edit-2511/"
MODEL_VIDEO_WAN="${ROOT_DIR}/models/Wan-AI__Wan2.2-TI2V-5B-Diffusers/"

SERVE_MODE="all"
PIDS_TO_KILL=()

usage() {
  cat <<EOF
Usage: $(basename "${BASH_SOURCE[0]}") [OPTIONS]

Options:
  --serve MODE   Services to launch (default: all)
                 Modes: all, text+image, text, image, video
  -h, --help     Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --serve)   SERVE_MODE="$2"; shift 2 ;;
    --serve=*) SERVE_MODE="${1#*=}"; shift ;;
    -h|--help) usage; exit 0 ;;
    *)         echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

SERVE_TEXT=false
SERVE_IMAGE=false
SERVE_VIDEO=false

case "${SERVE_MODE}" in
  all)         SERVE_TEXT=true; SERVE_IMAGE=true; SERVE_VIDEO=true ;;
  text+image)  SERVE_TEXT=true; SERVE_IMAGE=true ;;
  text)        SERVE_TEXT=true ;;
  image)       SERVE_IMAGE=true ;;
  video)       SERVE_VIDEO=true ;;
  *)
    echo "[ERROR] Invalid mode for --serve: ${SERVE_MODE}" >&2
    echo "        Valid modes: all, text+image, text, image, video" >&2
    exit 1
    ;;
esac

echo "============================================================"
echo " 📦 Bundle Launcher: Qwen 3.8 + Image Edit + Wan 2.2 Video"
echo " Target Arch: Co-habitation on 4x A100 (40GB)"
echo " Environment: Script-relative 'models/' deployment"
echo " Mode:        ${SERVE_MODE}"
echo "============================================================"

if [[ "${SERVE_TEXT}" == true ]]; then
  echo "▶️ Lancement de Qwen 3.8 Multimodal (Port 8000)..."
  nohup bash "${ROOT_DIR}/serve_qwen_3.8_4x_A100.sh" "${MODEL_TEXT_VISION}" --port 8000 > "${ROOT_DIR}/vllm_qwen3.8.log" 2>&1 &
  PID_TEXT=$!
  PIDS_TO_KILL+=("${PID_TEXT}")
  echo "   👉 PID : ${PID_TEXT} | Logs : qwen/qwen_3.8/vllm_qwen3.8.log"
  sleep 5
fi

if [[ "${SERVE_IMAGE}" == true ]]; then
  echo "▶️ Lancement de Qwen Image Edit (Port 8001)..."
  nohup bash "${ROOT_DIR}/serve_qwen_image_edit_A100.sh" "${MODEL_IMAGE_EDIT}" --port 8001 > "${ROOT_DIR}/vllm_image_edit.log" 2>&1 &
  PID_IMAGE=$!
  PIDS_TO_KILL+=("${PID_IMAGE}")
  echo "   👉 PID : ${PID_IMAGE} | Logs : qwen/qwen_3.8/vllm_image_edit.log"
  sleep 5
fi

if [[ "${SERVE_VIDEO}" == true ]]; then
  echo "▶️ Lancement de Wan 2.2 Video Generation (Port 8002)..."
  nohup bash "${ROOT_DIR}/serve_wan_2.2_5B_A100.sh" "${MODEL_VIDEO_WAN}" --port 8002 > "${ROOT_DIR}/vllm_wan_video.log" 2>&1 &
  PID_VIDEO=$!
  PIDS_TO_KILL+=("${PID_VIDEO}")
  echo "   👉 PID : ${PID_VIDEO} | Logs : qwen/qwen_3.8/vllm_wan_video.log"
  sleep 5
fi

if [[ ${#PIDS_TO_KILL[@]} -eq 0 ]]; then
  echo "[WARN] No services were launched. Exiting."
  exit 0
fi

KILL_CMD="kill ${PIDS_TO_KILL[*]}"

echo "============================================================"
echo " ✅ Bundle initialisé en arrière-plan !"
echo "============================================================"
if [[ "${SERVE_TEXT}" == true ]]; then
  echo " Suivre le LLM/VLM :   tail -f vllm_qwen3.8.log"
fi
if [[ "${SERVE_IMAGE}" == true ]]; then
  echo " Suivre l'Éditeur :    tail -f vllm_image_edit.log"
fi
if [[ "${SERVE_VIDEO}" == true ]]; then
  echo " Suivre la Vidéo :     tail -f vllm_wan_video.log"
fi
echo " Pour tout couper :    ${KILL_CMD}"
echo "============================================================"
