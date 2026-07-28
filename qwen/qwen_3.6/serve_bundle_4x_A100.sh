#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE}")" && pwd)"

MODEL_TEXT_VISION="${ROOT_DIR}/models/QuantTrio__Qwen3.6-27B-AWQ/"
MODEL_IMAGE_EDIT="${ROOT_DIR}/models/Qwen__Qwen-Image-Edit-2511/"

echo "============================================================"
echo " 📦 Bundle Launcher: Qwen 3.6 + Image Edit"
echo " Target Arch: Co-habitation on 4x A100 (40GB)"
echo " Environment: Script-relative 'models/' deployment"
echo "============================================================"

# 1. Lancement de Qwen 3.6 sur les GPU 0, 1, 2
echo "▶️ [1/2] Lancement de Qwen 3.6 Multimodal (Port 8000)..."
nohup bash "${ROOT_DIR}/serve_qwen_3.6_4x_A100.sh" "${MODEL_TEXT_VISION}" --port 8000 > "${ROOT_DIR}/vllm_qwen3.6.log" 2>&1 &
PID_TEXT=$!
echo "       👉 PID : ${PID_TEXT} | Logs : qwen/qwen_3.6/vllm_qwen3.6.log"

sleep 5

# 2. Lancement de Qwen Image Edit sur le GPU 3
echo "▶️ [2/2] Lancement de Qwen Image Edit (Port 8001)..."
nohup bash "${ROOT_DIR}/serve_qwen_image_edit_A100.sh" "${MODEL_IMAGE_EDIT}" --port 8001 > "${ROOT_DIR}/vllm_image_edit.log" 2>&1 &
PID_IMAGE=$!
echo "       👉 PID : ${PID_IMAGE} | Logs : qwen/qwen_3.6/vllm_image_edit.log"

echo "============================================================"
echo " ✅ Bundle initialisé en arrière-plan !"
echo "============================================================"
echo " Suivre le LLM/VLM :   tail -f vllm_qwen3.6.log"
echo " Suivre l'Éditeur :    tail -f vllm_image_edit.log"
echo " Pour tout couper :    kill ${PID_TEXT} ${PID_IMAGE}"
echo "============================================================"
