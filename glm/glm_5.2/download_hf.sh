#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/../../venv"

if [[ -f "${VENV_DIR}/bin/activate" ]]; then
  source "${VENV_DIR}/bin/activate"
else
  echo "Error: Virtual environment not found at ${VENV_DIR}" >&2
  exit 1
fi

exec python3 "${ROOT_DIR}/download_hf.py" "$@"