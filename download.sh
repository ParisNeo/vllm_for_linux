#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
  echo "Virtual environment not found at ${VENV_DIR}" >&2
  exit 1
fi

source "${VENV_DIR}/bin/activate"
python "${ROOT_DIR}/download.py" "$@"
