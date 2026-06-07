#!/usr/bin/env bash
set -euo pipefail

echo "=== Clearing vLLM, Torch, and FlashInfer caches ==="

# User's home directory
HOME_DIR="${HOME:-/home/user}"

# Cache locations to clear
VLLM_CACHE="$HOME_DIR/.cache/vllm"
TORCH_CACHE="$HOME_DIR/.cache/torch"
FLASHINFER_CACHE="$HOME_DIR/.cache/flashinfer"
COMPILER_CACHE="$HOME_DIR/.cache/compiler"
INDUCTOR_CACHE="$HOME_DIR/.cache/vllm/torch_compile_cache"

# Create list of directories to remove
DIRS_TO_REMOVE=(
  "$VLLM_CACHE"
  "$TORCH_CACHE"
  "$FLASHINFER_CACHE"
  "$COMPILER_CACHE"
  "$INDUCTOR_CACHE"
)

# Remove each directory
for dir in "${DIRS_TO_REMOVE[@]}"; do
  if [[ -d "$dir" ]]; then
    echo "Removing: $dir"
    rm -rf "$dir"
  else
    echo "Skipped (not found): $dir"
  fi
done

# Also clear any shared memory resources that might be leaked
if command -v sync >/dev/null 2>&1; then
  echo "Flushing filesystem buffers..."
  sync
fi

# Clear CUDA memory if any processes are running
if command -v nvidia-smi >/dev/null 2>&1; then
  RUNNING_VLLM=$(ps aux | grep -c '[v]llm' || echo 0)
  if [[ "$RUNNING_VLLM" != "0" ]]; then
    echo "Warning: vLLM processes are still running. Please stop them first with: pkill -f vllm"
  fi
fi

echo "=== Cache clearing complete ==="
echo "All compiled artifacts have been removed. Next run will rebuild everything from scratch."
