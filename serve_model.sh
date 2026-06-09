#!/usr/bin/env bash
set -euo pipefail

# Smart Model Server - Intelligent vLLM Launcher
# Automatically analyzes GPU resources and configures optimal settings
# Supports configuration caching, multimodal models, and fallback mechanisms

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check for --help
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat << 'EOF'
Smart Model Server - Intelligent vLLM Configuration

Usage:
  $0 --model <path-or-repo> [options]
  $0 <path-or-repo>  (legacy mode)

Options:
  --model PATH        Model directory or HuggingFace repo ID (required)
  --port PORT         Server port (default: 8000)
  --host HOST         Server host (default: 127.0.0.1)
  --max-model-len N   Maximum context length
  --dry-run           Show configuration without launching

Configuration Control:
  --reset-config      Reset cached configuration and re-optimize
  --no-cache          Skip cache, always re-optimize

GPU Constraints:
  --gpus IDS          Comma-separated GPU IDs (e.g., '0,1,2,3')
  --max-util FLOAT    Maximum GPU memory utilization (default: 0.90)
  --min-util FLOAT    Minimum GPU memory utilization (default: 0.50)
  --tp-size INT       Force specific tensor parallelism
  --min-free-gb FLOAT Min free GPU memory to consider GPU available (default: 10.0)
  --include-busy-gpus Don't filter out busy GPUs

Fallback Control:
  --no-fallback       Disable progressive fallback on OOM
  --fallback-steps N  Number of fallback attempts (default: 3)

Manual Mode (skip smart analysis):
  --manual            Use environment variables instead of auto-config

Environment Variables (for --manual mode):
  TP_SIZE             Tensor parallelism (default: auto)
  GPU_MEM_UTIL        GPU memory utilization (default: auto)
  PORT, HOST, MAX_MODEL_LEN

Examples:
  # Auto-optimize for model
  $0 --model models/google__gemma-4-31B-it/
  
  # Use specific GPUs only
  $0 --model models/llama-3.1-8b/ --gpus 0,1
  
  # Conservative memory usage
  $0 --model models/llama-3.1-8b/ --max-util 0.75
  
  # Force TP=4, re-optimize
  $0 --model models/llama-3.1-8b/ --tp-size 4 --reset-config
  
  # Dry run to see config
  $0 --model models/llama-3.1-8b/ --dry-run
  
  # Manual mode with env vars
  GPU_MEM_UTIL=0.70 $0 --manual --model models/llama-3.1-8b/
  
  # HuggingFace repo
  $0 --model google/gemma-4-31b-it --port 8080
EOF
  exit 0
fi

# Parse arguments
MODEL_INPUT=""
PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
DRY_RUN=""
MANUAL_MODE=""
SMART_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL_INPUT="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --host)
      HOST="$2"
      shift 2
      ;;
    --max-model-len)
      MAX_MODEL_LEN="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN="--dry-run"
      shift
      ;;
    --manual)
      MANUAL_MODE="1"
      shift
      ;;
    --reset-config|--no-cache|--no-fallback)
      SMART_ARGS+=("$1")
      shift
      ;;
    --gpus|--max-util|--min-util|--tp-size|--fallback-steps)
      SMART_ARGS+=("$1" "$2")
      shift 2
      ;;
    -*)
      echo "Unknown option: $1" >&2
      echo "Use --help for usage information" >&2
      exit 1
      ;;
    *)
      # Legacy mode: positional argument is model path
      if [[ -z "$MODEL_INPUT" ]]; then
        MODEL_INPUT="$1"
      fi
      shift
      ;;
  esac
done

if [[ -z "$MODEL_INPUT" ]]; then
  echo "❌ Error: --model is required" >&2
  echo "Use --help for usage information" >&2
  exit 1
fi

# Activate virtual environment
if [[ -d "venv" ]]; then
  source venv/bin/activate
elif [[ -d "../venv" ]]; then
  source ../venv/bin/activate
fi

# Set common environment variables
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"

# Run smart analyzer or fall back to manual mode
if [[ -z "$MANUAL_MODE" && -f "${SCRIPT_DIR}/smart_serve.py" ]]; then
  # Smart mode: use Python analyzer
  CMD=(python "${SCRIPT_DIR}/smart_serve.py" --model "$MODEL_INPUT" --port "$PORT" --host "$HOST")
  
  if [[ -n "$MAX_MODEL_LEN" ]]; then
    CMD+=(--max-model-len "$MAX_MODEL_LEN")
  fi
  
  if [[ -n "$DRY_RUN" ]]; then
    CMD+=(--dry-run)
  fi
  
  # Add smart configuration args
  if [[ ${#SMART_ARGS[@]} -gt 0 ]]; then
    CMD+=("${SMART_ARGS[@]}")
  fi
  
  exec "${CMD[@]}"
else
  # Manual mode: use environment variables (legacy behavior)
  TP_SIZE="${TP_SIZE:-8}"
  GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.75}"
  SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-}"
  
  echo "🔧 Manual mode - using environment variables"
  echo "   TP_SIZE=$TP_SIZE, GPU_MEM_UTIL=$GPU_MEM_UTIL"
  echo
  
  MODEL="$MODEL_INPUT"
  EXTRA_ARGS=()
  
  if [[ -d "$MODEL_INPUT" || -f "$MODEL_INPUT" ]]; then
    MODEL="$(realpath "$MODEL_INPUT")"
    export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
  else
    export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
    export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
  fi
  
  if [[ -n "$SERVED_MODEL_NAME" ]]; then
    EXTRA_ARGS+=(--served-model-name "$SERVED_MODEL_NAME")
  fi
  
  if [[ -n "$MAX_MODEL_LEN" ]]; then
    EXTRA_ARGS+=(--max-model-len "$MAX_MODEL_LEN")
  fi
  
  MODEL_LC="$(echo "$MODEL" | tr '[:upper:]' '[:lower:]')"
  
  case "$MODEL_LC" in
    *qwen3* )
      EXTRA_ARGS+=(--reasoning-parser qwen3)
      ;;
  
    *deepseek-r1*|*deepseek*r1* )
      EXTRA_ARGS+=(--reasoning-parser deepseek_r1)
      ;;
  
    *glm-4.5*|*glm45* )
      EXTRA_ARGS+=(--reasoning-parser glm45)
      ;;
  
    *gemma-4* )
      EXTRA_ARGS+=(--reasoning-parser gemma4)
      ;;
  
    *mistral*|*mistralai*|*ministral*|*magistral* )
      EXTRA_ARGS+=(
        --tokenizer_mode mistral
        --config_format mistral
        --load_format mistral
      )
  
      case "$MODEL_LC" in
        *reasoning*|*magistral*|*think* )
          EXTRA_ARGS+=(--reasoning-parser mistral)
          ;;
      esac
      ;;
  esac
  
  python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --tensor-parallel-size "$TP_SIZE" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    "${EXTRA_ARGS[@]}"
fi
