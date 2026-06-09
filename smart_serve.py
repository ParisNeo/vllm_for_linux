#!/usr/bin/env python3
"""
Smart Model Server - Intelligent vLLM Configuration Analyzer

Automatically detects GPU resources, calculates optimal tensor parallelism,
configures memory utilization for stable model serving, and caches working
configurations to avoid re-optimization.

Features:
- GPU memory analysis with per-GPU granularity
- Model size estimation from config.json or architecture patterns
- Multimodal awareness (adjusts memory for vision encoders)
- Configuration caching (~/.smart_serve_cache.json)
- Progressive fallback on OOM (reduce util by 10% increments)
- User constraints: GPU selection, utilization bounds, TP hints
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

CACHE_FILE = Path.home() / ".smart_serve_cache.json"

def get_gpu_info() -> List[Dict]:
    """Query GPU information using nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.total,memory.free,name", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        gpus = []
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                gpus.append({
                    "index": int(parts[0]),
                    "total_mb": int(parts[1]),
                    "free_mb": int(parts[2]),
                    "name": parts[3]
                })
        return gpus
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️  Warning: nvidia-smi not available. Assuming single GPU with 24GB.", file=sys.stderr)
        return [{"index": 0, "total_mb": 24576, "free_mb": 20000, "name": "Unknown"}]

def filter_gpus(gpus: List[Dict], gpu_ids: Optional[List[int]]) -> List[Dict]:
    """Filter GPUs by user-specified IDs."""
    if gpu_ids is None:
        return gpus
    return [g for g in gpus if g["index"] in gpu_ids]

def estimate_model_size(model_path: str) -> Tuple[float, float, bool]:
    """
    Estimate model size in GB from config.json or model name.
    Returns (params_billions, estimated_memory_gb, is_multimodal)
    """
    config_path = Path(model_path) / "config.json"
    is_multimodal = False
    
    # Try to read config.json
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
            
            # Extract parameter count if available
            if "num_parameters" in config:
                params_b = config["num_parameters"] / 1e9
            else:
                # Estimate from architecture
                hidden_size = config.get("hidden_size", 4096)
                num_layers = config.get("num_hidden_layers", 32)
                vocab_size = config.get("vocab_size", 32000)
                intermediate_size = config.get("intermediate_size", hidden_size * 4)
                
                # Rough estimate: params ≈ 12 * hidden^2 * layers (for transformer)
                params_b = (12 * hidden_size * hidden_size * num_layers) / 1e9
            
            # Check for multimodal indicators
            multimodal_keywords = ["vision", "image", "clip", "siglip", "paligemma", "llava", "fuyu", "idefics"]
            arch = config.get("architectures", [])
            model_type = config.get("model_type", "")
            is_multimodal = any(
                keyword in model_path.lower() or 
                keyword in str(arch).lower() or 
                keyword in model_type.lower()
                for keyword in multimodal_keywords
            )
            
            # Memory estimate: params * 2 bytes (BF16) + overhead
            base_memory_gb = params_b * 2
            
            # Add multimodal overhead (vision encoder + image embeddings)
            if is_multimodal:
                # Typical vision encoder: 300M-1B params = 0.6-2GB
                # Image embeddings cache: 2-4GB depending on resolution
                base_memory_gb += 4.0  # Conservative multimodal overhead
                print(f"   🖼️  Multimodal model detected - adding 4GB vision overhead")
            
            # Add 20% overhead for KV cache and activations
            memory_gb = base_memory_gb * 1.2
            return params_b, memory_gb, is_multimodal
            
        except Exception as e:
            print(f"⚠️  Warning: Could not parse config.json: {e}", file=sys.stderr)
    
    # Fallback: estimate from model name
    model_name = model_path.lower()
    is_multimodal = any(kw in model_name for kw in ["vision", "image", "llava", "paligemma", "fuyu"])
    
    if "72b" in model_name or "70b" in model_name:
        params_b, memory_gb = 72, 172.8
    elif "35b" in model_name:
        params_b, memory_gb = 35, 84
    elif "31b" in model_name or "33b" in model_name:
        params_b, memory_gb = 33, 79.2
    elif "14b" in model_name:
        params_b, memory_gb = 14, 33.6
    elif "8b" in model_name or "7b" in model_name:
        params_b, memory_gb = 8, 19.2
    elif "4b" in model_name or "3b" in model_name:
        params_b, memory_gb = 4, 9.6
    elif "1b" in model_name or "0.5b" in model_name:
        params_b, memory_gb = 1, 2.4
    else:
        params_b, memory_gb = 8, 19.2
    
    if is_multimodal:
        memory_gb += 4.0
        print(f"   🖼️  Multimodal model detected - adding 4GB vision overhead")
    
    return params_b, memory_gb, is_multimodal

def get_model_hash(model_path: str) -> str:
    """Generate a hash for model identification in cache."""
    # Use path + config.json mtime for cache key
    config_path = Path(model_path) / "config.json"
    if config_path.exists():
        mtime = str(config_path.stat().st_mtime)
    else:
        mtime = str(time.time())
    
    content = f"{model_path}:{mtime}"
    return hashlib.md5(content.encode()).hexdigest()[:12]

def load_cache() -> Dict:
    """Load configuration cache from disk."""
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE) as f:
                return json.load(f)
        except:
            pass
    return {"models": {}}

def save_cache(cache: Dict):
    """Save configuration cache to disk."""
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)

def get_cached_config(model_path: str) -> Optional[Dict]:
    """Retrieve cached configuration for a model."""
    cache = load_cache()
    model_hash = get_model_hash(model_path)
    
    if model_hash in cache.get("models", {}):
        cached = cache["models"][model_hash]
        # Check if cache is recent (within 7 days)
        age_days = (time.time() - cached.get("last_used", 0)) / 86400
        if age_days < 7:
            print(f"✅ Using cached configuration (last used {age_days:.1f} days ago)")
            print(f"   TP_SIZE={cached['tp_size']}, GPU_MEM_UTIL={cached['gpu_mem_util']:.2f}")
            return cached
    return None

def save_successful_config(model_path: str, tp_size: int, gpu_mem_util: float, context_len: int):
    """Save a working configuration to cache."""
    cache = load_cache()
    model_hash = get_model_hash(model_path)
    
    cache["models"][model_hash] = {
        "tp_size": tp_size,
        "gpu_mem_util": gpu_mem_util,
        "context_len": context_len,
        "success_count": cache["models"].get(model_hash, {}).get("success_count", 0) + 1,
        "last_used": time.time(),
        "model_path": model_path
    }
    
    save_cache(cache)
    print(f"💾 Configuration saved to cache ({CACHE_FILE})")

def reset_config(model_path: str):
    """Remove cached configuration for a model."""
    cache = load_cache()
    model_hash = get_model_hash(model_path)
    
    if model_hash in cache.get("models", {}):
        del cache["models"][model_hash]
        save_cache(cache)
        print(f"🗑️  Configuration reset for {model_path}")
    else:
        print(f"ℹ️  No cached configuration found for {model_path}")

def calculate_optimal_config(
    gpus: List[Dict], 
    model_memory_gb: float, 
    model_path: str,
    max_util: float = 0.90,
    min_util: float = 0.50,
    tp_hint: Optional[int] = None
) -> Tuple[int, float]:
    """
    Calculate optimal TP_SIZE and GPU_MEM_UTIL based on available resources.
    
    Args:
        gpus: List of GPU info dicts
        model_memory_gb: Estimated model memory requirement
        model_path: Path for logging
        max_util: Maximum allowed GPU memory utilization
        min_util: Minimum allowed GPU memory utilization
        tp_hint: User-suggested tensor parallelism
    
    Returns:
        (tp_size, gpu_mem_util)
    """
    num_gpus = len(gpus)
    
    if num_gpus == 0:
        print("❌ Error: No GPUs available", file=sys.stderr)
        sys.exit(1)
    
    # Calculate average free memory per GPU (in GB)
    avg_free_gb = sum(g["free_mb"] for g in gpus) / num_gpus / 1024
    avg_total_gb = sum(g["total_mb"] for g in gpus) / num_gpus / 1024
    
    # Minimum memory per GPU for model weights (with TP)
    kv_overhead_gb = 4  # Reserve 4GB per GPU for KV cache and overhead
    
    # If user provided TP hint, try it first
    if tp_hint is not None and tp_hint <= num_gpus:
        memory_per_gpu_needed = (model_memory_gb / tp_hint) + kv_overhead_gb
        required_util = memory_per_gpu_needed / avg_total_gb
        
        if required_util <= max_util and memory_per_gpu_needed <= avg_free_gb * 0.90:
            print(f"   Using user-specified TP_SIZE={tp_hint}")
            return tp_hint, min(required_util, max_util)
        else:
            print(f"⚠️  TP_SIZE={tp_hint} not feasible with current memory", file=sys.stderr)
    
    # Try different TP sizes from num_gpus down to 1
    best_tp = num_gpus
    best_util = 0.0
    
    for tp in range(num_gpus, 0, -1):
        memory_per_gpu_needed = (model_memory_gb / tp) + kv_overhead_gb
        
        # Calculate what utilization this would require
        required_util = memory_per_gpu_needed / avg_total_gb
        
        # Check if feasible with current free memory (with 10% safety margin)
        available_with_margin = avg_free_gb * 0.90
        
        if memory_per_gpu_needed <= available_with_margin and required_util <= max_util:
            # This TP size works, calculate actual utilization
            actual_util = max(min_util, min(required_util, max_util))
            
            best_tp = tp
            best_util = actual_util
            break
    else:
        # Even TP=num_gpus doesn't fit, use all GPUs with max safe utilization
        best_tp = num_gpus
        best_util = min(max_util, avg_free_gb * 0.90 / avg_total_gb)
        
        if best_util < min_util:
            print("⚠️  Warning: Insufficient GPU memory. Model may not fit.", file=sys.stderr)
            print(f"   Available: {avg_free_gb:.1f} GB free/GPU, Need: {model_memory_gb / best_tp:.1f} GB/GPU", file=sys.stderr)
    
    return best_tp, best_util

def get_reasoning_parser_args(model_path: str) -> List[str]:
    """Get reasoning parser arguments based on model type."""
    model_lower = model_path.lower()
    args = []
    
    if "gemma-4" in model_lower or "gemma4" in model_lower:
        args.extend(["--reasoning-parser", "gemma4"])
    elif "qwen3" in model_lower:
        args.extend(["--reasoning-parser", "qwen3"])
    elif "deepseek-r1" in model_lower or ("deepseek" in model_lower and "r1" in model_lower):
        args.extend(["--reasoning-parser", "deepseek_r1"])
    elif "glm-4.5" in model_lower or "glm45" in model_lower:
        args.extend(["--reasoning-parser", "glm45"])
    elif "mistral" in model_lower or "magistral" in model_lower:
        args.extend([
            "--tokenizer_mode", "mistral",
            "--config_format", "mistral",
            "--load_format", "mistral",
        ])
        if "reasoning" in model_lower or "magistral" in model_lower or "think" in model_lower:
            args.extend(["--reasoning-parser", "mistral"])
    
    return args

def main():
    parser = argparse.ArgumentParser(description="Smart vLLM Model Server")
    parser.add_argument("--model", required=True, help="Path to model directory or HuggingFace repo ID")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)), help="Server port")
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"), help="Server host")
    parser.add_argument("--max-model-len", type=int, default=None, help="Maximum model context length")
    parser.add_argument("--dry-run", action="store_true", help="Show configuration without launching")
    
    # Configuration control
    parser.add_argument("--reset-config", action="store_true", help="Reset cached configuration and re-optimize")
    parser.add_argument("--no-cache", action="store_true", help="Skip cache, always re-optimize")
    
    # GPU constraints
    parser.add_argument("--gpus", type=str, default=None, help="Comma-separated GPU IDs to use (e.g., '0,1,2,3')")
    parser.add_argument("--max-util", type=float, default=0.90, help="Maximum GPU memory utilization (default: 0.90)")
    parser.add_argument("--min-util", type=float, default=0.50, help="Minimum GPU memory utilization (default: 0.50)")
    parser.add_argument("--tp-size", type=int, default=None, help="Force specific tensor parallelism")
    
    # Fallback control
    parser.add_argument("--no-fallback", action="store_true", help="Disable progressive fallback on OOM")
    parser.add_argument("--fallback-steps", type=int, default=3, help="Number of fallback attempts (default: 3)")
    
    args = parser.parse_args()
    
    print("🔍 Smart Model Server - Analyzing system...")
    print()
    
    # Handle reset-config
    if args.reset_config:
        reset_config(args.model)
    
    # Step 1: Detect GPUs
    all_gpus = get_gpu_info()
    print(f"📊 Detected {len(all_gpus)} GPU(s):")
    for gpu in all_gpus:
        free_gb = gpu["free_mb"] / 1024
        total_gb = gpu["total_mb"] / 1024
        print(f"   GPU {gpu['index']}: {free_gb:.1f} GB free / {total_gb:.1f} GB total ({gpu['name']})")
    print()
    
    # Filter GPUs if specified
    gpu_ids = None
    if args.gpus:
        gpu_ids = [int(x.strip()) for x in args.gpus.split(",")]
        all_gpus = filter_gpus(all_gpus, gpu_ids)
        if len(all_gpus) == 0:
            print(f"❌ Error: No GPUs found matching IDs: {gpu_ids}", file=sys.stderr)
            sys.exit(1)
        print(f"🎯 Using GPUs: {gpu_ids}")
        print()
    
    # Step 2: Analyze model
    model_path = args.model
    if not os.path.exists(model_path):
        print(f"ℹ️  Model not found locally, treating as HuggingFace repo: {model_path}")
    
    params_b, model_memory_gb, is_multimodal = estimate_model_size(model_path)
    print(f"📦 Model: {model_path}")
    print(f"   Estimated parameters: {params_b:.1f}B")
    print(f"   Estimated memory (BF16 + overhead): {model_memory_gb:.1f} GB")
    if is_multimodal:
        print(f"   🖼️  Multimodal: Yes")
    print()
    
    # Step 3: Check cache
    cached_config = None
    if not args.no_cache and not args.reset_config:
        cached_config = get_cached_config(model_path)
    
    # Step 4: Calculate optimal configuration
    if cached_config:
        tp_size = cached_config["tp_size"]
        gpu_mem_util = cached_config["gpu_mem_util"]
    else:
        print("⚙️  Calculating optimal configuration...")
        tp_size, gpu_mem_util = calculate_optimal_config(
            all_gpus, 
            model_memory_gb, 
            model_path,
            max_util=args.max_util,
            min_util=args.min_util,
            tp_hint=args.tp_size
        )
        print(f"   Tensor Parallelism: {tp_size}")
        print(f"   GPU Memory Utilization: {gpu_mem_util:.2f} ({gpu_mem_util * 100:.0f}%)")
        print()
    
    # Step 5: Build command
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model,
        "--host", args.host,
        "--port", str(args.port),
        "--tensor-parallel-size", str(tp_size),
        "--gpu-memory-utilization", f"{gpu_mem_util:.2f}",
    ]
    
    # Add reasoning parser
    cmd.extend(get_reasoning_parser_args(args.model))
    
    if args.max_model_len:
        cmd.extend(["--max-model-len", str(args.max_model_len)])
    
    # Set environment variables for offline mode if local path
    if os.path.exists(args.model):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        print("🔒 Offline mode enabled (local model detected)")
    
    # Set CUDA_VISIBLE_DEVICES
    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    
    print()
    print("🚀 Launch command:")
    print("   " + " ".join(cmd))
    print()
    
    if args.dry_run:
        print("(Dry run - not executing)")
        return 0
    
    # Step 6: Launch with fallback logic
    if not args.no_fallback and not cached_config:
        # Try with fallback
        current_util = gpu_mem_util
        fallback_steps = args.fallback_steps
        
        for attempt in range(fallback_steps + 1):
            print(f"🎯 Starting vLLM server (attempt {attempt + 1}/{fallback_steps + 1})...")
            print(f"   GPU_MEM_UTIL={current_util:.2f}, TP_SIZE={tp_size}")
            print("=" * 60)
            
            # Build command with current util
            cmd_with_util = cmd[:-2] + [f"--gpu-memory-utilization", f"{current_util:.2f}"]
            
            try:
                os.execvp(sys.executable, cmd_with_util)
            except Exception as e:
                if attempt < fallback_steps:
                    current_util *= 0.90  # Reduce by 10%
                    print(f"\n⚠️  Attempt {attempt + 1} failed, reducing utilization to {current_util:.2f}")
                    print()
                else:
                    print(f"\n❌ All {fallback_steps + 1} attempts failed", file=sys.stderr)
                    sys.exit(1)
    else:
        # Direct launch (cached config or no fallback)
        print("🎯 Starting vLLM server...")
        print("=" * 60)
        
        # Save successful config before launching
        save_successful_config(args.model, tp_size, gpu_mem_util, args.max_model_len or 8192)
        
        os.execvp(sys.executable, cmd)

if __name__ == "__main__":
    sys.exit(main())
