# GLM-5.3 (Native FP8) - vLLM Deployment

This directory contains optimized scripts to deploy **GLM-5.3 (zai-org/GLM-5.3)** using vLLM on an 8x H200 setup. The model is loaded using native FP8 weights and FP8 KV cache for maximum memory efficiency and throughput.

## 📂 Directory Architecture

- `download_hf.py` / `download_hf.sh`: Utility to download model snapshots from Hugging Face.
- `serve_glm_5.3_8x_H200.sh`: Launches the GLM-5.3 LLM on GPUs 0-7 (Tensor Parallel = 8).

## 🚀 Quick Start

### 1. Prerequisites
Ensure you have activated the global Python virtual environment located at the project root:
```bash
source ../../venv/bin/activate
```
*(All shell scripts automatically attempt to source this environment, but doing it manually ensures `vllm` and `huggingface_hub` are in your PATH).*

### 2. Download Model
The model is downloaded into the script-relative `glm/glm_5.3/models/` directory. The downloader automatically excludes legacy weight formats (`.pth`, `.pt`, `.bin`) to save disk space.

```bash
# Make the downloader executable if needed
chmod +x download_hf.sh

# Download GLM-5.3 (Native FP8)
./download_hf.sh zai-org/GLM-5.3
```

### 3. Run the Server

```bash
chmod +x serve_glm_5.3_8x_H200.sh

# Launch with defaults (0.0.0.0:8000, TP=8, FP8)
./serve_glm_5.3_8x_H200.sh
```

*Overrides:*
```bash
# Change port and host
./serve_glm_5.3_8x_H200.sh --host 0.0.0.0 --port 9000

# Specify a custom model path
./serve_glm_5.3_8x_H200.sh /path/to/custom/model
```

## 🛡️ Pre-flight Checks
The serving script includes automated pre-flight safety checks:
1. **Memory Verification**: Queries `nvidia-smi` to ensure at least `15360 MiB` (15 GB) is free on the target GPUs. If a previous process is hanging, the script will abort safely.
2. **Cache Clearing**: Executes `torch.cuda.empty_cache()` before initialization to prevent memory fragmentation locks.

## 🔧 Environment Variables
The scripts respect the following environment variables for advanced tuning:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `HOST` | `0.0.0.0` | Network interface to bind the server to. |
| `PORT` | `8000` | Port for the OpenAI-compatible API. |
| `TP_SIZE` | `8` | Tensor Parallel size. |
| `GPU_MEM_UTIL` | `0.90` | vLLM GPU memory utilization limit. |
| `KV_CACHE_DTYPE` | `fp8` | KV Cache quantization format. |
| `QUANTIZATION` | `fp8` | Model weight quantization format. |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | PyTorch memory management strategy. |
| `VLLM_RPC_TIMEOUT` | `600` | Timeout for vLLM RPC calls in seconds. |
| `OMP_NUM_THREADS` | `1` | OpenMP threads per process. |