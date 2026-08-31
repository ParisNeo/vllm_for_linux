# GLM-5.3-Flash - vLLM Deployment Bundle

This directory contains optimized scripts to deploy **GLM-5.3-Flash (FP8)** using vLLM. 

The architecture is specifically tuned for an **8x H200** setup, utilizing Tensor Parallelism to distribute the massive 320B (18B active) Mixture-of-Experts model across the GPUs.

## 📂 Directory Architecture

- `download_hf.py` / `download_hf.sh`: Utility to download model snapshots from Hugging Face.
- `serve_glm_5.3_8x_H200.sh`: Launches the GLM-5.3-Flash LLM on all 8 GPUs (Tensor Parallel = 8).

## 🗺️ GPU Allocation Strategy

The bundle is designed to fully utilize the 8x H200 layout, assigning all hardware targets to the LLM to maximize throughput and context window capacity.

| Service | GPUs Used | VRAM Target | Port |
| :--- | :--- | :--- | :--- |
| **GLM-5.3-Flash (LLM/VLM)** | 0, 1, 2, 3, 4, 5, 6, 7 (TP=8) | ~90% Utilization | `8000` |

## 🚀 Quick Start

### 1. Prerequisites
Ensure you have activated the global Python virtual environment located at the project root:
```bash
source ../../venv/bin/activate
```
*(All shell scripts automatically attempt to source this environment, but doing it manually ensures `vllm` and `huggingface_hub` are in your PATH).*

### 2. Download Models
Models are downloaded into the script-relative `glm/glm_5.3/models/` directory. The downloader automatically excludes legacy weight formats (`.pth`, `.pt`, `.bin`) to save disk space.

```bash
# Make the downloader executable if needed
chmod +x download_hf.sh

# Download GLM-5.3-Flash (FP8)
./download_hf.sh unsloth/GLM-5.3-Flash-FP8
```

### 3. Run the Server
To launch the LLM:

```bash
chmod +x serve_glm_5.3_8x_H200.sh
./serve_glm_5.3_8x_H200.sh
```

*Overrides:*
```bash
# Change port and host
./serve_glm_5.3_8x_H200.sh --host 0.0.0.0 --port 9000

# Specify a custom model path
./serve_glm_5.3_8x_H200.sh /path/to/custom/model
```

## ⚙️ Advanced Usage

### Environment Variables
The scripts respect the following environment variables for advanced tuning:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `HOST` | `0.0.0.0` | Network interface to bind the server to. |
| `PORT` | `8000` | Port for the OpenAI-compatible API. |
| `TP_SIZE` | `8` | Tensor Parallel size (number of GPUs to split across). |
| `GPU_MEM_UTIL` | `0.90` | Target VRAM utilization fraction. |
| `MAX_MODEL_LEN` | `auto` | Maximum context length. |
| `KV_CACHE_DTYPE` | `fp8` | Data type for KV cache (optimized for FP8 on H200). |
| `DTYPE` | `bfloat16` | Compute data type. |
| `QUANTIZATION` | `compressed-tensors` | Quantization format. |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | PyTorch memory management strategy. |
| `VLLM_RPC_TIMEOUT` | `600` | Timeout for vLLM RPC calls in seconds. |
| `OMP_NUM_THREADS` | `1` | OpenMP threads per process. |