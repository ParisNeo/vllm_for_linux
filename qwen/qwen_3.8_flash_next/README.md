# Qwen 3.8 Flash Next (FP8) - vLLM Deployment Bundle

This directory contains optimized scripts to deploy **Qwen 3.8 Flash Next (FP8) (Text + Vision)**, **Qwen Image Edit**, and **Wan 2.2 Video Generation** simultaneously on a single machine using vLLM. 

The architecture is specifically tuned for a **4x A100 (40GB)** setup, utilizing Tensor Parallelism for the LLM and isolating the diffusion pipelines on dedicated GPUs. The Flash Next model is loaded in native FP8 quantization to optimize VRAM usage and inference speed.

## 📂 Directory Architecture

- `download_hf.py` / `download_hf.sh`: Utility to download model snapshots from Hugging Face.
- `serve_qwen_3.8_4x_A100.sh`: Launches the Qwen 3.8 Flash Next Multimodal LLM on GPUs 0 and 1 (Tensor Parallel = 2).
- `serve_qwen_image_edit_A100.sh`: Launches the Qwen Image Edit pipeline on GPU 3.
- `serve_wan_2.2_5B_A100.sh`: Launches the Wan 2.2 TI2V-5B Video Generation pipeline on GPU 2.
- `serve_bundle_4x_A100.sh`: Orchestrator that launches all services dynamically in the background.

## 🗺️ GPU Allocation Strategy

The bundle is designed to fully utilize the 4x A100 layout, assigning distinct hardware targets to each service to prevent VRAM conflicts.

| Service | GPUs Used | VRAM Target | Port |
| :--- | :--- | :--- | :--- |
| **Qwen 3.8 Flash Next (LLM/VLM)** | 0, 1 (TP=2) | ~92% Utilization | `8000` |
| **Wan 2.2 Video** | 2 (TP=1) | ~85% Utilization | `8002` |
| **Qwen Image Edit** | 3 (TP=1) | ~85% Utilization | `8001` |

## 🚀 Quick Start

### 1. Prerequisites
Ensure you have activated the global Python virtual environment located at the project root:
```bash
source ../../venv/bin/activate
```
*(All shell scripts automatically attempt to source this environment, but doing it manually ensures `vllm` and `huggingface_hub` are in your PATH).*

### 2. Download Models
Models are downloaded into the script-relative `qwen/qwen_3.8_flash_next/models/` directory. The downloader automatically excludes legacy weight formats (`.pth`, `.pt`, `.bin`) to save disk space.

```bash
# Make the downloader executable if needed
chmod +x download_hf.sh

# Download Qwen 3.8 Flash Next (FP8)
./download_hf.sh Qwen/Qwen3.8-Flash-Next-FP8

# Download Qwen Image Edit
./download_hf.sh Qwen/Qwen-Image-Edit-2511

# Download Wan 2.2 Video
./download_hf.sh Wan-AI/Wan2.2-TI2V-5B-Diffusers
```

### 3. Run the Bundle
To launch services simultaneously in the background, use the `serve_bundle_4x_A100.sh` orchestrator. By default, it launches all available services.

```bash
chmod +x serve_bundle_4x_A100.sh

# Launch everything (Text + Image + Video)
./serve_bundle_4x_A100.sh

# Launch only Text and Image services
./serve_bundle_4x_A100.sh --serve text+image

# Launch only the Video service
./serve_bundle_4x_A100.sh --serve video
```

**Monitoring:**
- Follow LLM logs: `tail -f vllm_qwen3.8.log`
- Follow Image Edit logs: `tail -f vllm_image_edit.log`
- Follow Video logs: `tail -f vllm_wan_video.log`

**Stopping:**
The bundle script outputs the `kill` command with the PIDs of the launched background processes. To stop them, simply copy and paste the printed `kill` command.

## ⚙️ Advanced Usage

If you wish to run the services individually for debugging or specific configurations, you can execute them directly.

### Running Qwen 3.8 Flash Next Only
```bash
./serve_qwen_3.8_4x_A100.sh
```
*Overrides:*
```bash
# Change port and host
./serve_qwen_3.8_4x_A100.sh --host 0.0.0.0 --port 9000

# Specify a custom model path
./serve_qwen_3.8_4x_A100.sh /path/to/custom/model
```

### Running Image Edit Only
```bash
./serve_qwen_image_edit_A100.sh
```
*Overrides:*
```bash
# Change port
./serve_qwen_image_edit_A100.sh --port 9001
```

### Running Wan Video Only
```bash
./serve_wan_2.2_5B_A100.sh
```
*Overrides:*
```bash
# Change port
./serve_wan_2.2_5B_A100.sh --port 9002
```

## 🛡️ Pre-flight Checks
All serving scripts include automated pre-flight safety checks:
1. **Memory Verification**: Queries `nvidia-smi` to ensure at least `15360 MiB` (15 GB) is free on the target GPUs. If a previous process is hanging, the script will abort safely.
2. **Cache Clearing**: Executes `torch.cuda.empty_cache()` before initialization to prevent memory fragmentation locks.

## 🔧 Environment Variables
The scripts respect the following environment variables for advanced tuning:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `HOST` | `127.0.0.1` | Network interface to bind the server to. |
| `PORT` | `8000` / `8001` / `8002` | Port for the OpenAI-compatible API. |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | PyTorch memory management strategy. |
| `VLLM_RPC_TIMEOUT` | `600` | Timeout for vLLM RPC calls in seconds. |
| `OMP_NUM_THREADS` | `1` | OpenMP threads per process. |