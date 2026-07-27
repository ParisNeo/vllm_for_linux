# Qwen 3.6 - vLLM Deployment Bundle

This directory contains optimized scripts to deploy **Qwen 3.6 (Text + Vision)** and **Qwen Image Edit** simultaneously on a single machine using vLLM. 

The architecture is specifically tuned for a **4x A100 (40GB)** setup, utilizing Tensor Parallelism for the LLM and isolating the diffusion pipeline on a dedicated GPU.

## 📂 Directory Architecture

- `download_hf.py` / `download_hf.sh`: Utility to download model snapshots from Hugging Face.
- `serve_qwen_3.6_4x_A100.sh`: Launches the Qwen 3.6 Multimodal LLM on GPUs 0 and 1 (Tensor Parallel = 2).
- `serve_qwen_image_edit_A100.sh`: Launches the Qwen Image Edit pipeline on GPU 3.
- `serve_bundle_4x_A100.sh`: Orchestrator that launches both services in the background.

## 🗺️ GPU Allocation Strategy

The bundle is designed to leave GPU 2 completely free for other workloads.

| Service | GPUs Used | VRAM Target | Port |
| :--- | :--- | :--- | :--- |
| **Qwen 3.6 (LLM/VLM)** | 0, 1 (TP=2) | ~82% Utilization | `8000` |
| **Qwen Image Edit** | 3 (TP=1) | ~85% Utilization | `8001` |
| *Free* | 2 | 0% | - |

## 🚀 Quick Start

### 1. Prerequisites
Ensure you have activated the global Python virtual environment located at the project root:
```bash
source ../../venv/bin/activate
```
*(All shell scripts automatically attempt to source this environment, but doing it manually ensures `vllm` and `huggingface_hub` are in your PATH).*

### 2. Download Models
Models are downloaded into the script-relative `qwen/qwen_3.6/models/` directory. The downloader automatically excludes legacy weight formats (`.pth`, `.pt`, `.bin`) to save disk space.

```bash
# Make the downloader executable if needed
chmod +x download_hf.sh

# Download Qwen 3.6 (Text + Vision)
./download_hf.sh Qwen/Qwen3.6-27B

# Download Qwen Image Edit
./download_hf.sh Qwen/Qwen-Image-Edit-2511
```

### 3. Run the Bundle
To launch both services simultaneously in the background:

```bash
chmod +x serve_bundle_4x_A100.sh
./serve_bundle_4x_A100.sh
```

**Monitoring:**
- Follow LLM logs: `tail -f vllm_qwen3.6.log`
- Follow Image Edit logs: `tail -f vllm_image_edit.log`

**Stopping:**
The bundle script outputs the PIDs of the background processes. To stop them:
```bash
kill <PID_TEXT> <PID_IMAGE>
```

## ⚙️ Advanced Usage

If you wish to run the services individually for debugging or specific configurations, you can execute them directly.

### Running Qwen 3.6 Only
```bash
./serve_qwen_3.6_4x_A100.sh
```
*Overrides:*
```bash
# Change port and host
./serve_qwen_3.6_4x_A100.sh --host 0.0.0.0 --port 9000

# Specify a custom model path
./serve_qwen_3.6_4x_A100.sh /path/to/custom/model
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

## 🛡️ Pre-flight Checks
Both serving scripts include automated pre-flight safety checks:
1. **Memory Verification**: Queries `nvidia-smi` to ensure at least `15360 MiB` (15 GB) is free on the target GPUs. If a previous process is hanging, the script will abort safely.
2. **Cache Clearing**: Executes `torch.cuda.empty_cache()` before initialization to prevent memory fragmentation locks.

## 🔧 Environment Variables
The scripts respect the following environment variables for advanced tuning:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `HOST` | `127.0.0.1` | Network interface to bind the server to. |
| `PORT` | `8000` (LLM) / `8001` (Edit) | Port for the OpenAI-compatible API. |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | PyTorch memory management strategy. |
| `VLLM_RPC_TIMEOUT` | `600` | Timeout for vLLM RPC calls in seconds. |
| `OMP_NUM_THREADS` | `1` | OpenMP threads per process. |
