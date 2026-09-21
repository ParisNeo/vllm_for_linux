# Qwen-Image-2.1 - vLLM-Omni Deployment

This directory contains optimized scripts to deploy **Qwen-Image-2.1** — a unified text-to-image and image-editing pipeline — on a single A100 (40GB) using vLLM-Omni.

One 7.1B single-stream DiT with block-causal attention and an exact cross-step prefix KV cache serves both text-to-image and image-conditioned editing. Prompts and reference images are encoded together by a Qwen3-VL-8B text encoder, so an edit request takes the same serving path as plain text-to-image.

## 📂 Directory Architecture

- `download_hf.py` / `download_hf.sh`: Utility to download the model snapshot from Hugging Face.
- `serve_qwen_image_2.1_A100.sh`: Launches the Qwen-Image-2.1 pipeline on a single isolated GPU.

## 🗺️ GPU Allocation Strategy

| Service | GPUs Used | VRAM Target | Port |
| :--- | :--- | :--- | :--- |
| **Qwen-Image-2.1 (T2I + Edit)** | 3 (TP=1) | ~85% Utilization | `8001` |

BF16 peak memory is ~34 GB at 1024x1024 / 40 steps, which fits a 40GB A100 with headroom. The 16x RGBA autoencoder preserves transparency through the round trip.

## ⚠️ Prerequisites

Support for Qwen-Image-2.1 is **not in a tagged vLLM release** and requires vLLM-Omni nightly with PR [vllm-project/vllm-omni#7759](https://github.com/vllm-project/vllm-omni/pull/7759) applied:

```bash
git clone https://github.com/vllm-project/vllm-omni.git && cd vllm-omni
git fetch origin pull/7759/head:qwen-image-2.1 && git checkout qwen-image-2.1
uv venv --python 3.12 --seed && source .venv/bin/activate
uv pip install vllm==0.29.0 --torch-backend=auto && uv pip install -e .
```

NVIDIA GPUs only.

## 🚀 Quick Start

### 1. Prerequisites
Ensure you have activated the global Python virtual environment located at the project root:
```bash
source ../../venv/bin/activate
```

### 2. Download the Model
The model is downloaded into the script-relative `qwen/qwen_image_2.1/models/` directory. The downloader automatically excludes legacy weight formats (`.pth`, `.pt`, `.bin`) to save disk space.

```bash
chmod +x download_hf.sh
./download_hf.sh Qwen/Qwen-Image-2.1
```

### 3. Run the Server

```bash
chmod +x serve_qwen_image_2.1_A100.sh
./serve_qwen_image_2.1_A100.sh
```

*Overrides:*
```bash
# Change port and host
./serve_qwen_image_2.1_A100.sh --host 0.0.0.0 --port 9001

# Isolate on a different GPU
./serve_qwen_image_2.1_A100.sh --gpu 2

# Specify a custom model path
./serve_qwen_image_2.1_A100.sh /path/to/custom/model
```

## 📡 API Usage

### Text to Image (JSON)
Send JSON to `/v1/images/generations`. The picture comes back base64-encoded at `.data[0].b64_json`.

```bash
curl -X POST http://localhost:8001/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen-Image-2.1",
       "prompt": "A ceramic teapot on a wooden table",
       "size": "1024x1024",
       "num_inference_steps": 40,
       "true_cfg_scale": 1.0,
       "seed": 42}'
```

### Image Editing (multipart form — NOT JSON)
Send a form to `/v1/images/edits`. The picture is a file upload and everything else is a form field. **A JSON body to this endpoint resets the connection.** Repeat `-F image=@...` for up to four reference images.

```bash
curl -X POST http://localhost:8001/v1/images/edits \
  -F image=@plate.png \
  -F 'prompt=Write the words "FRESH BASIL" on this plate in dark green lettering' \
  -F model=Qwen/Qwen-Image-2.1 \
  -F size=1024x1024 \
  -F num_inference_steps=40 \
  -F true_cfg_scale=1.0 \
  -F seed=42
```

### Chat Completions (JSON alternative)
The chat endpoint also accepts pictures and returns the result at `.choices[0].message.content[0].image_url.url`:

```bash
curl -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen-Image-2.1",
       "messages": [{"role": "user", "content": [
         {"type": "image_url", "image_url": {"url": "data:image/png;base64,<BASE64>"}},
         {"type": "text", "text": "Write the words \"FRESH BASIL\" on this plate"}
       ]}],
       "modalities": ["image"],
       "extra_body": {"size": "1024x1024", "num_inference_steps": 40,
                      "true_cfg_scale": 1.0, "seed": 42}}'
```

## 🧠 Behavior Notes

- **Sampling defaults are per-request only**: The vLLM-Omni serve CLI does not expose `--num-inference-steps` / `--cfg-scale` (they exist only in the offline inference examples). The server's own defaults are 50 steps and CFG 4.0, which do **not** match this checkpoint — every client must send `num_inference_steps: 40` and `true_cfg_scale: 1.0` in the request body.
- **CFG**: `true_cfg_scale` above 1 only engages alongside a `negative_prompt`; otherwise it is ignored with a warning. When it engages, the DiT runs twice per step, roughly doubling latency.
- **Prefix KV cache**: On automatically and rebuilt per generation. Repeating a prompt costs about a third of the first submission; four reference images leave only about a fifth of the sequence to recompute.
- **CUDA graphs**: Fixed-shape decode steps are captured automatically. Expect the first request after startup to cost about 1.3x a later one.
- **Reference images**: Up to 4 per request (a fifth is rejected with a 400). Each is resized independently to ~1024x1024 of area and tagged `<image1>`, `<image2>` in order.
- **RGBA**: Pass `--color-format RGBA` in offline inference to preserve transparency.
- **Unsupported**: `cache_dit` / `tea_cache` backends conflict with the model's own prefix cache and must not be used.

## 🛡️ Pre-flight Checks
The serving script includes automated pre-flight safety checks:
1. **Memory Verification**: Queries `nvidia-smi` to ensure at least `15360 MiB` (15 GB) is free on the target GPU. If a previous process is hanging, the script will abort safely.
2. **Cache Clearing**: Executes `torch.cuda.empty_cache()` before initialization to prevent memory fragmentation locks.

## 🔧 Environment Variables

| Variable | Default | Description |
| :--- | :--- | :--- |
| `HOST` | `127.0.0.1` | Network interface to bind the server to. |
| `PORT` | `8001` | Port for the OpenAI-compatible API. |
| `TARGET_GPU` | `3` | Physical GPU to isolate the pipeline on. |
| `GPU_MEM_UTIL` | `0.85` | vLLM GPU memory utilization limit. |
| `MAX_MODEL_LEN` | `4096` | Maximum text-encoder context length. |
| `PREFIX_KV_CACHE_DTYPE` | *(empty)* | Opt-in FP8 prefix KV cache: `fp8_v` (~41dB PSNR) or `fp8` (~35dB). Forces eager decode. |
| `ENABLE_STEP_EXECUTION` | `0` | Set to `1` for step-level continuous batching with `--max-num-seqs`. |
| `MAX_NUM_SEQS` | `8` | Concurrent request cap when step execution is enabled. |
| `ENFORCE_EAGER` | `0` | Set to `1` to disable CUDA graph capture. |
| `MIN_FREE_MB` | `15360` | Pre-flight minimum free VRAM in MiB. |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | PyTorch memory management strategy. |
| `VLLM_RPC_TIMEOUT` | `600` | Timeout for vLLM RPC calls in seconds. |
| `OMP_NUM_THREADS` | `1` | OpenMP threads per process. |