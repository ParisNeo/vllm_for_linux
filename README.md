# vllm_for_linux




[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/ParisNeo/vllm_for_linux/blob/main/LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Ubuntu-orange.svg)](https://ubuntu.com/)
[![Python](https://img.shields.io/badge/Python-3.9--3.12-blue.svg)](https://www.python.org/)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](#contributing)
[![Powered by vLLM](https://img.shields.io/badge/Powered%20by-vLLM-success.svg)](https://github.com/vllm-project/vllm)

<img width="1693" height="929" alt="image" src="https://github.com/user-attachments/assets/285ea1be-1950-42be-a198-ae3650c4eda5" />



A practical Linux setup toolkit for serving LLMs with **vLLM**, downloading models from Hugging Face into deterministic local folders, validating CUDA, and launching many model families from either local paths or Hub repositories.

> By ParisNeo

## Overview

`vllm_for_linux` is a small utility bundle for people who want reproducible, local-first, scriptable vLLM deployments on Linux without constantly rebuilding the same shell snippets.

It is designed for workstation and server environments where you want to:

- control exactly where models are downloaded,
- validate the CUDA stack before wasting time on broken launches,
- run models from either local folders or Hub IDs,
- keep launch scripts readable and adjustable per model family,
- work online for downloads and then switch to offline local serving.

## What it does

The toolkit focuses on a few practical tasks that usually get repeated in every fresh deployment:

- Bootstrapping a clean Python environment.
- Installing `vllm`, `huggingface_hub`, and `ascii-colors`.
- Downloading models from Hugging Face into local folders you choose.
- Validating NVIDIA driver, CUDA, PyTorch, and GPU visibility.
- Serving models from local paths or directly from the Hub.
- Supporting model-specific startup flags for families like Qwen, DeepSeek, and Mistral.
- Making it easier to clear caches and retry clean rebuilds when kernels or compiled artifacts go bad.

## Workflow

The project follows a simple and practical flow:

1. Install the environment with `install.sh`.
2. Validate CUDA with `test_cuda.py`.
3. Download a model snapshot locally into your `models/` directory.
4. Start the vLLM OpenAI-compatible server with the appropriate launcher.
5. Connect your client app, agent, or benchmark tool to the exposed API endpoint.

A common pattern is to download once from the Hub, store the snapshot in a project-controlled folder, and then serve from that local path with offline mode enabled.

## Project layout

```text
vllm_for_linux/
├── install.sh
├── test_cuda.py
├── download_model.py
├── serve_model.sh
├── run_qwen.sh
├── clear.sh
└── README.md
```

## Features

| Feature | Description |
|---|---|
| Environment bootstrap | Uses `uv` when available and can fall back to `python -m venv` when needed. |
| Python handling | Prefers a managed Python runtime through `uv` for better reproducibility. |
| CUDA validation | Checks `nvidia-smi`, `libcuda`, PyTorch CUDA support, GPU count, compute capability, and a real CUDA matmul test. |
| Hugging Face downloads | Downloads full model snapshots into deterministic local directories instead of relying on opaque cache locations. |
| Local-first serving | Lets you serve models from absolute local folders after download, which is useful for offline or controlled deployments. |
| Token guidance | Explains how to configure `HF_TOKEN` for gated models, higher limits, and more reliable downloads. |
| Model launcher | Supports local model paths and Hub repo IDs. |
| Family-aware serving | Can attach model-specific options such as reasoning parsers and Mistral loader settings. |
| Qwen launcher | Includes a Qwen-oriented launcher with multimodal and text-only modes based on the official vLLM recipe. |
| Cache reset support | Makes it easy to clear vLLM, Torch, and related caches so kernels and compiled artifacts rebuild cleanly. |
| CLI-friendly UX | Uses `ascii-colors` banners and readable console diagnostics. |

## Requirements

- Linux
- NVIDIA GPU and working NVIDIA driver
- CUDA-compatible PyTorch runtime
- Python 3.10+ for fallback mode, with `uv` preferred for managing the right Python version
- Internet access for first-time package and model downloads, unless you work only from pre-downloaded local assets
- Sufficient GPU memory for the chosen model and parallelism strategy

For larger models, also ensure that your chosen tensor parallel or data parallel layout matches the number of visible GPUs and that your system has enough VRAM headroom for weights, KV cache, and multimodal encoder overhead.

## Installation

### Recommended

```bash
chmod +x install.sh
./install.sh
```

This installer:

- Uses `uv` if it is available.
- Tries to install `uv` automatically if it is missing.
- Falls back to `python -m venv` when possible.
- Installs the required Python packages.
- Runs a CUDA validation pass at the end.

### Manual environment setup

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -U vllm huggingface_hub ascii-colors
```

If you are pinning versions for stability, this is the right place to freeze `vllm`, `torch`, and `flashinfer` to combinations you have validated on your target machine.

## CUDA validation

After installation, the toolkit runs `test_cuda.py` to verify that the machine is ready.

It checks:

- Python runtime information
- `CUDA_VISIBLE_DEVICES`
- `nvidia-smi`
- NVIDIA driver visibility
- `/dev/nvidia*` devices
- `torch.cuda.is_available()`
- GPU count and compute capability
- Real CUDA tensor computation
- `libcuda.so` visibility through `ldconfig`

If something is broken, the script prints targeted advice instead of failing silently. This helps distinguish between driver issues, PyTorch build problems, hidden GPUs, and library path problems.

## Downloading models

```bash
source venv/bin/activate
python download_model.py --model mistralai/Mistral-7B-Instruct-v0.2 --dir models
```

A key design goal is that model snapshots are downloaded to folders you control rather than being left only inside Hugging Face cache directories. This makes storage management, backup, offline reuse, and cleanup much easier.

### With authentication

```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxx"
python download_model.py --model meta-llama/Llama-3.1-8B-Instruct --dir models
```

### Dry run

```bash
python download_model.py --model Qwen/Qwen3-32B --dir models --dry-run
```

### Local snapshot strategy

A practical workflow is:

1. Download once from the Hub into `models/YourModelName`.
2. Verify the files exist locally.
3. Serve using the local absolute path.
4. Enable offline mode for serving.

That approach avoids unpredictable cache-only behavior and gives full control over where large model files live.

## Serving models

```bash
source venv/bin/activate
./serve_model.sh Qwen/Qwen3-32B
```

### Local model path

```bash
./serve_model.sh /absolute/path/to/models/Qwen__Qwen3-32B
```

### Example environment overrides

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TP_SIZE=4
export PORT=8000
export MAX_MODEL_LEN=32768
./serve_model.sh mistralai/Mistral-Nemo-Instruct-2407
```

The launcher is intended to remain simple while still supporting practical overrides for port, GPU visibility, model length, tensor parallel size, and model-family-specific arguments.

## Qwen multimodal launcher

The project can also include a Qwen-specific launcher such as `run_qwen.sh` based on the official vLLM recipe.

Example local multimodal run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
DP_SIZE=8 \
./run_qwen.sh /data/vllm_for_linux/models/Qwen__Qwen3.5-397B-A17B-FP8
```

Example local text-only mode:

```bash
MODE=text \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
DP_SIZE=8 \
./run_qwen.sh /data/vllm_for_linux/models/Qwen__Qwen3.5-397B-A17B-FP8
```

This split is useful because Qwen3.5 models are multimodal by design, but for pure text serving you may want to skip the vision path and recover memory for more KV cache.

## Supported model families

| Family | Notes |
|---|---|
| Qwen | Can use Qwen reasoning parser for supported reasoning-capable variants and may need multimodal-aware launcher settings. |
| DeepSeek | Can attach DeepSeek R1 reasoning parser when applicable. |
| Mistral | Can use Mistral tokenizer, config, or loader settings for Mistral-family models. |
| Local mirrored models | Works with absolute local paths for offline or air-gapped setups. |
| Other Hub models | Can be added with simple preset logic as long as their vLLM requirements are known. |

## Hugging Face token setup

Public models can often be downloaded without authentication, but using a token is still recommended.

Why use `HF_TOKEN`:

- Better rate limits
- More reliable downloads
- Access to gated or private repositories you are authorized for
- Cleaner automation on servers and CI-style setups

Create a token here:

- [Hugging Face Access Tokens](https://huggingface.co/settings/tokens)

Then either:

```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxx"
```

or:

```bash
hf auth login
```

If you are downloading large gated models, also make sure you have accepted the model terms on the model page before scripting the download.

## Example API launch

```bash
python -m vllm.entrypoints.openai.api_server \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --host 127.0.0.1 \
  --port 8000
```

After the server starts, you can use OpenAI-compatible clients by pointing them at `http://127.0.0.1:8000/v1` or the host and port you configured.

## Cache clearing and rebuilds

When debugging broken kernels, stale compiled artifacts, or incompatible runtime caches, it is often useful to wipe cached state and force a full rebuild.

A helper script such as `clear.sh` can remove:

- `~/.cache/vllm`
- `~/.cache/torch`
- `~/.cache/flashinfer`
- temporary or compiled vLLM artifacts

This is especially useful after changing CUDA, PyTorch, vLLM, or FlashInfer versions.

## Troubleshooting

| Problem | Likely cause | What to check |
|---|---|---|
| `nvidia-smi` fails | Driver missing or broken | Reinstall or repair NVIDIA drivers |
| `torch.cuda.is_available()` is `False` | CUDA runtime mismatch, wrong wheel, or hidden devices | Check driver, PyTorch build, and `CUDA_VISIBLE_DEVICES` |
| Zero visible GPUs | Masked devices or missing passthrough | Check environment variables, container runtime, and VM GPU mapping |
| CUDA test fails | Driver/runtime incompatibility | Recreate the venv and validate the CUDA stack again |
| Model download denied | Missing token or missing access approval | Set `HF_TOKEN` and accept model terms on Hugging Face |
| vLLM startup fails on model family quirks | Missing family-specific arguments | Extend `serve_model.sh` preset logic |
| Startup fails after version changes | Stale compiled kernels or caches | Clear vLLM, Torch, and FlashInfer caches and rebuild |
| Parallelism error at startup | TP or DP size does not match visible GPUs | Check `CUDA_VISIBLE_DEVICES`, `TP_SIZE`, `DP_SIZE`, and model launcher settings |

## Roadmap

- Add more family presets such as Llama, Phi, Granite, and EXAONE.
- Add YAML-based model presets.
- Add health checks and structured logs for the server launcher.
- Add optional Docker-based deployment helpers.
- Add a generated architecture infographic for the README.
- Add version pin presets for known-good `torch` + `vllm` + `flashinfer` combinations.

## Contributing

Contributions are welcome.

Good contribution targets include:

- new model-family presets,
- cleaner install flows,
- better diagnostics,
- version pinning matrices,
- documentation improvements,
- reproducible bugfixes for launch scripts.

If you open a PR, try to include the tested environment details, especially GPU model, CUDA version, PyTorch version, and vLLM version.

## Credits

Created by **ParisNeo**.

## License

This project is licensed under the **Apache 2.0 License**. See the [LICENSE](LICENSE) file for details.
