# vllm_for_linux




[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/ParisNeo/vllm_for_linux/blob/main/LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Ubuntu-orange.svg)](https://ubuntu.com/)
[![Python](https://img.shields.io/badge/Python-3.9--3.12-blue.svg)](https://www.python.org/)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](#contributing)
[![Powered by vLLM](https://img.shields.io/badge/Powered%20by-vLLM-success.svg)](https://github.com/vllm-project/vllm)

<img width="1693" height="929" alt="image" src="https://github.com/user-attachments/assets/285ea1be-1950-42be-a198-ae3650c4eda5" />



A practical Linux setup toolkit for serving LLMs with **vLLM**, downloading models from Hugging Face, validating CUDA, and launching many model families from either local paths or Hub repositories.

> By ParisNeo

## What it does

`vllm_for_linux` is a small utility bundle designed to make local and server-side vLLM deployment less painful on Linux.

It helps with:

- Bootstrapping a clean Python environment.
- Installing `vllm`, `huggingface_hub`, and `ascii-colors`.
- Downloading models from Hugging Face into local folders.
- Validating NVIDIA driver, CUDA, PyTorch, and GPU visibility.
- Serving models from local paths or directly from the Hub.
- Supporting model-specific startup flags for families like Qwen, DeepSeek, and Mistral.

## Workflow

The project follows a simple flow:

1. Install the environment with `install.sh`.
2. Validate CUDA with `test_cuda.py`.
3. Download a model snapshot locally.
4. Start the vLLM OpenAI-compatible server.
5. Connect your client app to the exposed API endpoint.

## Project layout

```text
vllm_for_linux/
├── install.sh
├── test_cuda.py
├── download_model.py
├── serve_model.sh
└── README.md
```

## Features

| Feature | Description |
|---|---|
| Environment bootstrap | Uses `uv` when available and can fall back to `python -m venv` when needed. |
| Python handling | Prefers a managed Python runtime through `uv` for better reproducibility. |
| CUDA validation | Checks `nvidia-smi`, `libcuda`, PyTorch CUDA support, GPU count, compute capability, and a real CUDA matmul test. |
| Hugging Face downloads | Downloads full model snapshots into deterministic local directories. |
| Token guidance | Explains how to configure `HF_TOKEN` for gated models, higher limits, and more reliable downloads. |
| Model launcher | Supports local model paths and Hub repo IDs. |
| Family-aware serving | Can attach model-specific options such as reasoning parsers and Mistral loader settings. |
| CLI-friendly UX | Uses `ascii-colors` banners and readable console diagnostics. |

## Requirements

- Linux
- NVIDIA GPU and driver
- CUDA-compatible PyTorch runtime
- Python 3.10+ for fallback mode, with `uv` preferred for managing the right Python version
- Internet access for first-time package/model downloads, unless you work from pre-downloaded local assets

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
- Installs all required Python packages.
- Runs a CUDA validation pass at the end.

### Manual environment setup

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -U vllm huggingface_hub ascii-colors
```

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

If something is broken, the script prints targeted advice instead of failing silently.

## Downloading models

```bash
source venv/bin/activate
python download_model.py --model mistralai/Mistral-7B-Instruct-v0.2 --dir models
```

### With authentication

```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxx"
python download_model.py --model meta-llama/Llama-3.1-8B-Instruct --dir models
```

### Dry run

```bash
python download_model.py --model Qwen/Qwen3-32B --dry-run
```

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

## Supported model families

| Family | Notes |
|---|---|
| Qwen | Can use Qwen reasoning parser for supported reasoning-capable variants. |
| DeepSeek | Can attach DeepSeek R1 reasoning parser when applicable. |
| Mistral | Can use Mistral tokenizer/config/load settings for Mistral-family models. |
| Local mirrored models | Works with absolute local paths for offline or air-gapped setups. |

## Hugging Face token setup

Public models can often be downloaded without authentication, but using a token is recommended.

Why use `HF_TOKEN`:

- Better rate limits
- More reliable downloads
- Access to gated/private repositories you are authorized for
- Cleaner automation on servers

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

## Example API launch

```bash
python -m vllm.entrypoints.openai.api_server \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --host 127.0.0.1 \
  --port 8000
```

## Troubleshooting

| Problem | Likely cause | What to check |
|---|---|---|
| `nvidia-smi` fails | Driver missing or broken | Reinstall or repair NVIDIA drivers |
| `torch.cuda.is_available()` is `False` | CUDA runtime mismatch, wrong wheel, hidden devices | Check driver, PyTorch build, `CUDA_VISIBLE_DEVICES` |
| Zero visible GPUs | Masked devices or missing passthrough | Check environment variables, container runtime, VM GPU mapping |
| CUDA test fails | Driver/runtime incompatibility | Recreate venv and validate CUDA stack again |
| Model download denied | Missing token or missing access approval | Set `HF_TOKEN` and accept model terms on Hugging Face |
| vLLM startup fails on model family quirks | Missing family-specific arguments | Extend `serve_model.sh` preset logic |

## Roadmap

- Add more family presets such as Llama, Phi, Granite, and EXAONE.
- Add YAML-based model presets.
- Add health checks and structured logs for the server launcher.
- Add a generated architecture infographic for the README.

## Credits

Created by **ParisNeo**.

## License

This project is licensed under the **Apache 2.0 License**. See the [LICENSE](LICENSE) file for details.
