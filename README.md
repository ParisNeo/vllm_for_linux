# vllm_for_linux

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/ParisNeo/vllm_for_linux/blob/main/LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Ubuntu-orange.svg)](https://ubuntu.com/)
[![Python](https://img.shields.io/badge/Python-3.9--3.12-blue.svg)](https://www.python.org/)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](#contributing)
[![Powered by vLLM](https://img.shields.io/badge/Powered%20by-vLLM-success.svg)](https://github.com/vllm-project/vllm)

<img width="1693" height="929" alt="image" src="https://github.com/user-attachments/assets/285ea1be-1950-42be-a198-ae3650c4eda5" />

## 🚀 Overview

`vllm_for_linux` is an intelligent model serving toolkit for Linux that automatically analyzes your GPU cluster, calculates optimal configurations, and generates reproducible runner scripts for production deployments.

### Key Features

| Feature | Description |
|---------|-------------|
| **Smart GPU Analysis** | Auto-detects free/busy GPUs and excludes occupied ones |
| **Auto-Optimization** | Calculates optimal TP_SIZE and GPU memory utilization |
| **Configuration Caching** | Saves working configs for 7 days to skip re-optimization |
| **Auto-Runner Generation** | Creates bash scripts with baked-in settings on success |
| **Multi-Server Support** | Run multiple models on different ports/GPUs simultaneously |
| **Fallback Retry Logic** | Progressively reduces memory utilization on OOM errors |
| **Multimodal Detection** | Auto-enables vision features for multimodal models |
| **Reasoning Parser Auto-Detect** | Applies correct parser for Gemma-4, Qwen3, DeepSeek-R1, etc. |

---

## 📦 Installation

```bash
chmod +x install.sh
./install.sh
```

This will:
- Create a Python virtual environment
- Install vLLM and dependencies
- Validate your CUDA setup
- Run a GPU health check

---

## 🎯 Quick Start

```bash
# 1. Serve a model (auto-optimizes and creates runner)
bash serve_model.sh --model models/google__gemma-4-31B-it/

# 2. Run the generated runner for faster startup next time
bash runners/google__gemma-4-31B-it_port8000.sh

# 3. Check what's running
nvidia-smi
```

---

## 🔧 Smart Serving System

### How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│  1. GPU Detection                                               │
│     └─> Query all GPUs via nvidia-smi                          │
│     └─> Filter out busy GPUs (<10 GB free by default)          │
│                                                                 │
│  2. Model Analysis                                              │
│     └─> Read config.json for parameter count                   │
│     └─> Detect multimodal capabilities                         │
│     └─> Estimate memory requirements (BF16 + overhead)         │
│                                                                 │
│  3. Configuration Calculation                                   │
│     └─> Try TP sizes from N GPUs down to 1                     │
│     └─> Use MINIMUM free memory (not average)                  │
│     └─> Add 10% safety margin                                  │
│                                                                 │
│  4. Launch with Fallback                                        │
│     └─> Try optimal config first                               │
│     └─> On OOM: reduce utilization by 10% and retry            │
│     └─> On success: save config to cache + generate runner     │
└─────────────────────────────────────────────────────────────────┘
```

### Command Reference

```bash
bash serve_model.sh --model <path> [options]
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model PATH` | Model directory or HuggingFace repo ID | **Required** |
| `--port PORT` | Server port | `8000` |
| `--host HOST` | Server host | `127.0.0.1` |
| `--name NAME` | Custom runner name | Auto-generated |
| `--gpus IDS` | Comma-separated GPU IDs (e.g., `0,1,2,3`) | Auto-select |
| `--max-util FLOAT` | Maximum GPU memory utilization | `0.90` |
| `--min-free-gb FLOAT` | Minimum free GPU memory to consider GPU available | `10.0` |
| `--tp-size INT` | Force specific tensor parallelism | Auto-calculate |
| `--no-multimodal` | Disable multimodal features | Enabled if detected |
| `--no-reasoning` | Disable reasoning parser | Enabled if detected |
| `--no-fallback` | Disable progressive fallback on OOM | Enabled |
| `--no-auto-runner` | Disable automatic runner generation | Enabled |
| `--reset-config` | Reset cached configuration and re-optimize | Use cache |
| `--dry-run` | Show configuration without launching | Execute |
| `--manual` | Use environment variables instead of auto-config | Smart mode |
| `--help` | Show help message | - |

### Examples

```bash
# Auto-optimize (recommended)
bash serve_model.sh --model models/google__gemma-4-31B-it/

# Custom port and host
bash serve_model.sh --model models/llama-3.1-8b/ --port 8001 --host 0.0.0.0

# Use specific GPUs only
bash serve_model.sh --model models/llama-3.1-8b/ --gpus 4,5,6

# Conservative memory usage
bash serve_model.sh --model models/large-model/ --max-util 0.70

# Disable multimodal for vision model
bash serve_model.sh --model llava-1.5-13b --no-multimodal

# Force re-optimization (ignore cache)
bash serve_model.sh --model models/llama-3.1-8b/ --reset-config

# Dry run to preview configuration
bash serve_model.sh --model models/llama-3.1-8b/ --dry-run

# Manual mode with environment variables
GPU_MEM_UTIL=0.70 TP_SIZE=4 bash serve_model.sh --manual --model models/llama-3.1-8b/
```

---

## 🏃 Runner System

### What Are Runners?

Runners are bash scripts with all settings **baked in** for reproducible, consistent deployments. They're stored in `runners/` (gitignored) and can be executed directly without re-optimization.

### Auto-Generated Runners

When a model server starts successfully, a runner is automatically created:

```bash
bash serve_model.sh --model models/google__gemma-4-31B-it/
# Creates: runners/google__gemma-4-31B-it_port8000.sh
```

### Custom Runner Names

```bash
bash serve_model.sh --model models/google__gemma-4-31B-it/ --name gemma_prod
# Creates: runners/gemma_prod.sh
```

### Generated Runner Example

```bash
#!/usr/bin/env bash
# Runner: gemma_prod
# Model: models/google__gemma-4-31B-it/
# Port: 8000
# Host: 127.0.0.1
# GPUs: 4,5,6,7
# TP Size: 4
# GPU Memory Util: 0.50
# Created: 2026-06-09 16:45:00

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

exec python smart_serve.py \
  --model "models/google__gemma-4-31B-it/" \
  --host "127.0.0.1" \
  --port "8000" \
  --gpus "4,5,6,7" \
  --max-util "0.90" \
  --min-free-gb "10.0" \
  --no-auto-runner
```

### Runner Management

```bash
# List all runners
ls -la runners/*.sh

# Run a specific runner
bash runners/gemma_prod.sh

# Delete a runner
rm runners/old_model.sh

# Regenerate with new settings
bash serve_model.sh --model models/google__gemma-4-31B-it/ --name gemma_prod --port 8005 --reset-config
```

---

## 🖥️ Multi-Server Deployment

### Running Multiple Models Simultaneously

```bash
# Server 1 - Gemma on port 8000 (GPUs 4,5,6,7)
bash serve_model.sh --model models/google__gemma-4-31B-it/ --name gemma_8000 --port 8000 --gpus 4,5,6,7 &

# Server 2 - Llama on port 8001 (GPUs 0,1,2,3)
bash serve_model.sh --model models/llama-3.1-8b/ --name llama_8001 --port 8001 --gpus 0,1,2,3 &

# Check running servers
ps aux | grep vllm

# Stop a specific server
pkill -f "port 8000"
```

### GPU Planning Table

| Server | Model | Port | GPUs | TP Size | Memory Util |
|--------|-------|------|------|---------|-------------|
| 1 | Gemma-4-31B | 8000 | 4,5,6,7 | 4 | 0.50 |
| 2 | Llama-3.1-8B | 8001 | 0,1,2,3 | 4 | 0.65 |
| 3 | Qwen3-32B | 8002 | 4,5 | 2 | 0.75 |

---

## 🧠 Configuration Caching

Working configurations are cached in `~/.smart_serve/config_cache.json`:

- **Valid for 7 days** or until GPU configuration changes
- **Skips re-optimization** on subsequent launches
- **Auto-invalidated** if model config.json changes

```bash
# View cache
cat ~/.smart_serve/config_cache.json

# Clear cache for a specific model
bash serve_model.sh --model models/llama-3.1-8b/ --reset-config

# Clear all caches
rm ~/.smart_serve/config_cache.json
```

---

## 🔍 Supported Model Families

| Family | Auto-Detect | Reasoning Parser | Notes |
|--------|-------------|------------------|-------|
| **Gemma-4** | ✅ | `gemma4` | Multimodal support |
| **Qwen3** | ✅ | `qwen3` | Reasoning-capable variants |
| **DeepSeek-R1** | ✅ | `deepseek_r1` | Reasoning parser |
| **GLM-4.5** | ✅ | `glm45` | Reasoning parser |
| **Mistral** | ✅ | `mistral` | Custom tokenizer/loader |
| **Magistral** | ✅ | `mistral` | Reasoning parser |
| **Llama-3** | ✅ | - | Standard deployment |
| **LLaVA** | ✅ | - | Multimodal auto-detect |
| **Other** | ⚠️ | - | Falls back to defaults |

---

## 🛠️ Troubleshooting

| Problem | Likely Cause | Solution |
|---------|--------------|----------|
| `Free memory on device cuda:X is less than desired` | GPU busy or utilization too high | Use `--gpus` to select free GPUs, or lower `--max-util` |
| `WorkerProc failed to start` | OOM during initialization | Run with `--min-free-gb 20` to exclude busy GPUs |
| `nvidia-smi not available` | NVIDIA driver missing | Install or repair NVIDIA drivers |
| `torch.cuda.is_available() is False` | CUDA runtime mismatch | Reinstall venv, validate CUDA stack |
| `Zero visible GPUs` | GPUs masked by other processes | Check `CUDA_VISIBLE_DEVICES`, use `--gpus` to specify |
| Model download denied | Missing HF token | Set `HF_TOKEN` or run `hf auth login` |
| vLLM startup fails on model quirks | Missing family-specific args | Use `--no-reasoning` or `--no-multimodal` |
| Startup fails after version changes | Stale caches | Clear `~/.smart_serve/config_cache.json` |
| Parallelism error at startup | TP size doesn't match visible GPUs | Check `--gpus` and `--tp-size` alignment |

### Quick Diagnostics

```bash
# Check GPU status
nvidia-smi

# Check which GPUs are free
bash serve_model.sh --model test --dry-run

# View smart serve logs
tail -f ~/.smart_serve/smart_serve.log

# Test CUDA setup
python test_cuda.py
```

---

## 📁 Project Structure

```
vllm_for_linux/
├── install.sh              # Environment bootstrap
├── test_cuda.py            # CUDA validation
├── download_model.py       # HuggingFace model downloader
├── serve_model.sh          # Smart model server launcher
├── smart_serve.py          # GPU analyzer and optimizer
├── create_runner.py        # Runner script generator
├── clear.sh                # Cache cleanup utility
├── runners/                # Auto-generated runners (gitignored)
│   └── .gitkeep
├── models/                 # Downloaded model snapshots
├── venv/                   # Python virtual environment
└── README.md               # This file
```

---

## 🔐 Hugging Face Token Setup

For gated models and better rate limits:

```bash
# Option 1: Environment variable
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxx"

# Option 2: Interactive login
hf auth login

# Option 3: Login with token
hf auth login --token hf_xxxxxxxxxxxxxxxxx
```

---

## 🤝 Contributing

Contributions are welcome! Good targets include:

- New model-family presets
- Better GPU allocation algorithms
- Improved diagnostics and logging
- Version pinning matrices
- Documentation improvements
- Reproducible bugfixes

**When submitting a PR, please include:**
- GPU model and count
- CUDA version
- PyTorch version
- vLLM version
- Steps to reproduce (if bugfix)

---

## 📄 License

This project is licensed under the **Apache 2.0 License**. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Credits

Created by **ParisNeo**

Built with ❤️ for the open-source AI community
