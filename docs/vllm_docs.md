# vLLM & vLLM-Omni Official Documentation

This document provides a comprehensive guide to installing, deploying, and utilizing vLLM and vLLM-Omni for both text and image-based multimodal models.

---

## Table of Contents
1. [Quickstart Guide](#1-quickstart-guide)
2. [GPU Setup & Installation](#2-gpu-setup--installation)
3. [Offline Inference](#3-offline-inference)
4. [Online Serving & API Reference](#4-online-serving--api-reference)
5. [Model Recipes & Usage Guides](#5-model-recipes--usage-guides)

---

## 1. Quickstart Guide

This guide will help you quickly get started with vLLM-Omni to perform:
- Offline batched inference
- Online serving using an OpenAI-compatible server

### Prerequisites
- **OS**: Linux
- **Python**: 3.12

### Installation

For installation on GPU from source:

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate

# On CUDA
uv pip install vllm==0.25.0 --torch-backend=auto

# On ROCm
uv pip install vllm==0.25.0+rocm723 --extra-index-url https://wheels.vllm.ai/rocm/0.25.0/rocm723

git clone https://github.com/vllm-project/vllm-omni.git
cd vllm-omni
uv pip install -e .
```

> **Note**
> It is important to install the same major & minor version of vLLM and vLLM Omni, otherwise things may not work as expected. If the versions are misaligned, you will see a warning when you import vLLM Omni.
>
> If you are seeing strange behavior with the `vllm` command not handling the `--omni` flag correctly, you most likely have a version mismatch with vLLM < 0.25.0 and vLLM Omni 0.25.0, as vLLM Omni no longer hijacks the vLLM entrypoint. Updating vLLM should resolve this issue.

---

## 2. GPU Setup & Installation

vLLM-Omni is a Python library that supports the following GPU variants. The library itself mainly contains python implementations for framework and models.

### Requirements
- **OS**: Linux
- **Python**: 3.12

> **Note**
> vLLM-Omni is currently not natively supported on Windows.

### NVIDIA CUDA Requirements
- GPU: compute capability 7.0 or higher (e.g., V100, T4, RTX20xx, A100, L4, H100, etc.)

### Set up using Python

#### Create a new Python environment
It's recommended to use `uv`, a very fast Python environment manager, to create and manage Python environments. Please follow the documentation to install `uv`. After installing `uv`, you can create a new Python environment using the following commands:

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
```

#### Pre-built wheels
> **Note:** Pre-built wheels are currently available for vLLM-Omni 0.11.0rc1, 0.12.0rc1, 0.14.0rc1, 0.14.0, 0.16.0, 0.18.0, 0.20.0, 0.21.0, 0.22.0, 0.23.0, 0.24.0, and 0.25.0. If you need a newer unreleased revision, please build from source.

**Installation of vLLM**
vLLM-Omni is built based on vLLM. Please install it with command below.

```bash
uv pip install vllm==0.25.0 --torch-backend=auto
```

**Installation of vLLM-Omni**

```bash
uv pip install vllm-omni
```

To run Gradio demos, also install the optional extras:

```bash
uv pip install 'vllm-omni[demo]'
```

#### Build wheel from source

**Installation of vLLM**
If you do not need to modify source code of vLLM, you can directly install the stable 0.25.0 release version of the library:

```bash
uv pip install vllm==0.25.0 --torch-backend=auto
```

The 0.25.0 release of vLLM ships CUDA 13.0-compatible binaries by default. If you need a different CUDA variant or want to reuse an existing PyTorch installation, build vLLM from source instead.

**Installation of vLLM-Omni**
Since `vllm-omni` is rapidly evolving, it's recommended to install it from source:

```bash
git clone https://github.com/vllm-project/vllm-omni.git
cd vllm-omni
uv pip install -e .
```

To run Gradio demos, install with optional extras:

```bash
uv pip install -e '.[demo]'
```

*(Optional) Installation of vLLM from source is also supported if you need to modify vLLM internals.*

### Set up using Docker

#### Pre-built images
vLLM-Omni offers an official docker image for deployment. These images are built on top of vLLM docker images and available on Docker Hub as `vllm/vllm-omni`. The version of vLLM-Omni indicates which release of vLLM it is based on.

Here's an example deployment command that has been verified on 2 x H100's:

```bash
docker run --runtime nvidia --gpus 2 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p 8091:8091 \
    --ipc=host \
    vllm/vllm-omni:v0.25.0 \
    vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091
```

> **Tip**
> The CUDA image does not define a default entrypoint, so include `vllm serve ... --omni` after the image name.

#### Build your own docker image

```bash
DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.cuda -t vllm-omni-cuda .
```

If you want to specify the base vLLM version:

```bash
DOCKER_BUILDKIT=1 docker build \
  -f docker/Dockerfile.cuda \
  --build-arg BASE_IMAGE=vllm/vllm-openai:v0.22.1 \
  -t vllm-omni-cuda .
```

**Launch the docker image**

> **Note**
> The model `Qwen/Qwen3-Omni-30B-A3B-Instruct` requires significant GPU memory. The example below has been verified on 2 x H100's.

*Launch with OpenAI API Server:*
```bash
docker run --runtime nvidia --gpus 2 \
  -v ${HF_HOME:-$HOME/.cache/huggingface}:/root/.cache/huggingface \
  --env "HF_TOKEN=$HF_TOKEN" \
  -p 8091:8091 \
  --ipc=host \
  vllm-omni-cuda \
  vllm serve --omni --model Qwen/Qwen3-Omni-30B-A3B-Instruct --port 8091
```

By default, this mounts `$HOME/.cache/huggingface` as the model cache directory. To use a custom location, set the `HF_HOME` environment variable before running the command (e.g., `export HF_HOME=/data/models`).

*Launch with interactive session for development:*
```bash
docker run --runtime nvidia --gpus all -it --rm \
  -v ${HF_HOME:-$HOME/.cache/huggingface}:/root/.cache/huggingface \
  --env "HF_TOKEN=$HF_TOKEN" \
  -p 8091:8091 \
  --ipc=host \
  --entrypoint bash \
  vllm-omni-cuda
```

---

## 3. Offline Inference

### Text-to-Image Generation

Text-to-image generation quickstart with vLLM-Omni:

```python
from vllm_omni.entrypoints.omni import Omni

if __name__ == "__main__":
    omni = Omni(model="Tongyi-MAI/Z-Image-Turbo")
    prompt = "a cup of coffee on the table"
    outputs = omni.generate(prompt)
    images = outputs[0].request_output.images
    images[0].save("coffee.png")
```

You can pass a list of prompts and wait for the independent requests to finish, as shown below.

> **Info**
> For diffusion pipelines, each prompt becomes a separate logical request. The runtime may automatically batch compatible in-flight requests through the scheduler and runner.

```python
from vllm_omni.entrypoints.omni import Omni

if __name__ == "__main__":
    omni = Omni(
        model="Tongyi-MAI/Z-Image-Turbo",
        # stage_configs_path="./stage-config.yaml",  # See below
    )
    prompts = [
        "a cup of coffee on a table",
        "a toy dinosaur on a sandy beach",
        "a fox waking up in bed and yawning",
    ]
    omni_outputs = omni.generate(prompts)
    for i_prompt, prompt_output in enumerate(omni_outputs):
        this_request_output = prompt_output.request_output
        this_images = this_request_output.images
        for i_image, image in enumerate(this_images):
            image.save(f"p{i_prompt}-img{i_image}.jpg")
            print("saved to", f"p{i_prompt}-img{i_image}.jpg")
            # saved to p0-img0.jpg
            # saved to p1-img0.jpg
            # saved to p2-img0.jpg
```

> **Info**
> For diffusion request-level batching controls such as `max_num_seqs` and `request_batch_max_wait_ms`, see Request-Level Batching.

For more usages, please refer to the offline inference documentation.

---

## 4. Online Serving & API Reference

### Chat Completions API
vLLM-Omni supports generating and editing images via the `/v1/chat/completions` endpoint using diffusion models. This page explains how to pass generation parameters (such as `num_inference_steps`, `height`, `width`) to diffusion models through this endpoint.

> **Tip**
> For dedicated endpoints that accept generation parameters as top-level fields, see the Image Generation API and Image Edit API.

#### Passing Generation Parameters
The `/v1/chat/completions` endpoint follows the OpenAI Chat API schema, which does not natively include diffusion-specific fields like `num_inference_steps` or `height`. How you pass these extra fields depends on your client.

**curl / Python requests**
Wrap generation parameters inside an `"extra_body"` key in the JSON body:

```bash
curl -s http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "A beautiful landscape painting"}
    ],
    "extra_body": {
      "num_inference_steps": 50,
      "seed": 42
    }
  }'
```

**OpenAI Python SDK**
Use the `extra_body` keyword argument. The SDK automatically merges these fields into the top-level request body:

```python
response = client.chat.completions.create(
    model="Qwen/Qwen-Image",
    messages=[{"role": "user", "content": "A beautiful landscape painting"}],
    extra_body={
        "num_inference_steps": 50,
        "seed": 42,
    },
)
```

> **Note: SDK extra_body vs. JSON extra_body**
> These two `extra_body` usages look similar but work differently under the hood. The SDK flattens the dict into the top-level request JSON, while the curl/requests approach sends it as a nested `"extra_body"` key. Both are handled correctly by the server.

> **Note: About the ignored fields warning**
> You may see a log message like:
> `WARNING: The following fields were present in the request but ignored: {'height', 'width', ...}`
> This is harmless. It is emitted by vLLM's request validation layer because these fields are not part of the standard OpenAI ChatCompletionRequest schema. The fields are still stored internally and correctly forwarded to the diffusion pipeline.

### Online Serving Quickstart

Text-to-image generation quickstart with vLLM-Omni:

```bash
vllm serve Tongyi-MAI/Z-Image-Turbo --omni --port 8091

curl -s http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "a cup of coffee on the table"}
    ],
    "extra_body": {
      "height": 1024,
      "width": 1024,
      "num_inference_steps": 50,
      "guidance_scale": 4.0,
      "seed": 42
    }
  }' | jq -r '.choices[0].message.content[0].image_url.url' | cut -d',' -f2 | base64 -d > coffee.png
```

### Image-to-Image (Image Editing)

This example demonstrates how to deploy image-to-image models for an online image editing service using vLLM-Omni.

Supported models include Qwen-Image-Edit, BAGEL, and other image-to-image pipelines.

For multi-image input editing, use Qwen-Image-Edit-2509 (`QwenImageEditPlusPipeline`) and send multiple images in the user message content.

#### Start Server

**Basic Start:**
```bash
vllm serve Qwen/Qwen-Image-Edit --omni --port 8092
```

> **Note**
> If you encounter Out-of-Memory (OOM) issues or have limited GPU memory, you can enable VAE slicing and tiling to reduce memory usage: `--vae-use-slicing --vae-use-tiling`

**Multi-Image Edit (Qwen-Image-Edit-2509):**
```bash
vllm serve Qwen/Qwen-Image-Edit-2509 --omni --port 8092
```

**BAGEL:**
```bash
vllm serve ByteDance-Seed/BAGEL-7B-MoT --omni --port 8091
```

#### API Calls

**Method 1: Using curl (Image Editing)**
```bash
# Image editing
bash run_curl_image_edit.sh input.png "Convert this image to watercolor style"

# Or execute directly
IMG_B64=$(base64 -w0 input.png)

cat <<EOF > request.json
{
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "Convert this image to watercolor style"},
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,$IMG_B64"}}
    ]
  }],
  "extra_body": {
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 50,
    "guidance_scale": 1,
    "seed": 42
  }
}
EOF

curl -s http://localhost:8092/v1/chat/completions   -H "Content-Type: application/json"   -d @request.json | jq -r '.choices[0].message.content[0].image_url.url' | cut -d',' -f2 | base64 -d > output.png
```

**Method 2: Using OpenAI Python SDK**
```python
import base64
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8092/v1", api_key="none")

with open("input.png", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

response = client.chat.completions.create(
    model="Qwen/Qwen-Image-Edit",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Convert to watercolor style"},
            {"type": "image_url", "image_url": {
                "url": f"data:image/png;base64,{img_b64}"
            }},
        ],
    }],
    extra_body={
        "num_inference_steps": 50,
        "guidance_scale": 1,
        "seed": 42,
    },
)

img_url = response.choices[0].message.content[0].image_url.url
_, b64_data = img_url.split(",", 1)
with open("output.png", "wb") as f:
    f.write(base64.b64decode(b64_data))
```

> **Note**
> The OpenAI SDK's `extra_body` keyword argument merges parameters into the top-level request body automatically. When using curl or Python requests, wrap generation parameters inside a literal `"extra_body"` key in the JSON instead.

**Method 3: Using Python Client Script**
```bash
python openai_chat_client.py --input input.png --prompt "Convert to oil painting style" --output output.png

# Multi-image editing (Qwen-Image-Edit-2509 server required)
python openai_chat_client.py --input input1.png input2.png --prompt "Combine these images into a single scene" --output output.png
```

Pass model-specific parameters through `--extra-body` (e.g. for BAGEL):
```bash
python openai_chat_client.py \
  --input input.png \
  --prompt "Make the scene look like a watercolor painting" \
  --server http://localhost:8091 \
  --extra-body '{"cfg_text_scale": 4.0, "cfg_img_scale": 1.5}'
```

**Method 4: Using Gradio Demo**
```bash
python gradio_demo.py
# Visit http://localhost:7861
```

### Request Formats

**Image Editing (Using `image_url` Format)**
```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Convert this image to watercolor style"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
      ]
    }
  ]
}
```

**Image Editing (Using Simplified `image` Format)**
```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"text": "Convert this image to watercolor style"},
        {"image": "BASE64_IMAGE_DATA"}
      ]
    }
  ]
}
```

**Image Editing with Parameters**
Use `extra_body` to pass generation parameters:
```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Convert to ink wash painting style"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
      ]
    }
  ],
  "extra_body": {
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 50,
    "guidance_scale": 7.5,
    "seed": 42
  }
}
```

### Layered Image Generation (Qwen-Image-Layered)

Qwen-Image-Layered generates multiple decomposed layers from a reference image and a text prompt. Start the server with:

```bash
vllm serve Qwen/Qwen-Image-Layered --omni --port 8093
```

**Using curl:**
```bash
IMG_B64=$(base64 -w0 input.png)

curl -sS http://localhost:8093/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "$(jq -n --arg img "$IMG_B64" '{
    messages: [{
      role: "user",
      content: [
        {type: "image_url", image_url: {url: ("data:image/png;base64," + $img)}},
        {type: "text", text: "a rabbit"}
      ]
    }],
    extra_body: {
      num_inference_steps: 50,
      cfg_scale: 4.0,
      seed: 0,
      layers: 4,
      resolution: 640
    }
  }')" \
  | jq -r '.choices[0].message.content[] | .image_url.url | split(",")[1]' \
  | while IFS= read -r b64; do
      ((i++)); echo "$b64" | base64 -d > "layer_${i}.png"
    done
```

**Using Python:**
```python
import base64
import requests

with open("input.png", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

payload = {
    "messages": [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {
                "url": f"data:image/png;base64,{img_b64}"
            }},
            {"type": "text", "text": "a rabbit"},
        ],
    }],
    "extra_body": {
        "num_inference_steps": 50,
        "cfg_scale": 4.0,
        "seed": 0,
        "layers": 4,
        "resolution": 640,
    },
}

resp = requests.post(
    "http://localhost:8093/v1/chat/completions",
    json=payload,
    timeout=600,
)
data = resp.json()

for i, item in enumerate(data["choices"][0]["message"]["content"]):
    _, b64_data = item["image_url"]["url"].split(",", 1)
    with open(f"layer_{i}.png", "wb") as f:
        f.write(base64.b64decode(b64_data))
```

The response contains multiple images in `choices[0].message.content` — one per generated layer.

#### Qwen-Image-Layered Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `layers` | int | 4 | Number of layers to decompose |
| `resolution` | int | 640 | Resolution for dimension calculation (640 or 1024) |
| `cfg_scale` | float | 4.0 | Classifier-free guidance scale (alias for `true_cfg_scale`) |
| `num_inference_steps` | int | 50 | Number of denoising steps |
| `seed` | int | None | Random seed for reproducibility |

### Multi-Image Editing (Qwen-Image-Edit-2509)

Provide multiple images in content (order matters):
```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Combine these images into a single scene"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."} },
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."} }
      ]
    }
  ]
}
```

### Generation Parameters

When using `/v1/chat/completions`, pass these inside `extra_body` in the curl JSON, or via the `extra_body` keyword argument in the OpenAI Python SDK. When using the dedicated `/v1/images/edits` endpoint, pass the supported generation controls as top-level form fields directly. For image dimensions and count, use `size` and `n` rather than `height`, `width`, or `num_outputs_per_prompt`.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `height` | int | None | Output image height in pixels |
| `width` | int | None | Output image width in pixels |
| `size` | str | None | Output image size (e.g., "1024x1024") |
| `num_inference_steps` | int | 50 | Number of denoising steps |
| `guidance_scale` | float | 1.0 | CFG guidance scale |
| `seed` | int | None | Random seed (reproducible) |
| `negative_prompt` | str | None | Negative prompt |
| `num_outputs_per_prompt` | int | 1 | Number of images to generate |
| `strength` | float | 0.6 | Z-Image only - Denoising start timestep for I2I. Range: [0.0, 1.0]. Lower preserves more of original image. |
| `layers` | int | 4 | Number of layers (Qwen-Image-Layered) |
| `resolution` | int | 640 | Resolution, 640 or 1024 (Qwen-Image-Layered) |

> **Note**
> Models like BAGEL accept additional parameters via `extra_body` (e.g. `cfg_text_scale`, `cfg_img_scale`). See the BAGEL recipe for the full list.

### Response Format

```json
{
  "id": "chatcmpl-xxx",
  "created": 1234567890,
  "model": "Qwen/Qwen-Image-Edit",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": [{
        "type": "image_url",
        "image_url": {
          "url": "data:image/png;base64,..."
        }
      }]
    },
    "finish_reason": "stop"
  }],
  "usage": {...}
}
```

### Common Editing Instructions Examples

| Instruction | Description |
| :--- | :--- |
| Convert this image to watercolor style | Style transfer |
| Convert the image to black and white | Desaturation |
| Enhance the color saturation | Color adjustment |
| Convert to cartoon style | Cartoonization |
| Add vintage filter effect | Filter effect |
| Convert daytime scene to nighttime | Scene conversion |

### File Description

| File | Description |
| :--- | :--- |
| `run_server.sh` | Server startup script |
| `run_curl_image_edit.sh` | curl image editing example |
| `openai_chat_client.py` | Python client |
| `gradio_demo.py` | Gradio interactive interface |

---

## 5. Model Recipes & Usage Guides

### Qwen-Image Usage Guide

Qwen-Image models include the following:

| Model | HuggingFace | Description |
| :--- | :--- | :--- |
| Qwen-Image | 🤗 Qwen/Qwen-Image | Text-to-image generation (20B parameters, Aug 2025) |
| Qwen-Image-2512 | 🤗 Qwen/Qwen-Image-2512 | Updated T2I with enhanced realism and text rendering (Dec 2025) |
| Qwen-Image-Edit | 🤗 Qwen/Qwen-Image-Edit | Single-image editing with semantic and appearance control (Aug 2025) |
| Qwen-Image-Edit-2509 | 🤗 Qwen/Qwen-Image-Edit-2509 | Multi-image editing with improved consistency (Sep 2025) |
| Qwen-Image-Edit-2511 | 🤗 Qwen/Qwen-Image-Edit-2511 | Further enhanced consistency, built-in LoRA support (Nov 2025) |
| Qwen-Image-Layered | 🤗 Qwen/Qwen-Image-Layered | Decomposes an input image into multiple RGBA layers (Dec 2025) |

All models share the same DiT transformer core; hence, the acceleration methods (e.g., cache methods, parallelism methods) are applicable across the entire series.

#### Installation

```bash
# Clone and install vllm-omni
git clone https://github.com/vllm-project/vllm-omni.git
cd vllm-omni
uv venv
source .venv/bin/activate
uv pip install -e . vllm==0.18.0
```

#### Text-to-Image (Qwen-Image, Qwen-Image-2512)

```bash
# Qwen-Image (default)
python3 ./examples/offline_inference/text_to_image/text_to_image.py \
    --model Qwen/Qwen-Image \
    --prompt "a cup of coffee on the table" \
    --output output_qwen_image.png \
    --num-inference-steps 50 \
    --cfg-scale 4.0

# Qwen-Image-2512
python3 ./examples/offline_inference/text_to_image/text_to_image.py \
    --model Qwen/Qwen-Image-2512 \
    --prompt "a cup of coffee on the table" \
    --output output_qwen_image_2512.png \
    --num-inference-steps 50 \
    --cfg-scale 4.0
```

> **Notes:**
> 1. vLLM-Omni enables `torch.compile` by default. Try `--enforce-eager` if you want to disable it.
> 2. vLLM-Omni does not enable CPU offload automatically. If you encounter OOM, please use `--enable-cpu-offload` or `--enable-layerwise-offload`.

#### Image Editing (Qwen-Image-Edit)

Qwen-Image-Edit simultaneously feeds the input image into Qwen2.5-VL (for visual semantic control) and the VAE Encoder (for visual appearance control), achieving capabilities in both semantic and appearance editing.

```bash
# Single image input (Qwen-Image-Edit)
python3 ./examples/offline_inference/image_to_image/image_edit.py \
    --model Qwen/Qwen-Image-Edit \
    --image qwen_bear.png \
    --prompt "Let this mascot dance under the moon, surrounded by floating stars and poetic bubbles such as 'Be Kind'" \
    --output output_image_edit.png \
    --num-inference-steps 50 \
    --cfg-scale 4.0
```

For multiple image inputs, use `Qwen/Qwen-Image-Edit-2509` or `Qwen/Qwen-Image-Edit-2511`:

```bash
# Qwen-Image-Edit-2511 example (multiple images)
python3 ./examples/offline_inference/image_to_image/image_edit.py \
    --model Qwen/Qwen-Image-Edit-2511 \
    --image image1.png image2.png \
    --prompt "Add a white art board written with colorful text 'vLLM-Omni' on grassland. Add a paintbrush in the bear's hands. position the bear standing in front of the art board as if painting" \
    --output output_image_edit.png \
    --num-inference-steps 50 \
    --cfg-scale 4.0
```

#### Image Layering (Qwen-Image-Layered)

```bash
python3 ./examples/offline_inference/image_to_image/image_edit.py \
    --model Qwen/Qwen-Image-Layered \
    --image input.png \
    --prompt "" \
    --output layered \
    --num-inference-steps 50 \
    --cfg-scale 4.0 \
    --layers 4 \
    --color-format "RGBA"
```

#### Key Arguments

| Argument | Description |
| :--- | :--- |
| `--model` | Model name or local path. Use `Qwen/Qwen-Image-Edit-2509` or later for multiple image support. |
| `--image` | Path(s) to the source image(s) (PNG/JPG, converted to RGB). Can specify multiple images. |
| `--prompt` / `--negative-prompt` | Text description (string). |
| `--cfg-scale` | True classifier-free guidance scale (default: 4.0). Enabled by setting `cfg_scale > 1` and providing a `negative_prompt`. Higher guidance scale encourages images closely linked to the text prompt, usually at the expense of lower image quality. |
| `--guidance-scale` | Guidance scale for guidance-distilled models (default: 1.0, disabled). Enabled when `guidance_scale > 1`. Ignored when not using guidance-distilled models. |
| `--num-inference-steps` | Diffusion sampling steps (more steps = higher quality, slower). |
| `--output` | Path to save the generated PNG. For Qwen-Image-Layered, this is used as the filename prefix. |
| `--vae-use-slicing` | Enable VAE slicing for memory optimization. |
| `--vae-use-tiling` | Enable VAE tiling for memory optimization. |
| `--cfg-parallel-size` | Set to 2 to enable CFG Parallel. |
| `--enable-cpu-offload` | Enable CPU offloading for diffusion models. |
| `--layers` | Number of layers to decompose the input image into (Qwen-Image-Layered only). |
| `--color-format` | Output color channel format (RGB or RGBA). Qwen-Image-Layered uses RGBA. |

#### Acceleration Methods

##### Cache Acceleration
vLLM-Omni supports `cache-dit` and `tea-cache` for Qwen-Image models.

**Cache-DiT**
```bash
# Text-to-Image with Cache-DiT
python3 ./examples/offline_inference/text_to_image/text_to_image.py \
    --model Qwen/Qwen-Image \
    --prompt "a cup of coffee on the table" \
    --cache-backend cache_dit
```
Advanced Cache-DiT options:
```bash
python3 ./examples/offline_inference/image_to_image/image_edit.py \
    --model Qwen/Qwen-Image-Edit \
    --image qwen_bear.png \
    --prompt "Edit description" \
    --cache-backend cache_dit \
    --cache-dit-max-continuous-cached-steps 3 \
    --cache-dit-residual-diff-threshold 0.24 \
    --cache-dit-enable-taylorseer
```

**TeaCache**
```bash
# Text-to-Image with TeaCache
python3 ./examples/offline_inference/text_to_image/text_to_image.py \
    --model Qwen/Qwen-Image \
    --prompt "a cup of coffee on the table" \
    --cache-backend tea_cache
```

##### Ulysses Sequence Parallelism
Distributes computation across GPUs without quality loss. Recommended for high-resolution images (>1536px) with 2–8 GPUs.
```bash
# Text-to-Image with Ulysses SP
python3 ./examples/offline_inference/text_to_image/text_to_image.py \
    --model Qwen/Qwen-Image \
    --prompt "a cup of coffee on the table" \
    --ulysses-degree 4

# Image Editing with Ulysses SP
python3 ./examples/offline_inference/image_to_image/image_edit.py \
    --model Qwen/Qwen-Image-Edit \
    --image qwen_bear.png \
    --prompt "Add a white art board written with colorful text 'vLLM-Omni' on grassland." \
    --output output_image_edit.png \
    --num-inference-steps 50 \
    --cfg-scale 4.0 \
    --ulysses-degree 4
```

##### Ring-Attention Sequence Parallelism
Ring-based sequence parallelism, suitable for memory-constrained environments with very long sequences.
```bash
# Text-to-Image with Ring-Attention
python3 ./examples/offline_inference/text_to_image/text_to_image.py \
    --model Qwen/Qwen-Image \
    --prompt "a cup of coffee on the table" \
    --ring-degree 4
```

##### CFG Parallelism
Splits classifier-free guidance positive/negative branches across 2 GPUs. Particularly effective for image editing with `cfg-scale > 1`.
```bash
# Image Editing with CFG Parallel (2 GPUs)
python3 ./examples/offline_inference/image_to_image/image_edit.py \
    --model Qwen/Qwen-Image-Edit \
    --image qwen_bear.png \
    --prompt "Edit description" \
    --cfg-parallel-size 2 \
    --num-inference-steps 50 \
    --cfg-scale 4.0
```

##### Tensor Parallelism
Shards model weights across multiple GPUs. Useful for running the 20B model across 2+ GPUs.
```bash
# Text-to-Image with Tensor Parallelism (2 GPUs)
python3 ./examples/offline_inference/text_to_image/text_to_image.py \
    --model Qwen/Qwen-Image \
    --prompt "a cup of coffee on the table" \
    --tensor-parallel-size 2
```

##### CPU Offload
Offloads DiT layers to CPU memory between forward passes. Enables inference on limited VRAM.
```bash
# Text-to-Image with CPU offload (module-wise)
python3 ./examples/offline_inference/text_to_image/text_to_image.py \
    --model Qwen/Qwen-Image \
    --prompt "a cup of coffee on the table" \
    --enable-cpu-offload

# Image Editing with CPU offload (layerwise, saves more VRAM, but slower)
python3 ./examples/offline_inference/image_to_image/image_edit.py \
    --model Qwen/Qwen-Image-Edit \
    --image qwen_bear.png \
    --prompt "Edit description" \
    --enable-layerwise-offload
```

##### VAE Patch Parallelism
Distributes VAE decode tiling across GPUs, reducing peak VAE memory usage at high resolutions.
```bash
# Text-to-Image with VAE Patch Parallelism
python3 ./examples/offline_inference/text_to_image/text_to_image.py \
    --model Qwen/Qwen-Image \
    --prompt "a cup of coffee on the table" \
    --height 1536 --width 1536 \
    --ulysses-degree 2 \
    --vae-patch-parallel-size 2
```
> **Note:** VAE patch parallelism cannot be used alone. It must be used together with other parallelism methods.

#### Quantization
Qwen-Image and Qwen-Image-2512 support FP8 and INT8 quantization. Qwen-Image-Edit variants do not support quantization.

**FP8**
```bash
# Text-to-Image with FP8 quantization
python3 ./examples/offline_inference/text_to_image/text_to_image.py \
    --model Qwen/Qwen-Image \
    --prompt "a cup of coffee on the table" \
    --quantization fp8

# Skip sensitive layers (recommended for better quality)
python3 ./examples/offline_inference/text_to_image/text_to_image.py \
    --model Qwen/Qwen-Image \
    --prompt "a cup of coffee on the table" \
    --quantization fp8 \
    --ignored-layers "img_mlp"
```

**INT8**
```bash
python3 ./examples/offline_inference/text_to_image/text_to_image.py \
    --model Qwen/Qwen-Image \
    --prompt "a cup of coffee on the table" \
    --quantization int8
```

#### Feature Support Summary
For detailed features support for Qwen-Image series models in vLLM-Omni, see the Feature Support Table. For detailed compatibility between features (e.g., combining Cache + SP + CFG-Parallel), see the Feature Compatibility Guide.

#### Combining Acceleration Methods

A few guidelines help pick the right combination:
- **Cache** (TeaCache or Cache-DiT) reduces redundant DiT computation per inference.
- **Parallelism** (SP, TP, CFG-parallel, VAE patch) splits work across GPUs. Use one cache backend together with any supported parallel strategy.
- TeaCache and Cache-DiT cannot be used together.
- **Sequence parallelism** (Ulysses / Ring) is the best parallelism choice for high-resolution or long-sequence workloads. It generally outperforms tensor parallelism (TP) in these settings.
- **Tensor parallelism** is most useful when model weights alone do not fit on a single GPU.
- **CFG parallelism** targets non-distilled diffusion with full classifier-free guidance (`--cfg-scale > 1`). It assigns the positive and negative CFG branches to separate GPUs, achieving up to ~1.5× speedup.
- To reduce peak VRAM, use `--enable-cpu-offload`, `--enable-layerwise-offload` or pair `--vae-patch-parallel-size` with another parallel method.
- To trade quality for speed, FP8 / INT8 quantization is available for Qwen-Image and Qwen-Image-2512.

**Examples:**

1) Sequence parallelism only:
```bash
python3 ./examples/offline_inference/text_to_image/text_to_image.py \
    --model Qwen/Qwen-Image \
    --prompt "a cup of coffee on the table" \
    --ulysses-degree 4
```

2) Cache only (single GPU):
```bash
python3 ./examples/offline_inference/text_to_image/text_to_image.py \
    --model Qwen/Qwen-Image \
    --prompt "a cup of coffee on the table" \
    --cache-backend cache_dit
```

3) Cache + SP (recommended for long sequence generation):
```bash
python3 ./examples/offline_inference/text_to_image/text_to_image.py \
    --model Qwen/Qwen-Image \
    --prompt "a cup of coffee on the table" \
    --cache-backend cache_dit \
    --ulysses-degree 4
```

4) SP + VAE patch parallel (high-resolution, VRAM-constrained):
```bash
python3 ./examples/offline_inference/text_to_image/text_to_image.py \
    --model Qwen/Qwen-Image \
    --prompt "a cup of coffee on the table" \
    --height 1536 --width 1536 \
    --ulysses-degree 2 \
    --vae-patch-parallel-size 2
```

5) Image editing: cache + CFG parallel:
```bash
python3 ./examples/offline_inference/image_to_image/image_edit.py \
    --model Qwen/Qwen-Image-Edit \
    --image qwen_bear.png \
    --prompt "Edit description" \
    --cache-backend cache_dit \
    --cfg-parallel-size 2 \
    --num-inference-steps 50 \
    --cfg-scale 4.0
```

6) CPU offload (add when OOM):
```bash
python3 ./examples/offline_inference/text_to_image/text_to_image.py \
    --model Qwen/Qwen-Image \
    --prompt "a cup of coffee on the table" \
    --enable-cpu-offload
```

7) Quantization + SP:
```bash
python3 ./examples/offline_inference/text_to_image/text_to_image.py \
    --model Qwen/Qwen-Image \
    --prompt "a cup of coffee on the table" \
    --quantization fp8 \
    --ulysses-degree 2
```

---

### Qwen3.5 & Qwen3.6 Usage Guide

Qwen3.5 and Qwen3.6 are multimodal mixture-of-experts models featuring a gated delta networks architecture. This guide covers how to efficiently deploy and serve both models using vLLM.

| Model | Total Params | Active Params | HuggingFace |
| :--- | :--- | :--- | :--- |
| Qwen3.6 | 35B | 3B (256 experts, 8 routed + 1 shared) | Qwen3.6-35B-A3B / FP8 |
| Qwen3.5 | 397B | 17B | Qwen3.5-397B-A17B / FP8 |

#### Installing vLLM

You can either install vLLM from pip or use the pre-built Docker image.

**Pip Install**

*NVIDIA:*
```bash
uv venv
source .venv/bin/activate
uv pip install -U vllm --torch-backend=auto
```

*AMD:*
> **Note:** The vLLM wheel for ROCm requires Python 3.12, ROCm 7.0, and glibc >= 2.35. If your environment does not meet these requirements, please use the Docker-based setup. Supported GPUs: MI300X, MI325X, MI355X.

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install vllm --extra-index-url https://wheels.vllm.ai/rocm
```

#### Running Qwen3.6

```bash
vllm serve Qwen/Qwen3.6-35B-A3B \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3
```

With MTP speculative decoding:
```bash
vllm serve Qwen/Qwen3.6-35B-A3B \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 2}'
```

#### Docker Deployment

**NVIDIA:**
```bash
docker run --gpus all \
  -p 8000:8000 \
  --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai Qwen/Qwen3.5-397B-A17B \
    --tensor-parallel-size 8 \
    --reasoning-parser qwen3 \
    --enable-prefix-caching
```
*(For Blackwell GPUs, use `vllm/vllm-openai:cu130-nightly`)*

**AMD:**
```bash
docker run --device=/dev/kfd --device=/dev/dri \
  --security-opt seccomp=unconfined \
  --group-add video \
  --ipc=host \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai-rocm:latest \
  Qwen/Qwen3.5-397B-A17B-FP8 \
  --tensor-parallel-size 8 \
  --reasoning-parser qwen3 \
  --enable-prefix-caching
```

#### Running Qwen3.5

The configurations below have been verified on 8x H200 GPUs and 8x MI300X/MI355X GPUs.

> **Tip**
> We recommend using the official FP8 checkpoint `Qwen/Qwen3.5-397B-A17B-FP8` for optimal serving efficiency.

##### Throughput-Focused Serving

**Text-Only:**
For maximum text throughput under high concurrency, use `--language-model-only` to skip loading the vision encoder and free up memory for KV cache as well as enabling Expert Parallelism.

```bash
vllm serve Qwen/Qwen3.5-397B-A17B-FP8 \
  -dp 8 \
  --enable-expert-parallel \
  --language-model-only \
  --reasoning-parser qwen3 \
  --enable-prefix-caching
```

**Multimodal:**
For multimodal workloads, use `--mm-encoder-tp-mode data` for data-parallel vision encoding and `--mm-processor-cache-type shm` to efficiently cache and transfer preprocessed multimodal inputs in shared memory.

```bash
vllm serve Qwen/Qwen3.5-397B-A17B-FP8 \
  -dp 8 \
  --enable-expert-parallel \
  --mm-encoder-tp-mode data \
  --mm-processor-cache-type shm \
  --reasoning-parser qwen3 \
  --enable-prefix-caching
```

> **Tip**
> To enable tool calling, add `--enable-auto-tool-choice --tool-call-parser qwen3_coder` to the serve command.

##### Latency-Focused Serving

For latency-sensitive workloads at low concurrency, enable MTP-1 speculative decoding and disable prefix caching. MTP-1 reduces time-per-output-token (TPOT) with a high acceptance rate, at the cost of lower throughput under load.

> **Note**
> MTP-1 speculative decoding for AMD GPUs is under development.

```bash
vllm serve Qwen/Qwen3.5-397B-A17B-FP8 \
  --tensor-parallel-size 8 \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}' \
  --reasoning-parser qwen3
```

##### GB200 Deployment

> **Tip**
> We recommend using the NVFP4 checkpoint `nvidia/Qwen3.5-397B-A17B-NVFP4` for optimal serving efficiency.

You can also deploy the model across 4 GPUs on a GB200 node, using the similar base configuration as H200.

```bash
vllm serve nvidia/Qwen3.5-397B-A17B-NVFP4 \
  -dp 4 \
  --enable-expert-parallel \
  --language-model-only \
  --reasoning-parser qwen3 \
  --enable-prefix-caching
```

##### MI355X Deployment

You can also deploy the model across 2 GPUs on a MI355X node, using the similar base configuration as above.

```bash
vllm serve Qwen/Qwen3.5-397B-A17B-FP8 \
  -tp 2 \
  --enable-expert-parallel \
  --language-model-only \
  --reasoning-parser qwen3 \
  --enable-prefix-caching
```

#### Configuration Tips
- **Disable Reasoning:** If you want to disable the reasoning mode via command-line parameters (instead of modifying the request body), you can add: `--reasoning-parser qwen3 --default-chat-template-kwargs '{"enable_thinking": false}'`.
- **Prefix Caching:** Prefix caching for Mamba cache "align" mode is currently experimental. Please report any issues you may observe.
- **Multi-token Prediction:** MTP-1 reduces per-token latency but degrades text throughput under high concurrency because speculative tokens consume KV cache capacity, reducing effective batch size. Depending on your use case, you may adjust `num_speculative_tokens` (1-5): higher values can improve latency further but may have varying acceptance rates and throughput trade-offs.
- **Encoder Data Parallelism:** Specifying `--mm-encoder-tp-mode data` deploys the vision encoder in a data-parallel fashion for better throughput performance. This consumes additional memory and may require adjustment of `--gpu-memory-utilization`.
- **Media Embedding Size:** You can adjust the maximum media embedding size allowed by modifying the HuggingFace processor config at server startup via passing `--mm-processor-kwargs`. To enable up to 224K visual context length, set the `longest_edge` to `469762048` (calculated as 2 * 32 * 32 * 224 * 1024): `--mm-processor-kwargs '{"videos_kwargs": {"size": {"longest_edge": 469762048, "shortest_edge": 4096}}}'`

You may encounter the following error:
```text
(Worker_TP0 pid=70) ERROR 02-08 08:39:04 [multiproc_executor.py:852]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_next.py", line 585, in _forward_core
(Worker_TP0 pid=70) ERROR 02-08 08:39:04 [multiproc_executor.py:852]     mixed_qkv_non_spec = causal_conv1d_update(
(Worker_TP0 pid=70) ERROR 02-08 08:39:04 [multiproc_executor.py:852]                          ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=70) ERROR 02-08 08:39:04 [multiproc_executor.py:852]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mamba/ops/causal_conv1d.py", line 1160, in causal_conv1d_update
(Worker_TP0 pid=70) ERROR 02-08 08:39:04 [multiproc_executor.py:852]     assert num_cache_lines >= batch
```
*This is because cuda graph capture size is larger than mamba cache size. Try reducing `--max-cudagraph-capture-size`, the default value is 512. See https://github.com/vllm-project/vllm/pull/34571 for details*

#### Benchmarking

Once the server is running, open another terminal and run the benchmark client:

```bash
vllm bench serve \
  --backend openai-chat \
  --endpoint /v1/chat/completions \
  --model Qwen/Qwen3.5-397B-A17B \
  --dataset-name random \
  --random-input-len 2048 \
  --random-output-len 512 \
  --num-prompts 1000 \
  --request-rate 20
```

#### Consume the OpenAI API Compatible Server

```python
import time
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
    timeout=3600
)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://ofasys-multimodal-wlcb-3-toshanghai.oss-accelerate.aliyuncs.com/wpf272043/keepme/image/receipt.png"
                }
            },
            {
                "type": "text",
                "text": "Read all the text in the image."
            }
        ]
    }
]

start = time.time()
response = client.chat.completions.create(
    model="Qwen/Qwen3.5-397B-A17B",
    messages=messages,
    max_tokens=2048
)
print(f"Response costs: {time.time() - start:.2f}s")
print(f"Generated text: {response.choices[0].message.content}")
```

#### Processing Ultra-Long Texts

Qwen3.5 natively supports context lengths of up to 262,144 tokens. For long-horizon tasks where the total length (including both input and output) exceeds this limit, we recommend using RoPE scaling techniques to handle long texts effectively (e.g., YaRN). You can override the `rope_parameters` in your running script.

```bash
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve ... --hf-overrides '{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0, "original_max_position_embeddings": 262144}}}' --max-model-len 1010000
```

---
*Documentation last updated: April 16, 2026*
