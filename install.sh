#!/usr/bin/env bash
set -euo pipefail

print_banner() {
  cat <<'EOF'
================================================================================
__     __ _ _                 __            _ _                      
\ \   / /| | |               / _|          | (_)                     
 \ \_/ / | | | _ __ ___     | |_ ___  _ __ | |_ _ __  _   ___  __    
  \   /  | | || '_ ` _ \    |  _/ _ \| '_ \| | | '_ \| | | \ \/ /    
   | |   | | || | | | | |   | || (_) | | | | | | | | | |_| |>  <     
   |_|   |_||_||_| |_| |_|   |_| \___/|_| |_|_|_|_| |_|\__,_/_/\_\    

                        vllm_for_linux setup tool
                              By ParisNeo
================================================================================
EOF
}

print_info() {
  echo "[INFO] $1"
}

print_warn() {
  echo "[WARN] $1"
}

print_error() {
  echo "[ERROR] $1"
}

print_banner

if ! command -v uv >/dev/null 2>&1; then
  print_error "uv is not installed."
  echo "Install it with:"
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

print_info "Creating virtual environment..."
uv venv venv

print_info "Activating virtual environment..."
# shellcheck disable=SC1091
source venv/bin/activate

print_info "Installing required Python packages..."
uv pip install -U \
  vllm \
  huggingface_hub \
  ascii-colors

print_info "Writing CUDA validation script..."
cat > test_cuda.py <<'PY'
from pathlib import Path
import os
import subprocess
import sys

errors = []
warnings = []

def section(title):
    print("=" * 80)
    print(title)
    print("=" * 80)

def info(msg):
    print(f"[INFO] {msg}")

def warn(msg):
    print(f"[WARN] {msg}")
    warnings.append(msg)

def error(msg):
    print(f"[ERROR] {msg}")
    errors.append(msg)

def parse_python_version_tuple():
    return sys.version_info.major, sys.version_info.minor

def check_python_compatibility():
    major, minor = parse_python_version_tuple()
    py = f"{major}.{minor}"
    print(f"Detected Python minor version: {py}")

    if major != 3:
        error(f"Unsupported Python major version: {major}. vLLM requires Python 3.")
        return

    if minor < 10:
        error(
            f"Python {py} is too old for current vLLM releases. "
            "Use Python 3.10+."
        )
    elif minor == 10 or minor == 11:
        warn(
            f"Python {py} may work depending on the vLLM release, but newer installs "
            "often work best on Python 3.12."
        )
    elif minor == 12:
        info("Python 3.12 detected. This is a strong choice for recent vLLM releases.")
    else:
        warn(
            f"Python {py} is newer than many documented examples. "
            "If install/runtime issues appear, try Python 3.12."
        )

def analyze_cuda_visible_devices():
    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    print(f"CUDA_VISIBLE_DEVICES : {value}")

    if value is None:
        info("CUDA_VISIBLE_DEVICES is not set; all visible GPUs are exposed by default.")
        return

    stripped = value.strip()
    if stripped == "":
        error("CUDA_VISIBLE_DEVICES is set to an empty string, which hides all GPUs.")
        return

    if stripped in {"-1", "none", "None"}:
        error(f"CUDA_VISIBLE_DEVICES={value!r} hides CUDA devices.")
        return

    parts = [p.strip() for p in stripped.split(",")]
    bad = [p for p in parts if not p.isdigit()]
    if bad:
        warn(
            "CUDA_VISIBLE_DEVICES contains non-integer entries: "
            + ", ".join(repr(x) for x in bad)
        )
    else:
        info("CUDA_VISIBLE_DEVICES format looks valid.")

def check_nvcc():
    section("NVCC")
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            print(result.stdout)
        else:
            warn("nvcc not found in PATH. This is OK for wheel installs, but needed for many source builds.")
    except Exception as ex:
        warn(f"Failed to run nvcc --version: {ex}")

section("SYSTEM")
print(f"Python executable : {sys.executable}")
print(f"Python version    : {sys.version}")
analyze_cuda_visible_devices()
print(f"LD_LIBRARY_PATH      : {os.environ.get('LD_LIBRARY_PATH')}")
print()

section("PYTHON COMPATIBILITY")
check_python_compatibility()
print()

section("NVIDIA-SMI")
nvidia_smi_ok = False
try:
    result = subprocess.run(
        ["nvidia-smi"],
        capture_output=True,
        text=True,
        check=False,
    )
    print(result.stdout)
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
    if result.returncode == 0 and result.stdout.strip():
        nvidia_smi_ok = True
    else:
        error("nvidia-smi failed or returned no usable output.")
except Exception as ex:
    print(f"Failed to run nvidia-smi: {ex}")
    error(f"nvidia-smi unavailable: {ex}")

print()

section("DRIVER VERSION")
driver_versions = []
try:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    print(result.stdout)
    if result.returncode == 0:
        driver_versions = [x.strip() for x in result.stdout.splitlines() if x.strip()]
    if not driver_versions:
        warn("Could not read NVIDIA driver version.")
except Exception as ex:
    print(f"Failed to query driver version: {ex}")
    warn(f"Driver version query failed: {ex}")

print()

section("/DEV/NVIDIA*")
try:
    nvidia_devices = sorted(Path("/dev").glob("nvidia*"))
    for device in nvidia_devices:
        print(device)
    if not nvidia_devices:
        warn("No /dev/nvidia* devices found.")
except Exception as ex:
    print(f"Failed to list NVIDIA devices: {ex}")
    warn(f"Failed to list /dev/nvidia* devices: {ex}")

print()

section("PYTORCH")
torch_ok = False
cuda_available = False
device_count = 0
compute_caps = []

try:
    import torch

    print(f"torch.__file__              : {torch.__file__}")
    print(f"torch.__version__           : {torch.__version__}")
    print(f"torch.version.cuda          : {torch.version.cuda}")

    if torch.version.cuda is None:
        error("Installed PyTorch does not appear to include CUDA support.")

    try:
        available = torch.cuda.is_available()
        cuda_available = bool(available)
        print(f"torch.cuda.is_available()   : {available}")
        if not available:
            error("torch.cuda.is_available() is False.")
    except Exception as ex:
        print(f"torch.cuda.is_available()   : ERROR -> {ex}")
        error(f"torch.cuda.is_available() failed: {ex}")

    try:
        count = torch.cuda.device_count()
        device_count = count
        print(f"torch.cuda.device_count()   : {count}")
        if count == 0:
            error("PyTorch sees zero CUDA devices.")
    except Exception as ex:
        print(f"torch.cuda.device_count()   : ERROR -> {ex}")
        error(f"torch.cuda.device_count() failed: {ex}")
        count = 0

    try:
        torch.cuda.init()
        print("torch.cuda.init()           : SUCCESS")
    except Exception as ex:
        print(f"torch.cuda.init()           : ERROR -> {ex}")
        error(f"torch.cuda.init() failed: {ex}")

    for i in range(count):
        print()
        print(f"GPU {i}")
        print("-" * 40)

        try:
            print(f"Name                : {torch.cuda.get_device_name(i)}")
        except Exception as ex:
            print(f"Name                : ERROR -> {ex}")

        try:
            props = torch.cuda.get_device_properties(i)
            print(f"Total memory (GB)   : {props.total_memory / 1024**3:.2f}")
            print(f"Compute capability  : {props.major}.{props.minor}")
            compute_caps.append((props.major, props.minor))
            if props.major < 7:
                warn(
                    f"GPU {i} has compute capability {props.major}.{props.minor}; "
                    "vLLM generally expects 7.0+ GPUs."
                )
        except Exception as ex:
            print(f"Properties          : ERROR -> {ex}")
            warn(f"Could not query properties for GPU {i}: {ex}")

    print()
    section("CUDA COMPUTE TEST")

    try:
        x = torch.randn(1000, 1000, device="cuda")
        y = torch.randn(1000, 1000, device="cuda")
        z = torch.matmul(x, y)
        torch.cuda.synchronize()

        print("CUDA computation: SUCCESS")
        print(f"Result shape : {z.shape}")
        print(f"Device       : {z.device}")
        torch_ok = True

    except Exception as ex:
        print(f"CUDA computation: ERROR -> {ex}")
        error(f"CUDA compute test failed: {ex}")

except Exception as ex:
    print(f"Failed to import torch: {ex}")
    error(f"Failed to import torch: {ex}")

print()
section("LIBCUDA")
libcuda_found = False
try:
    result = subprocess.run(
        ["ldconfig", "-p"],
        capture_output=True,
        text=True,
        check=False,
    )
    for line in result.stdout.splitlines():
        if "libcuda.so" in line.lower():
            print(line)
            libcuda_found = True

    if not libcuda_found:
        warn("libcuda.so not found in ldconfig -p output.")
except Exception as ex:
    print(f"Failed to inspect libcuda: {ex}")
    warn(f"Failed to inspect libcuda: {ex}")

print()
check_nvcc()
print()

section("DIAGNOSIS")

if not nvidia_smi_ok:
    print("- NVIDIA driver tools are not working.")
    print("  Advice: install or repair the NVIDIA driver, then verify `nvidia-smi` works.")

if not libcuda_found:
    print("- libcuda.so is not visible to the dynamic linker.")
    print("  Advice: ensure the NVIDIA driver is installed correctly and LD_LIBRARY_PATH is sane.")

if errors:
    print("- PyTorch/CUDA stack has blocking issues.")
    if any("too old for current vLLM releases" in e for e in errors):
        print("  Advice: recreate the environment with Python 3.10+; Python 3.12 is a good default.")
    if any("does not appear to include CUDA support" in e for e in errors):
        print("  Advice: reinstall in a fresh venv and verify you are getting a CUDA-enabled torch/vLLM build.")
    if any("torch.cuda.is_available() is False." in e for e in errors):
        print("  Advice: check the driver, CUDA visibility, GPU passthrough, and whether CUDA_VISIBLE_DEVICES hides devices.")
    if any("zero CUDA devices" in e or "device_count" in e for e in errors):
        print("  Advice: verify GPU access permissions and check CUDA_VISIBLE_DEVICES.")
    if any("torch.cuda.init()" in e for e in errors):
        print("  Advice: this often indicates a driver/runtime mismatch or broken CUDA userspace libraries.")
    if any("CUDA compute test failed" in e for e in errors):
        print("  Advice: initialization may succeed while kernels still fail; check driver/runtime compatibility.")
    if any("CUDA_VISIBLE_DEVICES is set to an empty string" in e or "hides CUDA devices" in e for e in errors):
        print("  Advice: unset CUDA_VISIBLE_DEVICES or set it to valid GPU indices such as 0 or 0,1.")

if compute_caps and any(major < 7 for major, _ in compute_caps):
    print("- One or more GPUs are below compute capability 7.0.")
    print("  Advice: vLLM generally targets Volta/Turing/Ampere/Hopper-class GPUs or newer.")

if not errors and torch_ok:
    print("Everything looks OK for CUDA/PyTorch basic usage.")
    print("vLLM should run if the selected model fits GPU memory and the installed wheel matches your platform.")

if warnings:
    print()
    print("Warnings:")
    for w in warnings:
        print(f"  - {w}")

print()
section("END OF REPORT")

sys.exit(0 if not errors else 1)
PY

print_info "Running CUDA validation..."
set +e
venv/bin/python test_cuda.py
TEST_STATUS=$?
set -e

echo
if [ "$TEST_STATUS" -eq 0 ]; then
  cat <<'EOF'
================================================================================
CUDA validation succeeded
By ParisNeo
================================================================================

Your installation looks usable for vLLM.

Next steps:
- Activate the environment:
    source venv/bin/activate
- Test a model server:
    python -m vllm.entrypoints.openai.api_server --model <your_model>

Optional but recommended for Hugging Face downloads:
- Create a token at: https://huggingface.co/settings/tokens
- Login with:
    hf auth login
  or:
    export HF_TOKEN="hf_xxxxxxxxxxxxxxxxx"
EOF
else
  cat <<'EOF'
================================================================================
CUDA validation reported problems
By ParisNeo
================================================================================

Common causes:
- NVIDIA driver missing or broken
- torch installed without CUDA support
- driver/runtime mismatch
- CUDA_VISIBLE_DEVICES hides your GPUs
- container or VM has no GPU passthrough
- GPU is too old for current vLLM builds
- Python version is not ideal for your vLLM release

Recommended checks:
1. Run:
     nvidia-smi
2. Verify inside Python:
     python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
3. Check your visible devices:
     echo "$CUDA_VISIBLE_DEVICES"
4. Prefer Python 3.12 for recent vLLM installs.
5. Ensure your GPU is compute capability 7.0 or newer.
6. Recreate the venv if needed, then reinstall.

The environment was installed, but you should fix the reported issues before using vLLM.
EOF
fi
