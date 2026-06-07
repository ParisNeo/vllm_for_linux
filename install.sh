#!/usr/bin/env bash
set -euo pipefail

TARGET_PYTHON="${TARGET_PYTHON:-3.12}"
USE_UV=0
ACTIVATE_PATH="venv/bin/activate"
PYTHON_BIN=""
BOOTSTRAP_MODE=""

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

print_info() { echo "[INFO] $1"; }
print_warn() { echo "[WARN] $1"; }
print_error() { echo "[ERROR] $1"; }

python_version_ok() {
  local py="$1"
  "$py" - <<'PY'
import sys
ok = (sys.version_info.major == 3 and sys.version_info.minor >= 10)
raise SystemExit(0 if ok else 1)
PY
}

have_venv_support() {
  local py="$1"
  "$py" - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("venv") else 1)
PY
}

try_install_uv() {
  print_info "uv not found. Trying to install it..."
  if command -v curl >/dev/null 2>&1; then
    if curl -LsSf https://astral.sh/uv/install.sh | sh; then
      return 0
    fi
  fi

  if command -v wget >/dev/null 2>&1; then
    if wget -qO- https://astral.sh/uv/install.sh | sh; then
      return 0
    fi
  fi

  return 1
}

refresh_path_for_uv() {
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
}

try_install_venv_system_package() {
  if ! command -v sudo >/dev/null 2>&1; then
    return 1
  fi

  if command -v apt-get >/dev/null 2>&1; then
    print_info "Trying to install python3-venv with apt..."
    sudo apt-get update && sudo apt-get install -y python3-venv
    return 0
  fi

  if command -v dnf >/dev/null 2>&1; then
    print_info "Trying to install python3-venv/python3 with dnf..."
    sudo dnf install -y python3
    return 0
  fi

  if command -v yum >/dev/null 2>&1; then
    print_info "Trying to install python3 with yum..."
    sudo yum install -y python3
    return 0
  fi

  if command -v pacman >/dev/null 2>&1; then
    print_info "Trying to install python-virtualenv with pacman..."
    sudo pacman -Sy --noconfirm python-virtualenv
    return 0
  fi

  return 1
}

setup_with_uv() {
  USE_UV=1
  BOOTSTRAP_MODE="uv"

  print_info "Using uv-managed Python ${TARGET_PYTHON}..."
  uv python install "${TARGET_PYTHON}"
  uv venv --python "${TARGET_PYTHON}" venv
  # shellcheck disable=SC1091
  source "$ACTIVATE_PATH"
  PYTHON_BIN="$(command -v python)"
}

setup_with_system_python() {
  local py="$1"
  BOOTSTRAP_MODE="system-python"

  print_info "Using system Python: $("$py" --version 2>&1)"
  "$py" -m venv venv
  # shellcheck disable=SC1091
  source "$ACTIVATE_PATH"
  PYTHON_BIN="$(command -v python)"
}

write_test_cuda() {
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

def check_python_runtime():
    section("PYTHON RUNTIME")
    print(f"Python executable : {sys.executable}")
    print(f"Python version    : {sys.version}")
    if sys.version_info.major != 3 or sys.version_info.minor < 10:
        error("Python is too old for current vLLM releases. Use Python 3.10+.")
    elif sys.version_info.minor == 12:
        info("Python 3.12 detected. Good match for many recent vLLM installs.")
    elif sys.version_info.minor > 12:
        warn("Python is newer than many documented examples. If issues appear, try Python 3.12.")

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
            warn("nvcc not found in PATH. This is acceptable for many wheel installs.")
    except Exception as ex:
        warn(f"Failed to run nvcc --version: {ex}")

check_python_runtime()
print()

section("SYSTEM")
analyze_cuda_visible_devices()
print(f"LD_LIBRARY_PATH      : {os.environ.get('LD_LIBRARY_PATH')}")
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
try:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    print(result.stdout)
    if result.returncode != 0 or not result.stdout.strip():
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
        print(f"torch.cuda.is_available()   : {available}")
        if not available:
            error("torch.cuda.is_available() is False.")
    except Exception as ex:
        print(f"torch.cuda.is_available()   : ERROR -> {ex}")
        error(f"torch.cuda.is_available() failed: {ex}")

    try:
        count = torch.cuda.device_count()
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
        print("  Advice: use Python 3.10+; Python 3.12 is a good default.")
    if any("does not appear to include CUDA support" in e for e in errors):
        print("  Advice: reinstall in a fresh venv and verify you installed vLLM/PyTorch with CUDA support.")
    if any("torch.cuda.is_available() is False." in e for e in errors):
        print("  Advice: check driver installation, GPU passthrough, and CUDA_VISIBLE_DEVICES.")
    if any("zero CUDA devices" in e or "device_count" in e for e in errors):
        print("  Advice: verify GPU access permissions and that CUDA_VISIBLE_DEVICES is not hiding devices.")
    if any("torch.cuda.init()" in e for e in errors):
        print("  Advice: this often indicates driver/runtime mismatch or broken CUDA libraries.")
    if any("CUDA compute test failed" in e for e in errors):
        print("  Advice: CUDA initialization may work while kernels still fail; check compatibility.")
    if any("CUDA_VISIBLE_DEVICES is set to an empty string" in e or "hides CUDA devices" in e for e in errors):
        print("  Advice: unset CUDA_VISIBLE_DEVICES or set valid indices like 0 or 0,1.")
if compute_caps and any(major < 7 for major, _ in compute_caps):
    print("- One or more GPUs are below compute capability 7.0.")
    print("  Advice: vLLM generally targets Volta/Turing/Ampere/Hopper-class GPUs or newer.")
if not errors and torch_ok:
    print("Everything looks OK for CUDA/PyTorch basic usage.")
    print("vLLM should run if the chosen model fits GPU memory.")
if warnings:
    print()
    print("Warnings:")
    for w in warnings:
        print(f"  - {w}")

print()
section("END OF REPORT")
sys.exit(0 if not errors else 1)
PY
}

print_manual_uv_help() {
  cat <<'EOF'
================================================================================
uv installation required
By ParisNeo
================================================================================

Automatic setup could not use uv, and the fallback Python environment path was not usable.

Please install uv, then rerun this script:

  curl -LsSf https://astral.sh/uv/install.sh | sh

Or with wget:

  wget -qO- https://astral.sh/uv/install.sh | sh

Then restart your shell if needed and run:
  ./install.sh
EOF
}

print_banner

refresh_path_for_uv
if command -v uv >/dev/null 2>&1; then
  print_info "uv detected."
  setup_with_uv
else
  if try_install_uv; then
    refresh_path_for_uv
  fi

  if command -v uv >/dev/null 2>&1; then
    print_info "uv installed successfully."
    setup_with_uv
  else
    print_warn "uv could not be installed automatically. Trying system Python fallback..."

    SYS_PYTHON=""
    if command -v python3 >/dev/null 2>&1; then
      SYS_PYTHON="python3"
    elif command -v python >/dev/null 2>&1; then
      SYS_PYTHON="python"
    fi

    if [[ -z "$SYS_PYTHON" ]]; then
      print_manual_uv_help
      exit 1
    fi

    if ! python_version_ok "$SYS_PYTHON"; then
      print_error "System Python is too old for vLLM fallback."
      print_manual_uv_help
      exit 1
    fi

    if ! have_venv_support "$SYS_PYTHON"; then
      print_warn "System Python is acceptable, but venv is missing. Trying to install it..."
      if ! try_install_venv_system_package; then
        print_error "Could not install venv support automatically."
        print_manual_uv_help
        exit 1
      fi
      if ! have_venv_support "$SYS_PYTHON"; then
        print_error "venv is still unavailable after installation attempt."
        print_manual_uv_help
        exit 1
      fi
    fi

    setup_with_system_python "$SYS_PYTHON"
  fi
fi

print_info "Installing required Python packages..."
python -m pip install --upgrade pip
python -m pip install -U \
  vllm \
  huggingface_hub \
  ascii-colors

write_test_cuda

print_info "Running CUDA validation..."
set +e
python test_cuda.py
TEST_STATUS=$?
set -e

echo
if [ "$TEST_STATUS" -eq 0 ]; then
  cat <<EOF
================================================================================
CUDA validation succeeded
By ParisNeo
================================================================================

Environment bootstrap mode: ${BOOTSTRAP_MODE}

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
  cat <<EOF
================================================================================
CUDA validation reported problems
By ParisNeo
================================================================================

Environment bootstrap mode: ${BOOTSTRAP_MODE}

Common causes:
- NVIDIA driver missing or broken
- torch installed without CUDA support
- driver/runtime mismatch
- CUDA_VISIBLE_DEVICES hides your GPUs
- container or VM has no GPU passthrough
- GPU is too old for current vLLM builds
- Python version is not suitable for fallback mode

Recommended checks:
1. Run:
     nvidia-smi
2. Verify inside Python:
     python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
3. Check your visible devices:
     echo "\$CUDA_VISIBLE_DEVICES"
4. Prefer uv with Python ${TARGET_PYTHON}.
5. Ensure your GPU is compute capability 7.0 or newer.
6. Recreate the venv if needed, then reinstall.

The environment was installed, but you should fix the reported issues before using vLLM.
EOF
fi
