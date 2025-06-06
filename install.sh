#!/bin/bash

# ==============================================================================
# vLLM All-in-One Installer for Ubuntu (Improved Version)
# ==============================================================================
# This script will:
# 1. Check for prerequisites (Ubuntu, NVIDIA GPU, CUDA drivers).
# 2. Check for an existing compatible Python version (3.9-3.12).
# 3. If no Python is found, it will rely on 'uv' to download a self-contained build.
# 4. Install dependencies like 'uv' package manager.
# 5. Create a dedicated user and directory structure.
# 6. Set up a Python virtual environment and install vLLM.
# 7. Generate a 'run_server.sh' script to start the OpenAI-compatible server.
# 8. Optionally, create and enable a systemd service to run vLLM on boot.
# ==============================================================================

# --- Script Configuration ---
VLLM_USER="vllm"
VLLM_HOME_DIR="/opt/vllm-server"
VENV_DIR="${VLLM_HOME_DIR}/.venv"
MODELS_DIR="${VLLM_HOME_DIR}/models"
# Default Python to use if none is found. Must be in the supported list below.
DEFAULT_PYTHON_VERSION="3.11"
SERVER_HOST="0.0.0.0"
SERVER_PORT="8000"

# --- Globals ---
PYTHON_TO_USE="" # This will be populated by the Python check

# --- Color Codes for Output ---
COLOR_GREEN='\033[0;32m'
COLOR_YELLOW='\033[1;33m'
COLOR_RED='\033[0;31m'
COLOR_NC='\033[0m'

# --- Helper Functions ---
info() {
    echo -e "${COLOR_GREEN}[INFO] $1${COLOR_NC}"
}

warn() {
    echo -e "${COLOR_YELLOW}[WARN] $1${COLOR_NC}"
}

error() {
    echo -e "${COLOR_RED}[ERROR] $1${COLOR_NC}"
    exit 1
}

# Ensure the script is run with sudo
if [ "$EUID" -ne 0 ]; then
  error "This script must be run as root. Please use 'sudo ./install.sh'"
fi

# --- Main Installation Logic ---

check_prerequisites() {
    info "Step 1: Checking prerequisites..."

    # Check for Ubuntu
    if ! grep -q "Ubuntu" /etc/os-release; then
        error "This script is designed for Ubuntu. Aborting."
    fi

    # Check for NVIDIA GPU and drivers
    if ! command -v nvidia-smi &> /dev/null; then
        error "NVIDIA driver not found. Please install the appropriate NVIDIA drivers for your GPU."
    fi
    info "NVIDIA drivers found."

    # Check for Compute Capability
    COMPUTE_CAPABILITY=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
    if (( $(echo "$COMPUTE_CAPABILITY < 7.0" | bc -l) )); then
        error "Your GPU's compute capability is ${COMPUTE_CAPABILITY}, but vLLM requires 7.0 or higher."
    fi
    info "GPU Compute Capability ${COMPUTE_CAPABILITY} is compatible."
}

find_python_and_install_deps() {
    info "Step 2: Detecting Python and installing dependencies..."

    # Find a suitable Python version
    # We check in reverse order to prefer newer versions.
    SUPPORTED_PYTHONS=("3.12" "3.11" "3.10" "3.9")
    for version in "${SUPPORTED_PYTHONS[@]}"; do
        if command -v "python${version}" &> /dev/null; then
            info "Found compatible system Python: python${version}"
            PYTHON_TO_USE="$version"
            break
        fi
    done

    if [ -z "$PYTHON_TO_USE" ]; then
        warn "No system-wide Python found in the supported range (3.9-3.12)."
        info "'uv' will automatically download and manage a standalone Python build."
        PYTHON_TO_USE="${DEFAULT_PYTHON_VERSION}"
    fi
    info "Will proceed using Python ${PYTHON_TO_USE} for the virtual environment."

    # Install core dependencies. We no longer force a specific Python version from apt.
    apt-get update
    apt-get install -y python3-pip python3-venv curl wget
    info "System dependencies installed."

    info "Installing 'uv' Python package manager..."
    # Install uv for the target user to avoid permission issues
    su -s /bin/bash -c "curl -LsSf https://astral.sh/uv/install.sh | sh" "$VLLM_USER"
    info "'uv' installed successfully for user '$VLLM_USER'."
}

setup_environment() {
    info "Step 3: Setting up user and directories..."

    # Create dedicated user if it doesn't exist
    if id "$VLLM_USER" &>/dev/null; then
        info "User '$VLLM_USER' already exists."
    else
        # The user's home directory will be the main vLLM directory
        useradd -r -m -d "$VLLM_HOME_DIR" -s /bin/bash "$VLLM_USER"
        info "Created dedicated user '$VLLM_USER' with home directory '$VLLM_HOME_DIR'."
    fi

    mkdir -p "$VENV_DIR"
    mkdir -p "$MODELS_DIR"
    # Ensure ownership is correct, especially if user already existed
    chown -R "$VLLM_USER:$VLLM_USER" "$VLLM_HOME_DIR"
    info "Created directories and set permissions."
}

install_vllm() {
    info "Step 4: Creating virtual environment and installing vLLM..."

    info "Creating Python virtual environment using Python ${PYTHON_TO_USE}..."
    su -s /bin/bash -c "cd '$VLLM_HOME_DIR' && /home/${VLLM_USER}/.cargo/bin/uv venv --python $PYTHON_TO_USE '$VENV_DIR'" "$VLLM_USER"
    info "Virtual environment created at '$VENV_DIR'."

    info "Installing vLLM into the virtual environment. This may take a few minutes..."
    # The command is run in a subshell as the target user to handle activation and installation
    su -s /bin/bash -c "source '${VENV_DIR}/bin/activate' && /home/${VLLM_USER}/.cargo/bin/uv pip install vllm --torch-backend=auto" "$VLLM_USER"

    # Verify installation
    local vllm_check
    vllm_check=$(su -s /bin/bash -c "source '${VENV_DIR}/bin/activate' && python -m vllm --version" "$VLLM_USER")
    if [[ $vllm_check == *"vllm version"* ]]; then
        info "vLLM installed successfully. Version: ${vllm_check}"
    else
        error "vLLM installation failed. Please check the logs."
    fi
}

create_run_script() {
    info "Step 5: Creating the 'run_server.sh' script..."
    local run_script_path="${VLLM_HOME_DIR}/run_server.sh"

    cat <<EOF > "$run_script_path"
#!/bin/bash
set -e

# --- vLLM Server Runner ---
# This script activates the virtual environment and starts the vLLM OpenAI-compatible server.

# Usage:
# ./run_server.sh [MODEL_IDENTIFIER]
# Example: ./run_server.sh meta-llama/Llama-2-7b-chat-hf

# --- Configuration ---
VENV_DIR="${VENV_DIR}"
MODELS_DIR="${MODELS_DIR}"
HOST="${SERVER_HOST}"
PORT="${SERVER_PORT}"
# Add other vLLM CLI options here if needed, e.g., --tensor-parallel-size
EXTRA_ARGS="--gpu-memory-utilization 0.90"

# --- Script Logic ---
if [ -z "\$1" ]; then
    echo "ERROR: No model identifier provided."
    echo "Usage: \$0 [MODEL_IDENTIFIER]"
    echo "Example: \$0 meta-llama/Llama-2-7b-chat-hf"
    exit 1
fi

MODEL_ID=\$1
shift # The rest of the arguments can be passed to vLLM
CLI_ARGS="\$@"

echo "Activating virtual environment..."
source "\${VENV_DIR}/bin/activate"

echo "Starting vLLM server for model: \${MODEL_ID}"
echo "Host: \${HOST}, Port: \${PORT}"
echo "Model cache directory: \${MODELS_DIR}"

# Set Hugging Face home to custom models directory
export HF_HOME="\${MODELS_DIR}"
export HUGGING_FACE_HUB_TOKEN=\${HUGGING_FACE_HUB_TOKEN} # Use environment variable for token

python -m vllm.entrypoints.openai.api_server \\
    --model "\${MODEL_ID}" \\
    --host "\${HOST}" \\
    --port "\${PORT}" \\
    \${EXTRA_ARGS} \\
    \${CLI_ARGS}
EOF

    chmod +x "$run_script_path"
    chown "$VLLM_USER:$VLLM_USER" "$run_script_path"
    info "Created 'run_server.sh' at '$run_script_path'."
}

setup_systemd_service() {
    info "Step 6: Optional - Setup systemd service."
    read -p "Do you want to create a systemd service to run the vLLM server on boot? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        warn "Skipping systemd service creation."
        return
    fi

    read -p "Enter the Hugging Face model identifier for the service (e.g., meta-llama/Llama-2-7b-chat-hf): " MODEL_FOR_SERVICE
    if [ -z "$MODEL_FOR_SERVICE" ]; then
        error "Model identifier cannot be empty. Aborting service creation."
    fi

    local service_file="/etc/systemd/system/vllm.service"
    info "Creating systemd service file at '$service_file'..."

    cat <<EOF > "$service_file"
[Unit]
Description=vLLM OpenAI-Compatible Server
After=network.target

[Service]
User=${VLLM_USER}
Group=${VLLM_USER}
WorkingDirectory=${VLLM_HOME_DIR}
ExecStart=${VLLM_HOME_DIR}/run_server.sh ${MODEL_FOR_SERVICE}
# You can set your Hugging Face token here if required for private models
# Environment="HUGGING_FACE_HUB_TOKEN=hf_..."
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    systemctl enable vllm.service

    info "Systemd service 'vllm.service' created and enabled."
    warn "The service is enabled but not started. You can start it with: sudo systemctl start vllm"
    info "To check the status and logs, use: sudo systemctl status vllm"
    info "To see live logs, use: sudo journalctl -u vllm -f"
}

# --- Main Execution Flow ---
main() {
    check_prerequisites
    # The order is important: create user first, then install deps for that user.
    setup_environment
    find_python_and_install_deps
    install_vllm
    create_run_script
    setup_systemd_service

    echo
    info "========================================================"
    info "          vLLM Installation Complete!                   "
    info "========================================================"
    echo
    info "Key files and directories:"
    echo -e "  - Installation Directory: ${COLOR_YELLOW}${VLLM_HOME_DIR}${COLOR_NC}"
    echo -e "  - Models Directory: ${COLOR_YELLOW}${MODELS_DIR}${COLOR_NC}"
    echo -e "  - Run Script: ${COLOR_YELLOW}${VLLM_HOME_DIR}/run_server.sh${COLOR_NC}"
    echo
    info "Next Steps:"
    echo -e "1. ${COLOR_YELLOW}Switch to the vLLM user:${COLOR_NC} sudo su - ${VLLM_USER}"
    echo -e "2. ${COLOR_YELLOW}(Optional) Set your Hugging Face token:${COLOR_NC} export HUGGING_FACE_HUB_TOKEN='your_token_here'"
    echo -e "3. ${COLOR_YELLOW}Run the server manually:${COLOR_NC} ./run_server.sh meta-llama/Llama-2-7b-chat-hf"
    echo
    info "If you created the systemd service:"
    echo -e "  - To start the service: ${COLOR_YELLOW}sudo systemctl start vllm${COLOR_NC}"
    echo -e "  - To check its status: ${COLOR_YELLOW}sudo systemctl status vllm${COLOR_NC}"
    echo -e "  - To view logs: ${COLOR_YELLOW}sudo journalctl -u vllm -f${COLOR_NC}"
    echo
}

main "$@"
