#!/bin/bash

# ==============================================================================
# vLLM All-in-One Installer for Ubuntu (v5 - Flexible Service Creation)
# ==============================================================================
# This script will:
# 1. Check for prerequisites (Ubuntu, NVIDIA GPU, CUDA drivers).
# 2. Check for an existing compatible Python version (3.9-3.12).
# 3. If no Python is found, rely on 'uv' to download a self-contained build.
# 4. Install dependencies like the 'uv' package manager.
# 5. Create a dedicated user and directory structure.
# 6. Set up a Python virtual environment and install vLLM.
# 7. Generate a 'run_server.sh' script to start the OpenAI-compatible server.
# 8. Optionally, create and enable a systemd service that can use either a
#    Hugging Face ID or a local model path.
# ==============================================================================

# --- Script Configuration ---
VLLM_USER="vllm"
VLLM_HOME_DIR="/opt/vllm-server"
VENV_DIR="${VLLM_HOME_DIR}/.venv"
MODELS_DIR="${VLLM_HOME_DIR}/models"
DEFAULT_PYTHON_VERSION="3.11"
SERVER_HOST="0.0.0.0"
SERVER_PORT="8000"

# --- Globals ---
PYTHON_TO_USE=""
UV_EXECUTABLE="${VLLM_HOME_DIR}/.local/bin/uv"

# --- Color Codes for Output ---
COLOR_GREEN='\033[0;32m'
COLOR_YELLOW='\033[1;33m'
COLOR_RED='\033[0;31m'
COLOR_NC='\033[0m'

# --- Helper Functions ---
info() {    echo -e "${COLOR_GREEN}[INFO] $1${COLOR_NC}"; }
warn() {    echo -e "${COLOR_YELLOW}[WARN] $1${COLOR_NC}"; }
error() {   echo -e "${COLOR_RED}[ERROR] $1${COLOR_NC}"; exit 1; }

# Ensure the script is run with sudo
if [ "$EUID" -ne 0 ]; then
  error "This script must be run as root. Please use 'sudo ./install.sh'"
fi

# --- Main Installation Logic ---

check_prerequisites() {
    info "Step 1: Checking prerequisites..."
    if ! grep -q "Ubuntu" /etc/os-release; then error "This script is designed for Ubuntu. Aborting."; fi
    if ! command -v nvidia-smi &> /dev/null; then error "NVIDIA driver not found. Please install the appropriate NVIDIA drivers for your GPU."; fi
    info "NVIDIA drivers found."
    if ! command -v bc &> /dev/null; then apt-get update && apt-get install -y bc; fi
    COMPUTE_CAPABILITY=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
    if (( $(echo "$COMPUTE_CAPABILITY < 7.0" | bc -l) )); then
        error "Your GPU's compute capability is ${COMPUTE_CAPABILITY}, but vLLM requires 7.0 or higher."
    fi
    info "GPU Compute Capability ${COMPUTE_CAPABILITY} is compatible."
}

find_python_and_install_deps() {
    info "Step 2: Detecting Python and installing dependencies..."
    SUPPORTED_PYTHONS=("3.12" "3.11" "3.10" "3.9")
    for version in "${SUPPORTED_PYTHONS[@]}"; do
        if command -v "python${version}" &> /dev/null; then
            info "Found compatible system Python: python${version}"
            PYTHON_TO_USE="$version"; break
        fi
    done
    if [ -z "$PYTHON_TO_USE" ]; then
        warn "No system-wide Python found in the supported range (3.9-3.12)."
        info "'uv' will automatically download and manage a standalone Python build."
        PYTHON_TO_USE="${DEFAULT_PYTHON_VERSION}"
    fi
    info "Will proceed using Python ${PYTHON_TO_USE} for the virtual environment."
    apt-get update >/dev/null
    apt-get install -y python3-pip python3-venv curl wget
    info "System dependencies installed."

    info "Installing 'uv' Python package manager..."
    su -s /bin/bash -c "curl -LsSf https://astral.sh/uv/install.sh | sh" "$VLLM_USER"
    info "'uv' installed successfully for user '$VLLM_USER'."
}

setup_environment() {
    info "Step 3: Setting up user and directories..."
    if id "$VLLM_USER" &>/dev/null; then
        info "User '$VLLM_USER' already exists."
    else
        useradd -r -m -d "$VLLM_HOME_DIR" -s /bin/bash "$VLLM_USER"
        info "Created dedicated user '$VLLM_USER' with home directory '$VLLM_HOME_DIR'."
    fi
    mkdir -p "$VENV_DIR" "$MODELS_DIR"
    chown -R "$VLLM_USER:$VLLM_USER" "$VLLM_HOME_DIR"
    info "Created directories and set permissions."
}

install_vllm() {
    info "Step 4: Creating virtual environment and installing vLLM..."
    info "Creating Python virtual environment using Python ${PYTHON_TO_USE}..."
    su -s /bin/bash -c "cd '$VLLM_HOME_DIR' && '${UV_EXECUTABLE}' venv --python $PYTHON_TO_USE '$VENV_DIR' --seed" "$VLLM_USER"
    info "Virtual environment created at '$VENV_DIR'."

    info "Installing vLLM into the virtual environment. This may take a few minutes..."
    su -s /bin/bash -c "source '${VENV_DIR}/bin/activate' && '${UV_EXECUTABLE}' pip install vllm --torch-backend=auto" "$VLLM_USER"

    info "Verifying vLLM installation..."
    local vllm_version
    vllm_version=$(su -s /bin/bash -c "source '${VENV_DIR}/bin/activate' && python -c 'import vllm; print(vllm.__version__)'" "$VLLM_USER" 2>/dev/null)
    if [ -n "$vllm_version" ]; then
        info "vLLM installed successfully. Version: ${vllm_version}"
    else
        error "vLLM installation failed. Could not retrieve version after installation. Please check the logs."
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
# It accepts a model identifier (Hugging Face ID or local path) and passes any
# additional arguments directly to the vLLM server command.

# --- Configuration ---
VENV_DIR="${VENV_DIR}"
MODELS_DIR="${MODELS_DIR}"
HOST="${SERVER_HOST}"
PORT="${SERVER_PORT}"
# Default arguments can be added here
EXTRA_ARGS="--gpu-memory-utilization 0.90"

if [ -z "\$1" ]; then
    echo "ERROR: No model identifier provided."
    echo "Usage: \$0 [MODEL_IDENTIFIER_OR_PATH] [ADDITIONAL_VLLM_ARGS...]"
    echo "Example (HF):   \$0 meta-llama/Llama-2-7b-chat-hf"
    echo "Example (Local):\$0 /path/to/my/model --tensor-parallel-size 2"
    exit 1
fi

MODEL_ID=\$1; shift; CLI_ARGS="\$@"

echo "Activating virtual environment..."; source "\${VENV_DIR}/bin/activate"
echo "Starting vLLM server for model: \${MODEL_ID}"
export HF_HOME="\${MODELS_DIR}" # For Hugging Face models, they will be cached here
export HUGGING_FACE_HUB_TOKEN=\${HUGGING_FACE_HUB_TOKEN}
python -m vllm.entrypoints.openai.api_server --model "\${MODEL_ID}" --host "\${HOST}" --port "\${PORT}" \${EXTRA_ARGS} \${CLI_ARGS}
EOF
    chmod +x "$run_script_path"
    chown "$VLLM_USER:$VLLM_USER" "$run_script_path"
    info "Created 'run_server.sh' at '$run_script_path'."
}

setup_systemd_service() {
    info "Step 6: Optional - Setup systemd service."
    # Use a standard read command to avoid issues with '-n 1' and the Enter key
    read -p "Do you want to create a systemd service to run the vLLM server on boot? (y/N) " -r CREATE_SERVICE_REPLY
    if [[ ! "$CREATE_SERVICE_REPLY" =~ ^[Yy]([Ee][Ss])?$ ]]; then
        warn "Skipping systemd service creation."
        return
    fi

    local model_location=""
    local extra_service_args=""

    while true; do
        read -p "Use a Hugging Face model ID or a local path for the service? (hf/local): " -r model_type
        case "$model_type" in
            [Hh][Ff])
                read -p "Enter the Hugging Face model identifier: " -r model_location
                if [ -z "$model_location" ]; then
                    warn "Model identifier cannot be empty."
                    continue
                fi
                break
                ;;
            [Ll][Oo][Cc][Aa][Ll])
                read -p "Enter the absolute path to your local model directory: " -r model_location
                if [ -z "$model_location" ]; then
                    warn "Path cannot be empty."
                    continue
                fi
                if [ ! -d "$model_location" ]; then
                    warn "Directory not found at '${model_location}'. Please provide a valid absolute path."
                    continue
                fi
                info "Ensuring '${VLLM_USER}' user has read access to '${model_location}'..."
                # Grant read access to all files and read/execute to all directories for everyone.
                # This is a safe and non-intrusive way to grant access.
                chmod -R a+rX "${model_location}"
                info "Permissions updated."
                extra_service_args="--disable-log-stats"
                info "Will use '--disable-log-stats' for a 100% local service."
                break
                ;;
            *)
                warn "Invalid input. Please enter 'hf' or 'local'."
                ;;
        esac
    done

    local service_file="/etc/systemd/system/vllm.service"
    info "Creating systemd service file at '${service_file}'..."
    # Note the quotes around model_location to handle paths with spaces
    cat <<EOF > "$service_file"
[Unit]
Description=vLLM OpenAI-Compatible Server
After=network.target

[Service]
User=${VLLM_USER}
Group=${VLLM_USER}
WorkingDirectory=${VLLM_HOME_DIR}
ExecStart=${VLLM_HOME_DIR}/run_server.sh "${model_location}" ${extra_service_args}
# To use a Hugging Face token for private models, uncomment and set the following line:
# Environment="HUGGING_FACE_HUB_TOKEN=hf_..."
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    systemctl enable vllm.service
    info "Systemd service 'vllm.service' created and enabled."
    warn "The service is enabled but not started. To start it, run: sudo systemctl start vllm"
    info "To check the status and logs, use: sudo systemctl status vllm"
}

# --- Main Execution Flow ---
main() {
    check_prerequisites
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
    echo -e "  - Models Directory (for HF cache): ${COLOR_YELLOW}${MODELS_DIR}${COLOR_NC}"
    echo -e "  - Run Script: ${COLOR_YELLOW}${VLLM_HOME_DIR}/run_server.sh${COLOR_NC}"
    echo
    info "Next Steps:"
    echo -e "1. ${COLOR_YELLOW}Switch to the vLLM user:${COLOR_NC} sudo su - ${VLLM_USER}"
    echo -e "2. ${COLOR_YELLOW}Run the server manually:${COLOR_NC} ./run_server.sh meta-llama/Llama-2-7b-chat-hf"
    echo
    info "If you created the systemd service:"
    echo -e "  - To start the service: ${COLOR_YELLOW}sudo systemctl start vllm${COLOR_NC}"
    echo -e "  - To check its status: ${COLOR_YELLOW}sudo systemctl status vllm${COLOR_NC}"
}

main "$@"
