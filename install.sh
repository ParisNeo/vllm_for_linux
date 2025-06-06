#!/bin/bash

# ==============================================================================
# vLLM All-in-One Installer for Ubuntu (v7 - Advanced Configuration & Summary)
# ==============================================================================
# This script will:
# 1. Check prerequisites and install system dependencies.
# 2. Create a dedicated user and directory structure.
# 3. Interactively ask for server and vLLM-specific configurations.
# 4. Install vLLM in an isolated Python environment using 'uv'.
# 5. Generate a 'run_server.sh' script and a system-wide 'vllm_help' command.
# 6. Optionally, create a systemd service with all the chosen configurations.
# 7. Display a final summary of the entire setup.
# ==============================================================================

# --- Default Configuration (can be overridden by user input) ---
VLLM_USER="vllm"
VLLM_HOME_DIR="/opt/vllm-server"
DEFAULT_PYTHON_VERSION="3.11"

# Server & vLLM Parameters
SERVER_HOST="0.0.0.0"
SERVER_PORT="8000"
GPU_MEMORY_UTILIZATION="0.90"
TENSOR_PARALLEL_SIZE="1"
MAX_MODEL_LEN="" # Leave empty for auto
DTYPE="auto"

# --- Globals ---
VENV_DIR="${VLLM_HOME_DIR}/.venv"
MODELS_DIR="${VLLM_HOME_DIR}/models"
PYTHON_TO_USE=""
UV_EXECUTABLE="${VLLM_HOME_DIR}/.local/bin/uv"
SERVICE_MODEL_LOCATION="" # To store the model choice for the final summary

# --- Color Codes ---
COLOR_GREEN='\033[0;32m'
COLOR_YELLOW='\033[1;33m'
COLOR_RED='\033[0;31m'
COLOR_CYAN='\033[0;36m'
COLOR_NC='\033[0m' # No Color

# --- Helper Functions ---
info() {    echo -e "${COLOR_GREEN}[INFO] $1${NC}"; }
warn() {    echo -e "${COLOR_YELLOW}[WARN] $1${NC}"; }
error(){    echo -e "${COLOR_RED}[ERROR] $1${NC}"; exit 1; }

# Ensure the script is run with sudo
if [ "$EUID" -ne 0 ]; then
  error "This script must be run as root. Please use 'sudo ./install.sh'"
fi

# --- Function Definitions ---

check_prerequisites() {
    info "Step 1: Checking prerequisites..."
    if ! grep -q "Ubuntu" /etc/os-release; then
        error "This script is designed for Ubuntu. Aborting."
    fi
    if ! command -v nvidia-smi &> /dev/null; then
        error "NVIDIA driver not found. Please install the appropriate NVIDIA drivers for your GPU."
    fi
    info "NVIDIA drivers found."
    if ! command -v bc &> /dev/null; then
        warn "'bc' command not found. Installing..."
        apt-get update &>/dev/null && apt-get install -y bc
    fi
    local COMPUTE_CAPABILITY
    COMPUTE_CAPABILITY=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
    if (( $(echo "$COMPUTE_CAPABILITY < 7.0" | bc -l) )); then
        error "Your GPU's compute capability is ${COMPUTE_CAPABILITY}, but vLLM requires 7.0 or higher."
    fi
    info "GPU Compute Capability ${COMPUTE_CAPABILITY} is compatible."
}

find_python_and_install_deps() {
    info "Step 2: Detecting Python and installing dependencies..."
    local SUPPORTED_PYTHONS=("3.12" "3.11" "3.10" "3.9")
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
    apt-get install -y python3-pip python3-venv curl wget >/dev/null
    info "System dependencies installed."

    info "Installing 'uv' Python package manager..."
    su -s /bin/bash -c "curl -LsSf https://astral.sh/uv/install.sh | sh" "$VLLM_USER"
    info "'uv' installed successfully for user '$VLLM_USER'."
}

ask_for_configurations() {
    info "Step 3: Configuring Server and vLLM Parameters..."
    
    read -p "Enter server host [${SERVER_HOST}]: " -r reply && SERVER_HOST=${reply:-$SERVER_HOST}
    read -p "Enter server port [${SERVER_PORT}]: " -r reply && SERVER_PORT=${reply:-$SERVER_PORT}
    read -p "Set GPU memory utilization (0.1 to 1.0) [${GPU_MEMORY_UTILIZATION}]: " -r reply && GPU_MEMORY_UTILIZATION=${reply:-$GPU_MEMORY_UTILIZATION}
    read -p "Set tensor parallel size (for multi-GPU) [${TENSOR_PARALLEL_SIZE}]: " -r reply && TENSOR_PARALLEL_SIZE=${reply:-$TENSOR_PARALLEL_SIZE}
    read -p "Set max model length (context size, leave empty for auto): " -r reply && MAX_MODEL_LEN=${reply:-$MAX_MODEL_LEN}
    read -p "Set model dtype (e.g., auto, float16, bfloat16) [${DTYPE}]: " -r reply && DTYPE=${reply:-$DTYPE}
}

setup_environment() {
    info "Step 4: Setting up user and directories..."
    if id "$VLLM_USER" &>/dev/null; then
        info "User '$VLLM_USER' already exists."
    else
        useradd -r -m -d "$VLLM_HOME_DIR" -s /bin/bash "$VLLM_USER"
        info "Created dedicated user '$VLLM_USER' with home directory '$VLLM_HOME_DIR'."
    fi
    mkdir -p "$VENV_DIR" "$MODELS_DIR"
    chown -R "$VLLM_USER:$VLLM_USER" "$VLLM_HOME_DIR"
    info "Directories created and permissions set."
}

install_vllm() {
    info "Step 5: Creating virtual environment and installing vLLM..."
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
        error "vLLM installation failed. Could not retrieve version after installation."
    fi
}

build_vllm_args() {
    local args="--gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} --dtype ${DTYPE}"
    if [ -n "$MAX_MODEL_LEN" ]; then
        args+=" --max-model-len ${MAX_MODEL_LEN}"
    fi
    echo "$args"
}

create_run_script() {
    info "Step 6: Creating the 'run_server.sh' script..."
    local run_script_path="${VLLM_HOME_DIR}/run_server.sh"
    local vllm_args
    vllm_args=$(build_vllm_args)

    cat <<EOF > "$run_script_path"
#!/bin/bash
set -e
# --- vLLM Server Runner ---
# This script starts the vLLM server with pre-defined configurations from the installer.
# It accepts a model identifier (Hugging Face ID or local path) and passes any
# additional arguments directly to the vLLM server command, overriding defaults.
VENV_DIR="${VENV_DIR}"
MODELS_DIR="${MODELS_DIR}"
HOST="${SERVER_HOST}"
PORT="${SERVER_PORT}"
# Default arguments from installer.
EXTRA_ARGS="${vllm_args}"
if [ -z "\$1" ]; then
    echo "ERROR: No model identifier provided."
    echo "Usage: \$0 [MODEL_IDENTIFIER_OR_PATH] [ADDITIONAL_VLLM_ARGS...]"
    exit 1
fi
MODEL_ID=\$1; shift; CLI_ARGS="\$@"
echo "Activating venv..."; source "\${VENV_DIR}/bin/activate"
echo "Starting vLLM server for model: \${MODEL_ID}"
export HF_HOME="\${MODELS_DIR}"; export HUGGING_FACE_HUB_TOKEN=\${HUGGING_FACE_HUB_TOKEN}
python -m vllm.entrypoints.openai.api_server --model "\${MODEL_ID}" --host "\${HOST}" --port "\${PORT}" \${EXTRA_ARGS} \${CLI_ARGS}
EOF
    chmod +x "$run_script_path"
    chown "$VLLM_USER:$VLLM_USER" "$run_script_path"
    info "Created 'run_server.sh' at '$run_script_path'."
}

create_help_command() {
    info "Step 7: Creating the system-wide 'vllm_help' command..."
    local help_script_path="/usr/local/bin/vllm_help"
    
    cat <<'EOF' > "$help_script_path"
#!/bin/bash
# --- vLLM Server Helper ---
G='\033[0;32m'; Y='\033[1;33m'; C='\033[0;36m'; NC='\033[0m'
echo -e "${C}--- vLLM Server Management Cheat Sheet ---${NC}"
echo -e "\n${Y}Managing the Service (systemd):${NC}"
echo -e "  ${G}sudo systemctl start vllm${NC}      # Start the server"
echo -e "  ${G}sudo systemctl stop vllm${NC}       # Stop the server"
echo -e "  ${G}sudo systemctl restart vllm${NC}    # Restart the server"
echo -e "  ${G}sudo systemctl status vllm${NC}     # Check the current status"
echo -e "  ${G}sudo journalctl -u vllm -f${NC}     # View live server logs"
echo -e "\n${Y}Verifying the Server is Running:${NC}"
echo -e "  Run this command to check the API endpoint:"
echo -e "  ${G}curl http://localhost:__SERVER_PORT__/v1/models${NC}"
echo -e "\n${Y}Manual Operation (for testing):${NC}"
echo -e "  1. Switch to the dedicated user: ${G}sudo su - __VLLM_USER__${NC}"
echo -e "  2. Run the server with a model: ${G}./run_server.sh model_id_or_path${NC}"
echo -e "\n${Y}Configuration & File Locations:${NC}"
echo -e "  - Service User:      ${C}__VLLM_USER__${NC}"
echo -e "  - Installation Dir:  ${C}__VLLM_HOME_DIR__${NC}"
echo -e "  - Service File:      ${C}/etc/systemd/system/vllm.service${NC}"
echo -e "\n${Y}How to Change the Service Configuration:${NC}"
echo -e "  1. Edit the service file: ${G}sudo nano /etc/systemd/system/vllm.service${NC}"
echo -e "  2. Find the 'ExecStart=' line and change the model or arguments."
echo -e "  3. Reload systemd and restart: ${G}sudo systemctl daemon-reload && sudo systemctl restart vllm${NC}"
EOF
    sed -i "s|__VLLM_USER__|${VLLM_USER}|g" "$help_script_path"
    sed -i "s|__SERVER_PORT__|${SERVER_PORT}|g" "$help_script_path"
    sed -i "s|__VLLM_HOME_DIR__|${VLLM_HOME_DIR}|g" "$help_script_path"
    chmod +x "$help_script_path"
    info "'vllm_help' command created. You can run it from anywhere."
}

setup_systemd_service() {
    info "Step 8: Optional - Setup systemd service."
    read -p "Do you want to create a systemd service to run the vLLM server on boot? (y/N) " -r CREATE_SERVICE_REPLY
    if [[ ! "$CREATE_SERVICE_REPLY" =~ ^[Yy] ]]; then warn "Skipping systemd service creation."; return; fi

    local model_location=""; local service_extra_args=""
    local vllm_args=$(build_vllm_args)

    while true; do
        read -p "Use a Hugging Face model ID or a local path for the service? (hf/local): " -r model_type
        case "$model_type" in
            [Hh][Ff])
                read -p "Enter the Hugging Face model identifier: " -r model_location
                if [ -n "$model_location" ]; then SERVICE_MODEL_LOCATION=$model_location; break; fi
                warn "Model ID cannot be empty."
                ;;
            [Ll][Oo][Cc][Aa][Ll])
                read -p "Enter the absolute path to your local model directory: " -r model_location
                if [ -z "$model_location" ]; then warn "Path cannot be empty."; continue; fi
                if [ ! -d "$model_location" ]; then warn "Directory not found."; continue; fi
                info "Ensuring '${VLLM_USER}' user has read access to '${model_location}'..."
                chmod -R a+rX "${model_location}"; info "Permissions updated."
                service_extra_args="--disable-log-stats"; SERVICE_MODEL_LOCATION=$model_location
                break
                ;;
            *) warn "Invalid input. Please enter 'hf' or 'local'.";;
        esac
    done

    local service_file="/etc/systemd/system/vllm.service"
    info "Creating systemd service file at '${service_file}'..."
    cat <<EOF > "$service_file"
[Unit]
Description=vLLM OpenAI-Compatible Server
After=network.target
[Service]
User=${VLLM_USER}
Group=${VLLM_USER}
WorkingDirectory=${VLLM_HOME_DIR}
ExecStart=${VLLM_HOME_DIR}/run_server.sh "${SERVICE_MODEL_LOCATION}" ${vllm_args} ${service_extra_args}
Restart=always
RestartSec=10
[Install]
WantedBy=multi-user.target
EOF
    systemctl daemon-reload; systemctl enable vllm.service; info "Systemd service 'vllm.service' created and enabled."; warn "To start it, run: sudo systemctl start vllm"
}

display_final_summary() {
    info "========================================================"
    info "          vLLM Installation Complete!                   "
    info "========================================================"
    echo
    printf "${COLOR_CYAN}%-25s${NC} %-s\n" "vLLM User:" "$VLLM_USER"
    printf "${COLOR_CYAN}%-25s${NC} %-s\n" "Installation Directory:" "$VLLM_HOME_DIR"
    printf "${COLOR_CYAN}%-25s${NC} %-s\n" "Python Environment:" "$VENV_DIR"
    echo
    printf "${COLOR_YELLOW}%-s${NC}\n" "--- Server Configuration ---"
    printf "${COLOR_CYAN}%-25s${NC} %-s\n" "Host:" "$SERVER_HOST"
    printf "${COLOR_CYAN}%-25s${NC} %-s\n" "Port:" "$SERVER_PORT"
    echo
    printf "${COLOR_YELLOW}%-s${NC}\n" "--- vLLM Core Parameters ---"
    printf "${COLOR_CYAN}%-25s${NC} %-s\n" "GPU Memory Utilization:" "$GPU_MEMORY_UTILIZATION"
    printf "${COLOR_CYAN}%-25s${NC} %-s\n" "Tensor Parallel Size:" "$TENSOR_PARALLEL_SIZE"
    printf "${COLOR_CYAN}%-25s${NC} %-s\n" "Max Model Length:" "${MAX_MODEL_LEN:-auto}"
    printf "${COLOR_CYAN}%-25s${NC} %-s\n" "Data Type (dtype):" "$DTYPE"
    if [ -n "$SERVICE_MODEL_LOCATION" ]; then
        echo
        printf "${COLOR_YELLOW}%-s${NC}\n" "--- Systemd Service ---"
        printf "${COLOR_CYAN}%-25s${NC} %-s\n" "Service Enabled:" "Yes"
        printf "${COLOR_CYAN}%-25s${NC} %-s\n" "Model:" "$SERVICE_MODEL_LOCATION"
    fi
    echo
    info "A system-wide command ${COLOR_CYAN}vllm_help${NC} has been created."
    info "Run it from anywhere to get a cheat sheet on how to manage your server."
    echo
}

# --- Main Execution Flow ---
main() {
    check_prerequisites
    find_python_and_install_deps
    ask_for_configurations
    setup_environment
    install_vllm
    create_run_script
    create_help_command
    setup_systemd_service
    display_final_summary
}

main "$@"
