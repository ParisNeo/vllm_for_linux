#!/bin/bash

# ==============================================================================
# vLLM All-in-One Installer for Ubuntu (v6 - With vllm_help Command)
# ==============================================================================
# This script will:
# 1. Check for prerequisites (Ubuntu, NVIDIA GPU, CUDA drivers).
# 2. Check for an existing compatible Python version (3.9-3.12).
# 3. If no Python is found, rely on 'uv' to download a self-contained build.
# 4. Install dependencies and the 'uv' package manager.
# 5. Create a dedicated user and directory structure.
# 6. Set up a Python virtual environment and install vLLM.
# 7. Generate a 'run_server.sh' script and a system-wide 'vllm_help' command.
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
COLOR_CYAN='\033[0;36m'
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
    # ... Function content is unchanged, omitted for brevity ...
    local run_script_path="${VLLM_HOME_DIR}/run_server.sh"
    cat <<EOF > "$run_script_path"
#!/bin/bash
set -e
VENV_DIR="${VENV_DIR}"; MODELS_DIR="${MODELS_DIR}"; HOST="${SERVER_HOST}"; PORT="${SERVER_PORT}"; EXTRA_ARGS="--gpu-memory-utilization 0.90"
if [ -z "\$1" ]; then echo "ERROR: No model identifier provided."; echo "Usage: \$0 [MODEL_ID] [ARGS...]"; exit 1; fi
MODEL_ID=\$1; shift; CLI_ARGS="\$@"; echo "Activating venv..."; source "\${VENV_DIR}/bin/activate"
echo "Starting vLLM server for model: \${MODEL_ID}"; export HF_HOME="\${MODELS_DIR}"
export HUGGING_FACE_HUB_TOKEN=\${HUGGING_FACE_HUB_TOKEN}
python -m vllm.entrypoints.openai.api_server --model "\${MODEL_ID}" --host "\${HOST}" --port "\${PORT}" \${EXTRA_ARGS} \${CLI_ARGS}
EOF
    chmod +x "$run_script_path"; chown "$VLLM_USER:$VLLM_USER" "$run_script_path"
    info "Created 'run_server.sh' at '$run_script_path'."
}

create_help_command() {
    info "Step 6: Creating the system-wide 'vllm_help' command..."
    local help_script_path="/usr/local/bin/vllm_help"
    
    # Using 'EOF' (with quotes) to prevent variable expansion inside the here-document
    # Then using sed to replace placeholders with actual values from the installer.
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
echo -e "  Run this command to check the API endpoint. It should return a JSON list of models."
echo -e "  ${G}curl http://localhost:__SERVER_PORT__/v1/models${NC}"

echo -e "\n${Y}Manual Operation (for testing and development):${NC}"
echo -e "  1. Switch to the dedicated user:"
echo -e "     ${G}sudo su - __VLLM_USER__${NC}"
echo -e "  2. (Optional) Set your Hugging Face token for private models:"
echo -e "     ${G}export HUGGING_FACE_HUB_TOKEN='your_token'${NC}"
echo -e "  3. Run the server with a model:"
echo -e "     ${G}./run_server.sh meta-llama/Llama-2-7b-chat-hf${NC}"
echo -e "     ${G}./run_server.sh /path/to/your/local/model${NC}"

echo -e "\n${Y}Configuration & File Locations:${NC}"
echo -e "  - Service User:      ${C}__VLLM_USER__${NC}"
echo -e "  - Installation Dir:  ${C}__VLLM_HOME_DIR__${NC}"
echo -e "  - Python venv:       ${C}__VENV_DIR__${NC}"
echo -e "  - HF Models Cache:   ${C}__MODELS_DIR__${NC}"
echo -e "  - Run Script:        ${C}__VLLM_HOME_DIR__/run_server.sh${NC}"
echo -e "  - Service File:      ${C}/etc/systemd/system/vllm.service${NC}"

echo -e "\n${Y}How to Change the Model in the Service:${NC}"
echo -e "  1. Edit the service file: ${G}sudo nano /etc/systemd/system/vllm.service${NC}"
echo -e "  2. Find the 'ExecStart=' line and change the model path or ID."
echo -e "  3. Reload systemd and restart the service:"
echo -e "     ${G}sudo systemctl daemon-reload && sudo systemctl restart vllm${NC}"
EOF

    # Replace placeholders with actual values
    sed -i "s|__VLLM_USER__|${VLLM_USER}|g" "$help_script_path"
    sed -i "s|__SERVER_PORT__|${SERVER_PORT}|g" "$help_script_path"
    sed -i "s|__VLLM_HOME_DIR__|${VLLM_HOME_DIR}|g" "$help_script_path"
    sed -i "s|__VENV_DIR__|${VENV_DIR}|g" "$help_script_path"
    sed -i "s|__MODELS_DIR__|${MODELS_DIR}|g" "$help_script_path"

    chmod +x "$help_script_path"
    info "'vllm_help' command created. You can now run it from anywhere to see this guide."
}

setup_systemd_service() {
    info "Step 7: Optional - Setup systemd service."
    # ... Function content is unchanged, omitted for brevity ...
    read -p "Do you want to create a systemd service to run the vLLM server on boot? (y/N) " -r CREATE_SERVICE_REPLY
    if [[ ! "$CREATE_SERVICE_REPLY" =~ ^[Yy] ]]; then warn "Skipping systemd service creation."; return; fi
    local model_location=""; local extra_service_args=""
    while true; do
        read -p "Use a Hugging Face model ID or a local path for the service? (hf/local): " -r model_type
        case "$model_type" in
            [Hh][Ff]) read -p "Enter the Hugging Face model identifier: " -r model_location; if [ -n "$model_location" ]; then break; fi; warn "Model ID cannot be empty.";;
            [Ll][Oo][Cc][Aa][Ll]) read -p "Enter the absolute path to your local model directory: " -r model_location; if [ -z "$model_location" ]; then warn "Path cannot be empty."; continue; fi; if [ ! -d "$model_location" ]; then warn "Directory not found."; continue; fi; info "Ensuring '${VLLM_USER}' user has access..."; chmod -R a+rX "${model_location}"; info "Permissions updated."; extra_service_args="--disable-log-stats"; info "Using '--disable-log-stats' for local service."; break;;
            *) warn "Invalid input. Please enter 'hf' or 'local'.";;
        esac
    done
    local service_file="/etc/systemd/system/vllm.service"; info "Creating systemd service file at '${service_file}'..."
    cat <<EOF > "$service_file"
[Unit]
Description=vLLM OpenAI-Compatible Server
After=network.target
[Service]
User=${VLLM_USER}; Group=${VLLM_USER}; WorkingDirectory=${VLLM_HOME_DIR}
ExecStart=${VLLM_HOME_DIR}/run_server.sh "${model_location}" ${extra_service_args}
Restart=always; RestartSec=10
[Install]
WantedBy=multi-user.target
EOF
    systemctl daemon-reload; systemctl enable vllm.service; info "Systemd service 'vllm.service' created and enabled."; warn "To start it, run: sudo systemctl start vllm"
}

# --- Main Execution Flow ---
main() {
    check_prerequisites
    setup_environment
    find_python_and_install_deps
    install_vllm
    create_run_script
    create_help_command
    setup_systemd_service

    echo
    info "========================================================"
    info "          vLLM Installation Complete!                   "
    info "========================================================"
    echo
    info "A system-wide command ${CYAN}vllm_help${NC} has been created."
    info "Run it from anywhere to get a cheat sheet on how to manage your server."
    echo
    info "Example:"
    echo -e "  ${COLOR_YELLOW}sudo systemctl start vllm${NC}"
    echo -e "  ${COLOR_YELLOW}vllm_help${NC}"
    echo
}

main "$@"
