# vLLM All-in-One Installer for Linux

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/ParisNeo/vllm_for_linux/blob/main/LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Ubuntu-orange.svg)](https://ubuntu.com/)
[![Python](https://img.shields.io/badge/Python-3.9--3.12-blue.svg)](https://www.python.org/)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](#contributing)
[![Powered by vLLM](https://img.shields.io/badge/Powered%20by-vLLM-success.svg)](https://github.com/vllm-project/vllm)

This repository provides an automated, all-in-one installation script to set up and configure the [vLLM](https://github.com/vllm-project/vllm) inference server on an Ubuntu-based system. The script is designed for developers and MLOps engineers who want a quick, repeatable, and robust setup for serving large language models with high performance.

The script automates the entire process, from checking prerequisites to deploying the server as an optional `systemd` service for production use.

## Key Features

-   **🚀 Automated Setup**: A single command (`./install.sh`) handles everything.
-   **🛡️ Best Practices**: Creates a dedicated, non-root user (`vllm`) for security and isolates dependencies in a Python virtual environment managed by `uv`.
-   **📂 Centralized Management**: All files, including the environment, models, and scripts, are organized under a single directory (`/opt/vllm-server`).
-   **⚙️ Production Ready**: Includes an option to generate and enable a `systemd` service for running the vLLM server automatically on boot.
-   **🤖 Easy to Use**: Generates a simple `run_server.sh` script to manually start the OpenAI-compatible API server.
-   **🔧 Customizable**: Key configuration variables (like directories, user, and port) can be easily modified at the top of the `install.sh` script.

## Prerequisites

Before running the installation script, please ensure your system meets the following requirements:

1.  **Operating System**: **Ubuntu Linux**.
2.  **Hardware**: An **NVIDIA GPU** with **Compute Capability 7.0 or higher** (e.g., V100, T4, RTX 20xx, A100, H100).
3.  **Software**: **NVIDIA drivers** and the **CUDA Toolkit** must be installed and functioning correctly. You can verify this by running `nvidia-smi`.
4.  **Permissions**: You must have `sudo` or `root` access to run the script.

## Installation

The installation is performed by a single script.

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/ParisNeo/vllm_for_linux.git
    cd vllm_for_linux
    ```

2.  **Make the script executable:**
    ```sh
    chmod +x install.sh
    ```

3.  **Run the installation script:**
    ```sh
    sudo ./install.sh
    ```
    The script will guide you through the process, check for prerequisites, and ask if you want to set up the `systemd` service for automatic startup.

## Post-Installation

After the installation is complete, you can run and manage the vLLM server.

### Running the Server Manually

This is the recommended way to test your setup or run the server for development purposes.

1.  **Switch to the dedicated `vllm` user:**
    ```sh
    sudo su - vllm
    ```

2.  **(Optional) Set your Hugging Face Hub Token**
    If you need to access private or gated models, export your token.
    ```sh
    export HUGGING_FACE_HUB_TOKEN='hf_YourTokenHere'
    ```

3.  **Run the server:**
    Use the generated `run_server.sh` script and pass the model identifier as an argument.
    ```sh
    # Usage: ./run_server.sh [MODEL_IDENTIFIER]
    ./run_server.sh meta-llama/Llama-2-7b-chat-hf
    ```
    The server will start, and the model will be downloaded to `/opt/vllm-server/models` on its first run.

### Managing the Systemd Service

If you chose to create the systemd service during installation, you can manage it using standard `systemctl` commands. This is ideal for production environments.

-   **Start the service:**
    ```sh
    sudo systemctl start vllm
    ```

-   **Check the status of the service:**
    ```sh
    sudo systemctl status vllm
    ```

-   **View live logs:**
    ```sh
    sudo journalctl -u vllm -f
    ```

-   **Stop the service:**
    ```sh
    sudo systemctl stop vllm
    ```

-   **Enable the service to start on boot (done by the script):**
    ```sh
    sudo systemctl enable vllm
    ```

-   **Disable the service from starting on boot:**
    ```sh
    sudo systemctl disable vllm
    ```

## Configuration

The `install.sh` script is designed to be easily configurable. You can modify the variables at the top of the file to change default paths, the dedicated user, server port, and more.

The generated `run_server.sh` script can also be edited to add or modify default vLLM command-line arguments, such as `--tensor-parallel-size` or `--gpu-memory-utilization`.

## Contributing

Help is welcome! We appreciate any contributions, from fixing a typo to adding new features. Please feel free to open an issue to report a bug or suggest an improvement.

If you'd like to contribute code, please follow these steps:
1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/my-new-feature`).
3.  Make your changes.
4.  Commit your changes (`git commit -am 'Add some feature'`).
5.  Push to the branch (`git push origin feature/my-new-feature`).
6.  Submit a new Pull Request.

## License

This project is licensed under the **Apache 2.0 License**. See the [LICENSE](LICENSE) file for details.
