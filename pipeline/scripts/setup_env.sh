#!/usr/bin/env bash
# ==============================================================================
# Thai LPR Pipeline - Linux Environment Setup (Python 3.10)
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

echo "=============================================================================="
echo "  THAI LPR PRODUCTION PIPELINE - LINUX ENVIRONMENT SETUP"
echo "  Target: Python 3.10.x / PyTorch / ONNX Runtime"
echo "=============================================================================="

# Detect Python 3.10
PYTHON_BIN=""
if command -v python3.10 &>/dev/null; then
    PYTHON_BIN="python3.10"
elif command -v python3 &>/dev/null; then
    PYTHON_BIN="python3"
else
    echo "[ERROR] Python 3.10 is required but not installed."
    exit 1
fi

echo "[*] Using Python: $($PYTHON_BIN --version)"

# Create virtual environment
if [ ! -d ".venv" ]; then
    echo "[*] Creating virtual environment (.venv)..."
    $PYTHON_BIN -m venv .venv
fi

source .venv/bin/activate

echo "[*] Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel --quiet

echo "[*] Detecting hardware acceleration..."
if command -v nvidia-smi &>/dev/null; then
    echo "[+] NVIDIA GPU detected. Installing CUDA 12.4 PyTorch..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
else
    echo "[!] Installing CPU PyTorch build..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

echo "[*] Installing production dependencies from requirements.txt..."
pip install -r requirements.txt

echo "[*] Running diagnostic self-check..."
python scripts/check_environment.py

echo "=============================================================================="
echo "[SUCCESS] Linux Environment setup complete!"
echo "=============================================================================="
