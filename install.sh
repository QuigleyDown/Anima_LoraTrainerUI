#!/bin/bash

echo "################################################"
echo "# Anima Preview 2 LoRA Trainer - Installer      #"
echo "################################################"

# Check for Python
if ! command -v python3 &> /dev/null
then
    echo "[ERROR] Python 3 not found. Please install Python 3.10 or later."
    exit 1
fi

# Create virtualenv
if [ ! -d "venv" ]; then
    echo "[INFO] Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtualenv
source venv/bin/activate

# Upgrade pip
echo "[INFO] Upgrading pip..."
python3 -m pip install --upgrade pip

# Clone sd-scripts if not exists
if [ ! -d "sd-scripts" ]; then
    echo "[INFO] Cloning sd-scripts repository..."
    git clone https://github.com/kohya-ss/sd-scripts.git
fi

# Install sd-scripts dependencies
echo "[INFO] Installing sd-scripts dependencies..."
cd sd-scripts
python3 -m pip install -r requirements.txt
cd ..

# Install app-specific dependencies
echo "[INFO] Installing app-specific dependencies..."
python3 -m pip install -r requirements.txt

# GPU Detection and PyTorch Installation
echo "[INFO] Detecting GPU to determine PyTorch version..."
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)
echo "[INFO] Detected GPU: $GPU_NAME"

if [[ "$GPU_NAME" == *"RTX 50"* ]] || [[ "$GPU_NAME" == *"Blackwell"* ]]; then
    echo "[INFO] RTX 50-series (Blackwell) detected. Installing PyTorch Nightly with CUDA 12.8..."
    python3 -m pip install --upgrade --force-reinstall --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
else
    echo "[INFO] Standard GPU detected. Installing stable PyTorch with CUDA 12.1..."
    python3 -m pip install --upgrade --force-reinstall torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
fi

echo ""
echo "[SUCCESS] Installation complete!"
echo "Use ./run.sh to start the application."
echo ""
chmod +x run.sh 2>/dev/null
