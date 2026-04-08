#!/bin/bash

if [ ! -d "venv" ]; then
    echo "[ERROR] Virtual environment not found. Please run ./install.sh first."
    exit 1
fi

source venv/bin/activate
echo "[INFO] Starting Anima LoRA Trainer..."
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
