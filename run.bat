@echo off
setlocal enabledelayedexpansion

if not exist venv (
    echo [ERROR] Virtual environment not found. Please run install.bat first.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

:: Pre-start health check
echo [INFO] Verifying PyTorch and CUDA environment...
python -c "import torch; print(f'Torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')" 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python environment is broken. Please run install.bat again.
    pause
    exit /b 1
)

echo [INFO] Starting Anima LoRA Trainer...
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
pause
