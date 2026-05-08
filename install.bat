@echo off
setlocal enabledelayedexpansion

echo ################################################
echo # Anima LoRA Trainer - Installer              #
echo ################################################

:: Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.10 or later and add it to your PATH.
    pause
    exit /b 1
)

:: Create virtualenv
if not exist venv (
    echo [INFO] Creating virtual environment...
    python -m venv venv
)

:: Activate virtualenv
call venv\Scripts\activate.bat

:: Upgrade pip
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip

:: Clone sd-scripts if not exists
if not exist sd-scripts (
    echo [INFO] Cloning sd-scripts repository...
    git clone https://github.com/kohya-ss/sd-scripts.git
)

:: Install sd-scripts dependencies
echo [INFO] Installing sd-scripts dependencies...
cd sd-scripts
python -m pip install -r requirements.txt
cd ..

:: Install app-specific dependencies
echo [INFO] Installing app-specific dependencies...
python -m pip install -r requirements.txt

:: GPU Detection and PyTorch Installation
echo [INFO] Detecting GPU...
for /f "tokens=*" %%i in ('nvidia-smi --query-gpu=name --format=csv,noheader 2^>nul') do set GPU_NAME=%%i
echo [INFO] Detected GPU: %GPU_NAME%

:: CLEAN UNINSTALL AND FORCE CUDA PYTORCH
echo [INFO] Uninstalling existing torch packages to ensure clean CUDA install...
python -m pip uninstall -y torch torchvision torchaudio xformers bitsandbytes

set "INSTALL_CMD="

:: Check for RTX 50-series (Blackwell)
echo %GPU_NAME% | findstr /i "RTX.50 Blackwell" >nul
if %errorlevel% equ 0 (
    echo [INFO] RTX 50-series (Blackwell) detected. Installing PyTorch Nightly with CUDA 12.8...
    set INSTALL_CMD=python -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
    set "IS_NEW_GPU=1"
) else (
    :: Check for RTX 40-series (Ada Lovelace)
    echo %GPU_NAME% | findstr /i "RTX.40 Ada" >nul
    if %errorlevel% equ 0 (
        echo [INFO] RTX 40-series (Ada Lovelace) detected. Installing PyTorch with CUDA 12.4...
        set INSTALL_CMD=python -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
        set "IS_NEW_GPU=1"
    ) else (
        echo [INFO] Standard GPU detected. Installing stable PyTorch with CUDA 12.1...
        set INSTALL_CMD=python -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
        set "IS_NEW_GPU=0"
    )
)

echo [INFO] Executing: %INSTALL_CMD%
%INSTALL_CMD%

echo [INFO] Installing updated bitsandbytes for modern GPU support...
python -m pip install bitsandbytes>=0.43.0

echo.
echo [INFO] Verifying installation...
python -c "import torch; print(f'Torch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

echo.
echo [SUCCESS] Installation complete!
echo If "CUDA available" is False above, your GPU drivers or CUDA toolkit may need updating.
echo Use run.bat to start the application.
echo.
pause
