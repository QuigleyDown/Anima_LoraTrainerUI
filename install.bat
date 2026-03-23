@echo off
setlocal enabledelayedexpansion

echo ################################################
echo # Anima Preview 2 LoRA Trainer - Installer      #
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

:: CLEAN UNINSTALL AND FORCE CUDA PYTORCH
echo [INFO] Uninstalling existing torch packages to ensure clean CUDA install...
python -m pip uninstall -y torch torchvision torchaudio xformers

echo [INFO] Installing CUDA-enabled PyTorch 2.5.1...
python -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

echo.
echo [INFO] Verifying installation...
python -c "import torch; print(f'Torch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

echo.
echo [SUCCESS] Installation complete!
echo If "CUDA available" is False above, your GPU drivers or CUDA toolkit may need updating.
echo Use run.bat to start the application.
echo.
pause
