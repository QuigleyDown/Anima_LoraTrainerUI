# Anima_LoraTrainerUI Documentation

This document provides detailed instructions on installing, configuring, and using the Anima_LoraTrainerUI.

---

## 1. System Requirements

### Hardware
- **GPU**: NVIDIA GPU is required. 
    - **Minimum**: 12GB VRAM (with `bf16` and `cache_latents` enabled).
    - **Recommended**: 16GB - 24GB VRAM for faster training and higher batch sizes.
- **Storage**: At least 20GB of free space (Base models ~10GB + Dependencies + Datasets/Outputs).
- **RAM**: 16GB+ System RAM.

### Software
- **Python**: 3.10 or newer.
- **Git**: Required for cloning `sd-scripts`.
- **Drivers**: Latest NVIDIA Drivers with CUDA support.

---

## 2. Installation

### Windows Installation
1. **Clone/Download** this repository to your desired folder.
2. **Double-click `install.bat`**.
   - This script will create a `venv` (virtual environment).
   - It will install PyTorch 2.5 with CUDA 12.1 support.
   - It will clone the `sd-scripts` repository automatically.
   - It will install all necessary dependencies for both the trainer and the web UI.
3. **Wait** for the process to finish. It may take several minutes depending on your internet speed.

### Linux Installation
1. **Open a terminal** in the project directory.
2. **Make the script executable**:
   ```bash
   chmod +x install.sh
   ```
3. **Run the installer**:
   ```bash
   ./install.sh
   ```
   *Note: If you encounter "Permission denied" even after chmod, you can try running it directly with bash:*
   ```bash
   bash install.sh
   ```
4. The script follows the same logic as the Windows version, setting up a virtual environment and downloading all requirements.

---

## 3. Using the Application

### Starting the App
- **Windows**: Run `run.bat`.
- **Linux**: Run `./run.sh`.
- Once the console says `Uvicorn running on http://0.0.0.0:8000`, open your web browser and navigate to `http://localhost:8000`.

### Workflow Steps
1. **Models Tab**: Click "Download Missing Models". This is mandatory for the first run. It downloads the DiT, Qwen3 Text Encoder, and the specialized VAE from Hugging Face.
2. **Datasets Tab**: 
   - Enter a unique name for your dataset.
   - Select your images and matching `.txt` caption files.
   - Click "Upload". Your files are stored in `data/datasets/<name>`.
3. **Training Tab**:
   - Fill in your Project Name (this will be the filename of your LoRA).
   - Select your dataset from the dropdown.
   - Configure parameters (see below).
   - Click **Start Training**.
   - Monitor the **Terminal Output** for progress and potential errors.

---

## 4. Configuration Settings Explained

Anima uses a **Diffusion Transformer (DiT)** architecture with **Rectified Flow**, which differs from standard Stable Diffusion v1.5/XL.

### Core Settings
- **Project Name**: The name used for the output folder and the `.safetensors` LoRA/Model file.
- **Training Type**: 
    - **LoRA**: (Recommended) Low-Rank Adaptation. Trains a small additional file that sits on top of the base model. Fast and low VRAM usage.
    - **Full Finetune**: Updates all weights of the DiT model. 
        - **VRAM Warning**: Requires at least 24GB VRAM. 
        - **Optimization**: Gradient Checkpointing is automatically enabled to save memory. 
        - **Recommended Optimizer**: Use `AdamW8bit` or `PagedAdamW8bit`. Avoid `Prodigy` for full finetunes unless you have very high VRAM (32GB+).
- **Rank (Dim)**: (LoRA only) The "capacity" of the LoRA. 
    - `16-32`: Good for characters or specific styles.
    - `64-128`: Better for complex concepts or high-detail subjects. Higher ranks use more VRAM.
- **Alpha**: (LoRA only) The scaling factor for the Rank. Usually set to `half of Rank` or `equal to Rank`. A lower Alpha relative to Rank can make training more stable.
    - *Note: For `Prodigy` or `DAdaptation`, Alpha is typically set to `1`.*
- **Optimizer**: The algorithm used to update model weights.
    - **AdamW8bit**: (Default) Fast and memory-efficient. Great for most users.
    - **AdamW / PagedAdamW8bit**: Standard AdamW or a "paged" version that can offload to system RAM if VRAM is full (slower but prevents crashes).
    - **Lion / Lion8bit**: Often requires a lower learning rate (e.g., `5e-5`) but can converge faster or with better detail in some cases.
    - **Prodigy / DAdaptation**: "Adaptive" optimizers that attempt to find the optimal learning rate automatically. 
        - **Important**: When using these, set **Learning Rate** to `1.0` and **Alpha** to `1`. The UI will attempt to set these defaults for you.
    - **Adafactor**: A memory-efficient optimizer often used for large model fine-tuning.
- **Timestep Sampling**: Controls how noise levels (timesteps) are distributed during training.
    - **sigmoid**: (Default) Recommended for Anima's Rectified Flow. Focuses on the "middle" of the diffusion process.
    - **uniform**: Even distribution across all noise levels.
    - **sigma**: Uses the noise schedule's sigma values directly.
    - **shift / flux_shift**: Applies a distribution shift. `flux_shift` is tuned specifically for flow-based models like FLUX or Anima.
- **Learning Rate**: How fast the model learns. 
    - `1e-4` (0.0001) is the standard starting point for AdamW.
    - Use `1.0` for **Prodigy** or **DAdaptation**.
    - If the model "deep fries" (distorts) images quickly, lower this to `5e-5` or `2e-5`.
- **Epochs**: The total number of passes through the entire dataset.
- **Save Every N Epochs**: Configures how often the model state is saved. For example, if set to `2`, a new `.safetensors` file will be created every 2 epochs, allowing you to test intermediate versions of your LoRA.
- **Resolution**: Anima is optimized for `1024`. You can use `896` or `768` to save VRAM, but `1024` yields the best quality.
- **Aspect Ratio Bucketing**: The app automatically enables bucketing by default. This allows you to train on images of various aspect ratios (landscape, portrait) without them being forcefully cropped to squares, preserving more of your dataset's composition.

### Outputs Tab
- This tab lists all `.safetensors` files found in the `data/outputs` directory.
- You can refresh the list to see new files as they are saved during training.
- Use the **Download** link next to any file to save it to your computer.

### Anima-Specific Settings
- **Timestep Sampling (Sigmoid)**: This controls how the model samples noise during training. `sigmoid` is the default and recommended setting for Anima's Rectified Flow implementation.
- **Discrete Flow Shift**: Controls the distribution of timesteps.
    - `1.0`: Default.
    - `3.0`: Some users find this helps the model focus more on the "structural" parts of the image, potentially improving composition.
- **Mixed Precision (bf16)**: **Highly Recommended.** `bf16` (BFloat16) provides the stability of FP32 with the speed and memory savings of FP16. Requires an RTX 30-series GPU or newer. Use `fp16` for older GPUs.

---

## 5. Troubleshooting

- **Out of Memory (OOM)**: 
    - **Enable Gradient Checkpointing**: This is now enabled by default to save VRAM.
    - **Use 8-bit Optimizers**: Switch to `AdamW8bit` or `PagedAdamW8bit`.
    - **Reduce Batch Size**: Set Batch Size to `1`.
    - **Lower Resolution**: Reduce to `896` or `768`.
    - **LoRA vs Full**: If Full Finetune keeps failing even with the above, use LoRA instead. LoRA is much more memory-efficient.
- **No GPU Found**: 
    - Ensure you have NVIDIA drivers installed. 
    - If the installer fails to install Torch with CUDA, you may need to install it manually within the `venv`.
- **Training is slow**:
    - This is a 2-billion parameter model; it is significantly heavier than SD1.5. This is normal behavior.
