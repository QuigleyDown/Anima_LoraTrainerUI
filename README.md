# Anima_LoraTrainerUI

A simple, automated web application for training LoRA models for the **Anima Preview 2** image diffusion model. This app streamlines the setup, model downloading, dataset management, and training process using the `sd-scripts` project.

## Features

- **One-Click Installation**: Automated scripts for Windows and Linux.
- **Model Management**: Easy download of required base models (DiT, Qwen3, VAE) from Hugging Face.
- **Dataset Management**: Upload images and captions directly through the web UI.
- **Tailored Configuration**: Optimized defaults for Anima Preview 2 training (Rank, Alpha, Learning Rate, Flow Shift, etc.).
- **Real-time Monitoring**: Live terminal output of the training process.
- **Cross-Platform**: Works on Windows and Linux.

## Requirements

- **GPU**: NVIDIA GPU with 12GB+ VRAM (16GB+ recommended).
- **Python**: Python 3.10 or later.
- **CUDA**: CUDA-capable environment.
- **Git**: Installed and in your PATH.

## Installation

### Windows
1. Double-click `install.bat`.
2. Wait for the installation to complete (this will download several GB of dependencies).

### Linux
1. Run `./install.sh` in your terminal.
2. Ensure you have the necessary build tools and Python development headers installed.

## How to Use

1. **Start the App**:
   - Windows: Run `run.bat`.
   - Linux: Run `./run.sh`.
2. **Access the UI**: Open your browser and go to `http://localhost:8000`.
3. **Download Models**: Go to the **Models** tab and click **Download Missing Models**. This will fetch the official Anima Preview 2 weights.
4. **Prepare Dataset**: 
   - Go to the **Datasets** tab.
   - Enter a name for your dataset.
   - Upload your images (and optionally `.txt` caption files with the same name as the images).
5. **Configure & Train**:
   - Go to the **Training** tab.
   - Select your uploaded dataset.
   - Adjust settings if needed (defaults are optimized for Anima).
   - Click **Start Training**.
6. **Get Results**: Your trained LoRA files will be saved in `data/outputs/<project_name>`.

## Important Note

Anima Preview 2 is an experimental model. LoRAs trained for Preview 2 may not be compatible with the final full release of Anima. It is recommended to use `bf16` mixed precision for stability.

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more details.

**Important Note on the Anima Model:**
While this UI code is MIT-licensed, the **Anima Preview 2** model weights and their derivatives (including LoRAs and full finetunes) are subject to the **CircleStone Labs Non-Commercial License** and the **NVIDIA Open Model License Agreement**. You are responsible for ensuring your use of the model complies with their respective terms.

## Acknowledgements

- [CircleStone Labs](https://huggingface.co/circlestone-labs) for the Anima model.
- [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) for the training engine.
- [CircleStone Labs](https://huggingface.co/circlestone-labs) for the Anima model.
- [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) for the training engine.
