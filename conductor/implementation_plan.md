# Implementation Plan - Anima Preview 2 LoRA Trainer

Create a Python web app to configure and run LoRA training for the Anima Preview 2 model using `sd-scripts`.

## User Review Required

> [!IMPORTANT]
> The application will require a GPU with at least 12GB+ VRAM (preferably 16GB+ for comfortable training) and a CUDA-capable environment. It will also require several GB of disk space for the base models and `sd-scripts` dependencies.

- **Model Source**: Official files will be downloaded from `circlestone-labs/Anima`.
- **Training Engine**: `sd-scripts` (will be cloned during installation).
- **Architecture**: FastAPI backend with a simple HTML/JS/Tailwind CSS frontend.

## Proposed Architecture

- **Backend**: FastAPI
    - Process management for `sd-scripts`.
    - Real-time log streaming using Server-Sent Events (SSE).
    - Model downloading via `huggingface_hub`.
    - Dataset management (upload and organization).
- **Frontend**: Single Page Application (Vanilla JS + Tailwind)
    - Configuration dashboard with Anima-specific defaults.
    - Real-time terminal output.
    - Model and Dataset management tabs.
- **File Structure**:
    - `app/`: FastAPI application code.
    - `sd-scripts/`: Cloned repository.
    - `data/`: Datasets, models, and training outputs.

## Phase 1: Environment & Setup
- [ ] Create `requirements.txt` with base dependencies.
- [ ] Create `install.bat` and `install.sh` to:
    - Set up virtualenv.
    - Install PyTorch with CUDA.
    - Clone `sd-scripts` and install its requirements.
    - Install app-specific requirements.
- [ ] Create `run.bat` and `run.sh` to start the FastAPI server.

## Phase 2: Backend Development (FastAPI)
- [ ] Initialize FastAPI app and static file serving.
- [ ] Implement Model Download API (using `huggingface_hub`).
- [ ] Implement Dataset Upload & Management API.
- [ ] Implement Training Configuration API (TOML generation).
- [ ] Implement Training Execution Engine:
    - Background process management using `subprocess.Popen`.
    - Async log reading and queueing for SSE.

## Phase 3: Frontend Development
- [ ] Build the main layout with Navigation (Models, Dataset, Config, Training).
- [ ] **Models Tab**: Check for existing files, download progress.
- [ ] **Dataset Tab**: File upload, folder listing.
- [ ] **Config Tab**: Form with specialized Anima Preview 2 parameters:
    - Rank (Dim), Alpha, Learning Rate, Resolution.
    - Timestep sampling, Discrete flow shift.
    - VAE chunk size, Mixed precision.
- [ ] **Training Tab**: Console log viewer, Start/Stop controls.

## Phase 4: Integration & Testing
- [ ] Verify `sd-scripts/anima_train_network.py` command generation.
- [ ] Test end-to-end flow: Download -> Upload -> Config -> Train.
- [ ] Add progress parsing (percentage, steps/s) if possible from `sd-scripts` output.

## Phase 5: Documentation & Polishing
- [ ] Add `README.md` with usage instructions.
- [ ] Ensure Windows and Linux scripts work correctly.

## Verification Plan

### Automated Tests
- Basic API endpoint tests (health check, config validation).

### Manual Verification
1. Run `install.bat` to ensure all dependencies and `sd-scripts` are correctly installed.
2. Run `run.bat` and verify the web UI opens.
3. Download Anima Preview 2 models via the UI.
4. Upload a small test dataset.
5. Start a training job with default settings and verify the process launches and logs appear in the UI.
6. Verify the output LoRA file is created in the `data/outputs` directory.
