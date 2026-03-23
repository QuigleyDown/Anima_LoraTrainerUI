import os
import asyncio
import subprocess
import signal
import json
import shutil
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from huggingface_hub import hf_hub_download
import toml

app = FastAPI(title="Anima Preview 2 LoRA Trainer")

# Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(DATA_DIR, "models")
DATASETS_DIR = os.path.join(DATA_DIR, "datasets")
OUTPUTS_DIR = os.path.join(DATA_DIR, "outputs")
CONFIGS_DIR = os.path.join(DATA_DIR, "configs")
SD_SCRIPTS_DIR = os.path.join(BASE_DIR, "sd-scripts")

# Model Info
ANIMA_REPO = "circlestone-labs/Anima"
REQUIRED_FILES = {
    "dit": "anima-preview2.safetensors",
    "qwen3": "qwen_3_06b_base.safetensors",
    "vae": "qwen_image_vae.safetensors"
}

# Global state for training
training_process = None
log_queue = asyncio.Queue()

class TrainingConfig(BaseModel):
    name: str
    rank: int = 32
    alpha: int = 16
    learning_rate: float = 1e-4
    batch_size: int = 1
    num_epochs: int = 10
    save_every_n_epochs: int = 1
    resolution: int = 1024
    mixed_precision: str = "bf16"
    optimizer: str = "AdamW8bit"
    timestep_sampling: str = "sigmoid"
    discrete_flow_shift: float = 1.0
    dataset_path: str

@app.on_event("startup")
async def startup_event():
    # Ensure directories exist
    for d in [MODELS_DIR, DATASETS_DIR, OUTPUTS_DIR, CONFIGS_DIR]:
        os.makedirs(d, exist_ok=True)

@app.get("/")
async def root():
    return FileResponse(os.path.join(BASE_DIR, "app/static/index.html"))

@app.get("/api/models/status")
async def get_models_status():
    status = {}
    for key, filename in REQUIRED_FILES.items():
        path = os.path.join(MODELS_DIR, filename)
        status[key] = {
            "exists": os.path.exists(path),
            "filename": filename
        }
    return status

@app.post("/api/models/download")
async def download_models(background_tasks: BackgroundTasks):
    background_tasks.add_task(do_download_models)
    return {"message": "Download started in background"}

async def do_download_models():
    for key, filename in REQUIRED_FILES.items():
        if not os.path.exists(os.path.join(MODELS_DIR, filename)):
            await log_to_ui(f"Downloading {filename}...")
            hf_hub_download(repo_id=ANIMA_REPO, filename=filename, local_dir=MODELS_DIR)
            await log_to_ui(f"Downloaded {filename}")
    await log_to_ui("All models downloaded successfully!")

@app.get("/api/datasets")
async def list_datasets():
    datasets = []
    for d in os.listdir(DATASETS_DIR):
        if os.path.isdir(os.path.join(DATASETS_DIR, d)):
            datasets.append(d)
    return datasets

@app.post("/api/datasets/upload")
async def upload_dataset(name: str = Form(...), files: List[UploadFile] = File(...)):
    dataset_path = os.path.join(DATASETS_DIR, name)
    os.makedirs(dataset_path, exist_ok=True)
    
    for file in files:
        file_path = os.path.join(dataset_path, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
    return {"message": f"Uploaded {len(files)} files to {name}"}

@app.get("/api/outputs")
async def list_outputs():
    outputs = []
    for root, dirs, files in os.walk(OUTPUTS_DIR):
        for file in files:
            if file.endswith(".safetensors"):
                rel_path = os.path.relpath(os.path.join(root, file), OUTPUTS_DIR)
                outputs.append({
                    "name": file,
                    "path": rel_path.replace("\\", "/"),
                    "size": os.path.getsize(os.path.join(root, file))
                })
    return outputs

@app.get("/api/outputs/download/{path:path}")
async def download_output(path: str):
    file_path = os.path.join(OUTPUTS_DIR, path)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, filename=os.path.basename(file_path))

async def log_to_ui(message: str):
    await log_queue.put(message)

@app.get("/api/train/logs")
async def train_logs():
    async def log_generator():
        while True:
            message = await log_queue.get()
            yield f"data: {json.dumps({'message': message})}\n\n"
    return StreamingResponse(log_generator(), media_type="text/event-stream")

@app.post("/api/train/start")
async def start_training(config: TrainingConfig):
    global training_process
    if training_process and training_process.poll() is None:
        raise HTTPException(status_code=400, detail="Training already in progress")
    
    # Generate dataset.toml
    dataset_toml_path = os.path.join(CONFIGS_DIR, f"{config.name}_dataset.toml")
    dataset_config = {
        "general": {
            "enable_bucket": True,
            "resolution": config.resolution,
            "shuffle_caption": True,
            "keep_tokens": 0
        },
        "datasets": [
            {
                "subsets": [
                    {
                        "image_dir": os.path.join(DATASETS_DIR, config.dataset_path),
                        "num_repeats": 1,
                        "class_tokens": None
                    }
                ]
            }
        ]
    }
    with open(dataset_toml_path, "w") as f:
        toml.dump(dataset_config, f)
    
    # Build command
    cmd = [
        "accelerate", "launch",
        "--num_processes", "1",
        "--num_machines", "1",
        "--mixed_precision", config.mixed_precision,
        "--dynamo_backend", "no",
        os.path.join(SD_SCRIPTS_DIR, "anima_train_network.py"),
        "--pretrained_model_name_or_path", os.path.join(MODELS_DIR, REQUIRED_FILES["dit"]),
        "--qwen3", os.path.join(MODELS_DIR, REQUIRED_FILES["qwen3"]),
        "--vae", os.path.join(MODELS_DIR, REQUIRED_FILES["vae"]),
        "--dataset_config", dataset_toml_path,
        "--output_dir", os.path.join(OUTPUTS_DIR, config.name),
        "--output_name", config.name,
        "--network_module", "networks.lora_anima",
        "--network_dim", str(config.rank),
        "--network_alpha", str(config.alpha),
        "--learning_rate", str(config.learning_rate),
        "--train_batch_size", str(config.batch_size),
        "--max_train_epochs", str(config.num_epochs),
        "--mixed_precision", config.mixed_precision,
        "--optimizer_type", config.optimizer,
        "--timestep_sampling", config.timestep_sampling,
        "--discrete_flow_shift", str(config.discrete_flow_shift),
        "--cache_latents",
        "--save_every_n_epochs", str(config.save_every_n_epochs),
        "--save_precision", "bf16"
    ]
    
    await log_to_ui(f"Starting training with command: {' '.join(cmd)}")
    
    # Run process
    # Set CWD to sd-scripts so it can find its library and networks
    # Set PYTHONPATH to include sd-scripts
    # Set PYTHONIOENCODING and PYTHONUTF8 to utf-8 to prevent UnicodeEncodeError on Windows
    # Set PYTHONUNBUFFERED to 1 to ensure real-time log output
    env = os.environ.copy()
    env["PYTHONPATH"] = SD_SCRIPTS_DIR + (os.pathsep + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else "")
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    
    # Optional: Force some libraries to be less shy about outputting
    env["ACCELERATE_LOG_LEVEL"] = "INFO"
    env["FORCE_COLOR"] = "1" # Try to keep some formatting if rich supports it
    
    training_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        bufsize=1,
        cwd=SD_SCRIPTS_DIR,
        env=env,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    )
    
    async def monitor_process():
        for line in iter(training_process.stdout.readline, ""):
            await log_to_ui(line.strip())
        training_process.stdout.close()
        return_code = training_process.wait()
        await log_to_ui(f"Training finished with return code {return_code}")

    asyncio.create_task(monitor_process())
    
    return {"message": "Training started"}

@app.post("/api/train/stop")
async def stop_training():
    global training_process
    if training_process and training_process.poll() is None:
        if os.name == "nt":
            training_process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            training_process.terminate()
        return {"message": "Training stopped"}
    return {"message": "No training in progress"}

# Serve static files last
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "app/static")), name="static")
