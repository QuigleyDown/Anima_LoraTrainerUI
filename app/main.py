import os
import asyncio
import sys

# On Windows, the ProactorEventLoop is required for subprocess support
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

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
import threading

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
    "dit": {
        "filename": "anima-preview2.safetensors", 
        "url": f"https://huggingface.co/{ANIMA_REPO}/resolve/main/split_files/diffusion_models/anima-preview2.safetensors"
    },
    "qwen3": {
        "filename": "qwen_3_06b_base.safetensors", 
        "url": f"https://huggingface.co/{ANIMA_REPO}/resolve/main/split_files/text_encoders/qwen_3_06b_base.safetensors"
    },
    "vae": {
        "filename": "qwen_image_vae.safetensors", 
        "url": f"https://huggingface.co/{ANIMA_REPO}/resolve/main/split_files/vae/qwen_image_vae.safetensors"
    }
}

# Global state
training_process = None
log_queue = asyncio.Queue()
download_status = {} # model_key -> progress percentage

class TrainingConfig(BaseModel):
    name: str
    training_type: str = "LoRA" # "LoRA" or "Full"
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
    # Initialize download status
    for key in REQUIRED_FILES:
        download_status[key] = 0

@app.get("/")
async def root():
    return FileResponse(os.path.join(BASE_DIR, "app/static/index.html"))

@app.get("/api/models/status")
async def get_models_status():
    status = {}
    for key, info in REQUIRED_FILES.items():
        filename = info["filename"]
        path = os.path.join(MODELS_DIR, filename)
        status[key] = {
            "exists": os.path.exists(path),
            "filename": filename,
            "default_source": info["url"],
            "progress": download_status.get(key, 0)
        }
    return status

@app.get("/api/models/progress")
async def get_models_progress():
    return download_status

class DownloadRequest(BaseModel):
    model_key: str
    source: Optional[str] = None

@app.post("/api/models/download")
async def download_model(req: DownloadRequest, background_tasks: BackgroundTasks):
    if req.model_key not in REQUIRED_FILES:
        raise HTTPException(status_code=400, detail="Invalid model key")
    
    download_status[req.model_key] = 0
    source = req.source if req.source and req.source.strip() else None
    
    # Run in a separate thread so it doesn't block the event loop
    loop = asyncio.get_running_loop()
    background_tasks.add_task(do_download_model_sync, req.model_key, source, loop)
    
    return {"message": f"Download of {req.model_key} started"}

def do_download_model_sync(key: str, source: Optional[str], loop):
    """Synchronous download function run in a thread."""
    try:
        info = REQUIRED_FILES[key]
        filename = info["filename"]
        url = source if source else info["url"]
        dest_path = os.path.join(MODELS_DIR, filename)

        # Notify UI via thread-safe call
        asyncio.run_coroutine_threadsafe(log_to_ui(f"Downloading {key} from: {url}"), loop)
        
        import requests
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, stream=True, headers=headers, allow_redirects=True)
        
        if response.status_code != 200:
            raise Exception(f"Server returned status code {response.status_code}")

        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        os.makedirs(MODELS_DIR, exist_ok=True)
        
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        download_status[key] = int((downloaded / total_size) * 100)
                    # Force disk write for metadata update
                    f.flush()
        
        download_status[key] = 100
        asyncio.run_coroutine_threadsafe(log_to_ui(f"Successfully downloaded {filename}"), loop)
    except Exception as e:
        asyncio.run_coroutine_threadsafe(log_to_ui(f"Error downloading {key}: {str(e)}"), loop)
        download_status[key] = 0

@app.get("/api/datasets")
async def list_datasets():
    datasets = []
    if not os.path.exists(DATASETS_DIR):
        return datasets
        
    for d in os.listdir(DATASETS_DIR):
        d_path = os.path.join(DATASETS_DIR, d)
        if os.path.isdir(d_path):
            file_count = 0
            total_size = 0
            for root, dirs, files in os.walk(d_path):
                file_count += len(files)
                for f in files:
                    total_size += os.path.getsize(os.path.join(root, f))
            
            datasets.append({
                "name": d,
                "file_count": file_count,
                "size": total_size
            })
    return datasets

@app.get("/api/datasets/{name}/files")
async def list_dataset_files(name: str):
    dataset_path = os.path.join(DATASETS_DIR, name)
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=404, detail="Dataset not found")
        
    files_map = {}
    for f in os.listdir(dataset_path):
        if not os.path.isfile(os.path.join(dataset_path, f)):
            continue
            
        base, ext = os.path.splitext(f)
        ext = ext.lower()
        
        if base not in files_map:
            files_map[base] = {"name": base, "image": None, "caption": None, "caption_text": ""}
            
        if ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
            files_map[base]["image"] = f
        elif ext == ".txt":
            files_map[base]["caption"] = f
            try:
                with open(os.path.join(dataset_path, f), "r", encoding="utf-8") as tf:
                    files_map[base]["caption_text"] = tf.read().strip()
            except:
                pass
                
    return sorted(list(files_map.values()), key=lambda x: x["name"])

@app.get("/api/datasets/{name}/view/{filename}")
async def serve_dataset_file(name: str, filename: str):
    file_path = os.path.join(DATASETS_DIR, name, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)

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
    
    if config.training_type == "Full":
        script_name = "anima_train.py"
    else:
        script_name = "anima_train_network.py"

    cmd = [
        "accelerate", "launch",
        "--num_processes", "1",
        "--num_machines", "1",
        "--mixed_precision", config.mixed_precision,
        "--dynamo_backend", "no",
        os.path.join(SD_SCRIPTS_DIR, script_name),
        "--pretrained_model_name_or_path", os.path.join(MODELS_DIR, REQUIRED_FILES["dit"]["filename"]),
        "--qwen3", os.path.join(MODELS_DIR, REQUIRED_FILES["qwen3"]["filename"]),
        "--vae", os.path.join(MODELS_DIR, REQUIRED_FILES["vae"]["filename"]),
        "--dataset_config", dataset_toml_path,
        "--output_dir", os.path.join(OUTPUTS_DIR, config.name),
        "--output_name", config.name,
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

    if config.training_type == "LoRA":
        cmd.extend([
            "--network_module", "networks.lora_anima",
            "--network_dim", str(config.rank),
            "--network_alpha", str(config.alpha),
        ])
    
    await log_to_ui(f"Starting {config.training_type} training with command: {' '.join(cmd)}")
    
    env = os.environ.copy()
    env["PYTHONPATH"] = SD_SCRIPTS_DIR + (os.pathsep + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else "")
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["ACCELERATE_LOG_LEVEL"] = "INFO"
    env["FORCE_COLOR"] = "1"
    
    training_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        cwd=SD_SCRIPTS_DIR,
        env=env,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    )
    
    def reader(loop):
        try:
            while True:
                line = training_process.stdout.readline()
                if not line:
                    break
                clean_line = line.strip()
                if clean_line:
                    asyncio.run_coroutine_threadsafe(log_to_ui(clean_line), loop)
        except Exception as e:
            asyncio.run_coroutine_threadsafe(log_to_ui(f"Error reading logs: {str(e)}"), loop)
        finally:
            training_process.stdout.close()
            return_code = training_process.wait()
            asyncio.run_coroutine_threadsafe(log_to_ui(f"Training finished with return code {return_code}"), loop)

    loop = asyncio.get_running_loop()
    threading.Thread(target=reader, args=(loop,), daemon=True).start()
    
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

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "app/static")), name="static")
