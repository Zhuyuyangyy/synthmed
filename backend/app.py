"""
SynthMed - AI医学影像合成平台 FastAPI后端
"""
import sys
import yaml
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
import time
import uuid
import base64
import io
from PIL import Image
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))

from core.diffusion.ldm import LatentDiffusionModel, MedicalImageSynthesizer, GenerationConfig
from core.controlnet.anatomical import AnatomicalControlNet, AnatomicalConstraint
from core.data.processor import (
    MedicalImagePreprocessor, VolumeRenderer, load_medical_image,
    DICOMProcessor, augment_medical_image
)

# Load config
config_path = Path(__file__).parent / "config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

app = FastAPI(
    title="SynthMed",
    description="AI医学影像合成数据平台 API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
synthesizer: Optional[MedicalImageSynthesizer] = None
ldm_model: Optional[LatentDiffusionModel] = None
controlnet: Optional[AnatomicalControlNet] = None

# Task queue
task_queue: Dict[str, Dict] = {}
task_results: Dict[str, Dict] = {}

# GPU monitoring
gpu_stats = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "memory_allocated": 0,
    "memory_reserved": 0,
    "utilization": 0
}


# === Pydantic Models ===

class GenerationRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = "low quality, blurry, artifacts, distorted anatomy"
    width: int = 512
    height: int = 512
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    seed: int = 42
    num_images: int = 1
    modality: str = "ct"
    body_part: str = "chest"
    pathology: Optional[str] = None
    severity: float = 0.5


class BatchGenerationRequest(BaseModel):
    requests: List[GenerationRequest]
    callback_url: Optional[str] = None


class MaskGenerationRequest(BaseModel):
    mask_base64: str
    target_prompt: str
    num_images: int = 1
    strength: float = 0.8
    seed: int = 42


class TaskResponse(BaseModel):
    task_id: str
    status: str
    progress: float
    created_at: str


class GenerationResponse(BaseModel):
    task_id: str
    status: str
    images: Optional[List[str]] = None
    metadata: Optional[Dict] = None
    error: Optional[str] = None


class VolumeRenderRequest(BaseModel):
    volume_base64: str
    axis: str = "axial"
    index: int = 0
    window_center: Optional[float] = None
    window_width: Optional[float] = None


class GPUMonitorResponse(BaseModel):
    device: str
    memory_allocated_mb: float
    memory_reserved_mb: float
    utilization_percent: float
    temperature_celsius: Optional[float] = None


# === Helper Functions ===

def init_models():
    """Initialize AI models"""
    global synthesizer, ldm_model, controlnet
    
    logger.info("Initializing AI models...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        ldm_model = LatentDiffusionModel(
            model_path="stabilityai/stable-diffusion-2-1",
            device=device
        )
        synthesizer = MedicalImageSynthesizer(ldm_model)
        
        controlnet = AnatomicalControlNet(
            base_model_path="stabilityai/stable-diffusion-2-1",
            device=device
        )
        
        logger.info(f"Models initialized on {device}")
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")
        logger.info("Running in demo mode (no GPU model loaded)")


def update_gpu_stats():
    """Update GPU statistics"""
    global gpu_stats
    
    if torch.cuda.is_available():
        gpu_stats["memory_allocated"] = torch.cuda.memory_allocated() / 1024**2
        gpu_stats["memory_reserved"] = torch.cuda.memory_reserved() / 1024**2
        
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            gpu_stats["utilization"] = util.gpu
            gpu_stats["temperature_celsius"] = temp
            pynvml.nvmlShutdown()
        except:
            pass


def image_to_base64(image: torch.Tensor) -> str:
    """Convert tensor image to base64 string"""
    # image: (C, H, W) or (H, W, C)
    if image.dim() == 3 and image.shape[0] == 3:
        image = image.permute(1, 2, 0)
    
    img_np = (image.cpu().numpy() * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_np)
    
    buffer = io.BytesIO()
    img_pil.save(buffer, format="PNG")
    
    return base64.b64encode(buffer.getvalue()).decode()


def base64_to_image(b64: str) -> torch.Tensor:
    """Convert base64 string to tensor"""
    data = base64.b64decode(b64)
    img = Image.open(io.BytesIO(data))
    img = img.convert("RGB")
    
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    
    return tensor


# === Startup ===

@app.on_event("startup")
async def startup():
    logger.info("Starting SynthMed API...")
    init_models()
    logger.info("SynthMed API ready!")


# === Health & Status ===

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "models_loaded": ldm_model is not None,
        "device": gpu_stats["device"]
    }


@app.get("/gpu/status", response_model=GPUMonitorResponse)
async def gpu_status():
    """Get GPU utilization status"""
    update_gpu_stats()
    return GPUMonitorResponse(
        device=gpu_stats["device"],
        memory_allocated_mb=gpu_stats["memory_allocated"],
        memory_reserved_mb=gpu_stats["memory_reserved"],
        utilization_percent=gpu_stats["utilization"],
        temperature_celsius=gpu_stats.get("temperature_celsius")
    )


# === Generation Endpoints ===

@app.post("/generate", response_model=GenerationResponse)
async def generate_single(req: GenerationRequest):
    """Generate a single medical image"""
    task_id = str(uuid.uuid4())
    
    # Create task
    task_queue[task_id] = {
        "status": "running",
        "progress": 0.0,
        "created_at": datetime.now().isoformat(),
        "config": req.dict()
    }
    
    try:
        if synthesizer is not None and ldm_model is not None:
            # Use real model
            config = GenerationConfig(
                prompt=req.prompt,
                negative_prompt=req.negative_prompt,
                width=req.width,
                height=req.height,
                num_inference_steps=req.num_inference_steps,
                guidance_scale=req.guidance_scale,
                seed=req.seed,
                num_images=req.num_images,
                modality=req.modality,
                body_part=req.body_part,
                pathology=req.pathology,
                severity=req.severity
            )
            
            results = ldm_model.generate(config)
            
            images_b64 = [image_to_base64(r.image) for r in results]
            
            task_queue[task_id]["status"] = "completed"
            task_queue[task_id]["progress"] = 1.0
            
            return GenerationResponse(
                task_id=task_id,
                status="completed",
                images=images_b64,
                metadata=results[0].metadata if results else None
            )
        else:
            # Demo mode - return placeholder
            task_queue[task_id]["status"] = "completed"
            
            # Create a demo gradient image
            demo_img = Image.new("RGB", (req.width, req.height))
            for y in range(req.height):
                for x in range(req.width):
                    r = int((x / req.width) * 255)
                    g = int((y / req.height) * 255)
                    demo_img.putpixel((x, y), (r, g, 100))
            
            buffer = io.BytesIO()
            demo_img.save(buffer, format="PNG")
            demo_b64 = base64.b64encode(buffer.getvalue()).decode()
            
            return GenerationResponse(
                task_id=task_id,
                status="completed",
                images=[demo_b64],
                metadata={
                    "modality": req.modality,
                    "body_part": req.body_part,
                    "seed": req.seed,
                    "demo": True
                }
            )
    
    except Exception as e:
        task_queue[task_id]["status"] = "failed"
        task_queue[task_id]["error"] = str(e)
        
        return GenerationResponse(
            task_id=task_id,
            status="failed",
            error=str(e)
        )


@app.post("/generate/batch")
async def generate_batch(req: BatchGenerationRequest):
    """Generate multiple medical images in batch"""
    task_id = str(uuid.uuid4())
    
    task_queue[task_id] = {
        "status": "running",
        "progress": 0.0,
        "total": len(req.requests),
        "completed": 0,
        "created_at": datetime.now().isoformat()
    }
    
    results = []
    
    try:
        for i, gen_req in enumerate(req.requests):
            config = GenerationConfig(
                prompt=gen_req.prompt,
                negative_prompt=gen_req.negative_prompt,
                width=gen_req.width,
                height=gen_req.height,
                num_inference_steps=gen_req.num_inference_steps,
                guidance_scale=gen_req.guidance_scale,
                seed=gen_req.seed,
                num_images=gen_req.num_images,
                modality=gen_req.modality,
                body_part=gen_req.body_part,
                pathology=gen_req.pathology,
                severity=gen_req.severity
            )
            
            if synthesizer is not None and ldm_model is not None:
                gen_results = ldm_model.generate(config)
                images_b64 = [image_to_base64(r.image) for r in gen_results]
            else:
                # Demo
                demo_img = Image.new("RGB", (gen_req.width, gen_req.height), (50 + i * 10, 100, 150))
                buffer = io.BytesIO()
                demo_img.save(buffer, format="PNG")
                images_b64 = [base64.b64encode(buffer.getvalue()).decode()]
            
            results.extend(images_b64)
            
            task_queue[task_id]["completed"] = i + 1
            task_queue[task_id]["progress"] = (i + 1) / len(req.requests)
        
        task_queue[task_id]["status"] = "completed"
        
        return GenerationResponse(
            task_id=task_id,
            status="completed",
            images=results
        )
    
    except Exception as e:
        task_queue[task_id]["status"] = "failed"
        return GenerationResponse(task_id=task_id, status="failed", error=str(e))


@app.post("/generate/from-mask")
async def generate_from_mask(req: MaskGenerationRequest):
    """Generate image from segmentation mask (ControlNet)"""
    task_id = str(uuid.uuid4())
    
    try:
        # Decode mask
        mask_tensor = base64_to_image(req.mask_base64)
        
        if synthesizer is not None and ldm_model is not None:
            results = synthesizer.generate_with_mask(
                mask=mask_tensor,
                target_prompt=req.target_prompt,
                num_images=req.num_images,
                strength=req.strength
            )
            
            images_b64 = [image_to_base64(r.image) for r in results]
        else:
            # Demo
            demo_img = Image.new("RGB", (512, 512), (100, 150, 200))
            buffer = io.BytesIO()
            demo_img.save(buffer, format="PNG")
            images_b64 = [base64.b64encode(buffer.getvalue()).decode()]
        
        return GenerationResponse(
            task_id=task_id,
            status="completed",
            images=images_b64
        )
    
    except Exception as e:
        return GenerationResponse(task_id=task_id, status="failed", error=str(e))


@app.get("/generate/template/list")
async def list_templates():
    """List available generation templates"""
    if synthesizer is None:
        return {"templates": {}}
    
    return {"templates": synthesizer.templates}


@app.get("/generate/template/{modality}/{body_part}")
async def get_template(modality: str, body_part: str):
    """Get specific template"""
    if synthesizer is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    template = synthesizer.templates.get(body_part, {}).get(modality, {})
    
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    
    return template


@app.post("/generate/template/generate")
async def generate_from_template(
    modality: str = Form(...),
    body_part: str = Form(...),
    pathology: Optional[str] = Form(None),
    num_images: int = Form(1),
    severity: float = Form(0.5)
):
    """Generate using preset template"""
    task_id = str(uuid.uuid4())
    
    try:
        if synthesizer is not None and ldm_model is not None:
            results = synthesizer.generate_from_template(
                modality=modality,
                body_part=body_part,
                pathology=pathology,
                num_images=num_images,
                severity=severity
            )
            
            images_b64 = [image_to_base64(r.image) for r in results]
        else:
            # Demo
            demo_img = Image.new("RGB", (512, 512), (80, 120, 180))
            buffer = io.BytesIO()
            demo_img.save(buffer, format="PNG")
            images_b64 = [base64.b64encode(buffer.getvalue()).decode()]
        
        return GenerationResponse(
            task_id=task_id,
            status="completed",
            images=images_b64
        )
    
    except Exception as e:
        return GenerationResponse(task_id=task_id, status="failed", error=str(e))


# === Task Management ===

@app.get("/task/{task_id}", response_model=TaskResponse)
async def get_task_status(task_id: str):
    """Get task status"""
    if task_id not in task_queue:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = task_queue[task_id]
    
    return TaskResponse(
        task_id=task_id,
        status=task["status"],
        progress=task.get("progress", 0.0),
        created_at=task["created_at"]
    )


@app.get("/tasks")
async def list_tasks():
    """List all tasks"""
    return {
        "tasks": [
            {
                "task_id": tid,
                "status": t["status"],
                "progress": t.get("progress", 0.0),
                "created_at": t["created_at"]
            }
            for tid, t in task_queue.items()
        ]
    }


# === Medical Image Processing ===

@app.post("/process/dicom/upload")
async def upload_dicom(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload and process DICOM file"""
    task_id = str(uuid.uuid4())
    
    content = await file.read()
    
    try:
        # Save temporarily
        temp_path = Path("data/temp") / f"{task_id}.dcm"
        temp_path.parent.mkdir(exist_ok=True)
        temp_path.write_bytes(content)
        
        # Process in background
        background_tasks.add_task(process_dicom_task, task_id, str(temp_path))
        
        return {"task_id": task_id, "status": "processing"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def process_dicom_task(task_id: str, filepath: str):
    """Process DICOM in background"""
    try:
        data, info = DICOMProcessor.read_single_dicom(filepath)
        
        task_results[task_id] = {
            "status": "completed",
            "shape": data.shape,
            "modality": info.modality,
            "window_center": info.window_center,
            "window_width": info.window_width
        }
    except Exception as e:
        task_results[task_id] = {"status": "failed", "error": str(e)}
    finally:
        Path(filepath).unlink(missing_ok=True)


@app.post("/process/volume/render")
async def render_volume(req: VolumeRenderRequest):
    """Render 3D volume slice"""
    try:
        # Decode volume
        volume_data = base64.b64decode(req.volume_base64)
        volume_np = np.frombuffer(volume_data, dtype=np.float32)
        
        # Get axis
        axis_map = {"axial": 2, "sagittal": 1, "coronal": 0}
        axis = axis_map.get(req.axis, 2)
        
        # Extract slice
        if req.index == 0:
            slice_2d = VolumeRenderer.create_mip(volume_np.reshape(64, 64, 64), axis=axis)
        else:
            slice_2d = VolumeRenderer.extract_slice_2d(volume_np.reshape(64, 64, 64), axis=axis, index=req.index)
        
        # Apply windowing if provided
        if req.window_center is not None and req.window_width is not None:
            slice_2d = DICOMProcessor.apply_windowing(slice_2d, req.window_center, req.window_width)
        
        # Convert to image
        slice_2d = (slice_2d * 255).astype(np.uint8)
        img = Image.fromarray(slice_2d)
        
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        
        return {
            "slice_base64": base64.b64encode(buffer.getvalue()).decode(),
            "shape": slice_2d.shape
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process/augment")
async def augment_image(
    image_base64: str = Form(...),
    augmentations: str = Form("flip_h,flip_v,brightness")
):
    """Augment medical image"""
    try:
        img_tensor = base64_to_image(image_base64)
        
        aug_list = augmentations.split(",")
        aug_list = [a.strip() for a in aug_list if a.strip()]
        
        aug_img, _ = augment_medical_image(img_tensor, augmentations=aug_list)
        
        result_b64 = image_to_base64(aug_img)
        
        return {"image_base64": result_b64}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# === Root ===

@app.get("/")
async def root():
    return {
        "service": "SynthMed",
        "version": "1.0.0",
        "description": "AI医学影像合成数据平台",
        "endpoints": {
            "generate": "/generate",
            "batch": "/generate/batch",
            "mask": "/generate/from-mask",
            "template": "/generate/template/list",
            "gpu": "/gpu/status",
            "process": "/process/volume/render"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8015, workers=1)
