"""
Qwen Image Editor - FastAPI Backend
Wraps ComfyUI workflow for AI-powered image editing.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import uuid
import json
import os
import copy
import shutil
from datetime import datetime
from pathlib import Path
import asyncio

app = FastAPI(title="Qwen Image Editor API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
COMFYUI_URL = os.environ.get("COMFYUI_URL", "http://localhost:1111")

WORKFLOW_FILE = Path("workflow_api.json")

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# In-memory job storage
jobs = {}


class GenerateRequest(BaseModel):
    image_path: str
    prompt: str
    steps: int = 4
    cfg: float = 1.0
    seed: int = -1
    strength: float = 1.0


def load_workflow_template() -> dict:
    """Load the ComfyUI workflow template."""
    with open(WORKFLOW_FILE, "r") as f:
        return json.load(f)


def modify_workflow(
    workflow: dict, image_filename: str, prompt: str, settings: dict
) -> dict:
    """
    Modify workflow with user inputs.

    Node mapping from the Qwen Image Edit workflow:
      - 78  (LoadImage)      → input image filename
      - 435 (Prompt)         → user edit prompt
      - 433:3 (KSampler)    → steps, cfg, seed
      - 433:75 (CFGNorm)    → strength
    """
    wf = copy.deepcopy(workflow)

    # Update image loading node (node 78 - LoadImage)
    if "78" in wf:
        wf["78"]["inputs"]["image"] = image_filename

    # Update prompt node (node 435 - PrimitiveStringMultiline)
    if "435" in wf:
        wf["435"]["inputs"]["value"] = prompt

    # Update sampler settings (node 433:3 - KSampler)
    sampler_node = "433:3"
    if sampler_node in wf:
        wf[sampler_node]["inputs"]["steps"] = settings.get("steps", 4)
        wf[sampler_node]["inputs"]["cfg"] = settings.get("cfg", 1.0)
        seed = settings.get("seed", -1)
        if seed >= 0:
            wf[sampler_node]["inputs"]["seed"] = seed
        else:
            # Random seed
            import random

            wf[sampler_node]["inputs"]["seed"] = random.randint(0, 2**32 - 1)

    # Update strength (node 433:75 - CFGNorm)
    strength_node = "433:75"
    if strength_node in wf:
        wf[strength_node]["inputs"]["strength"] = settings.get("strength", 1.0)

    return wf


async def run_comfyui_workflow(workflow: dict, job_id: str) -> str:
    """Execute ComfyUI workflow via its REST API and return the output filename."""
    import aiohttp

    try:
        async with aiohttp.ClientSession() as session:
            # Queue the prompt
            payload = {
                "prompt": workflow,
                "client_id": job_id,
            }

            async with session.post(f"{COMFYUI_URL}/prompt", json=payload) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise Exception(f"Failed to queue prompt: {text}")
                data = await resp.json()
                prompt_id = data["prompt_id"]

            # Poll for completion
            max_wait = 300  # 5-minute timeout
            elapsed = 0
            while elapsed < max_wait:
                await asyncio.sleep(2)
                elapsed += 2

                async with session.get(f"{COMFYUI_URL}/history/{prompt_id}") as resp:
                    if resp.status == 200:
                        history = await resp.json()
                        if prompt_id in history:
                            outputs = history[prompt_id].get("outputs", {})

                            # Node 60 is SaveImage
                            if "60" in outputs:
                                images = outputs["60"].get("images", [])
                                if images:
                                    return images[0]["filename"]

                            # Check if execution had errors
                            status_data = history[prompt_id].get("status", {})
                            if status_data.get("status_str") == "error":
                                msgs = status_data.get("messages", [])
                                raise Exception(f"ComfyUI execution error: {msgs}")

                # Update progress estimate
                if job_id in jobs:
                    jobs[job_id]["progress"] = min(90, 30 + (elapsed // 2) * 5)

            raise Exception("Timeout waiting for ComfyUI to complete")

    except aiohttp.ClientError as e:
        raise Exception(f"Cannot connect to ComfyUI at {COMFYUI_URL}: {e}")


@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    """Upload image for processing."""
    try:
        # Validate file type
        allowed_types = {"image/jpeg", "image/png", "image/webp"}
        if file.content_type not in allowed_types:
            raise HTTPException(400, "Invalid file type. Use PNG, JPG, or WEBP.")

        # Generate unique filename preserving extension
        ext = file.filename.rsplit(".", 1)[-1] if "." in file.filename else "png"
        filename = f"{uuid.uuid4()}.{ext}"
        file_path = UPLOAD_DIR / filename

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return {
            "success": True,
            "path": str(file_path),
            "filename": filename,
            "url": f"/uploads/{filename}",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Upload failed: {str(e)}")


@app.post("/api/generate")
async def generate_edit(request: GenerateRequest, background_tasks: BackgroundTasks):
    """Start image generation job."""
    job_id = str(uuid.uuid4())

    jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "progress": 0,
        "prompt": request.prompt,
        "created_at": datetime.now(),
        "output_url": None,
        "error": None,
        "completed_at": None,
    }

    background_tasks.add_task(process_generation, job_id, request)

    return {"success": True, "job_id": job_id, "status": "pending"}


async def upload_image_to_comfyui(file_path: str) -> str:
    """Upload an image to ComfyUI's input directory via its /upload/image API."""
    import aiohttp

    filepath = Path(file_path)
    async with aiohttp.ClientSession() as session:
        data = aiohttp.FormData()
        data.add_field(
            "image",
            open(filepath, "rb"),
            filename=filepath.name,
            content_type="image/png",
        )
        data.add_field("overwrite", "true")

        async with session.post(f"{COMFYUI_URL}/upload/image", data=data) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise Exception(f"Failed to upload image to ComfyUI: {text}")
            result = await resp.json()
            # ComfyUI returns {"name": "filename.png", "subfolder": "", "type": "input"}
            return result["name"]


async def download_from_comfyui(filename: str, dest: Path):
    """Download a generated image from ComfyUI's /view endpoint to a local path."""
    import aiohttp

    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"{COMFYUI_URL}/view", params={"filename": filename}
        ) as resp:
            if resp.status != 200:
                raise Exception(f"Failed to download image from ComfyUI: {resp.status}")
            with open(dest, "wb") as f:
                async for chunk in resp.content.iter_chunked(8192):
                    f.write(chunk)


async def process_generation(job_id: str, request: GenerateRequest):
    """Background task for image generation."""
    try:
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["progress"] = 10

        # Upload image to ComfyUI's input directory via API
        comfyui_image_name = await upload_image_to_comfyui(request.image_path)

        jobs[job_id]["progress"] = 20

        # Load and modify workflow
        workflow = load_workflow_template()
        settings = {
            "steps": request.steps,
            "cfg": request.cfg,
            "seed": request.seed,
            "strength": request.strength,
        }
        modified_workflow = modify_workflow(
            workflow, comfyui_image_name, request.prompt, settings
        )

        jobs[job_id]["progress"] = 30

        # Run ComfyUI
        output_filename = await run_comfyui_workflow(modified_workflow, job_id)

        jobs[job_id]["progress"] = 95

        # Download output image from ComfyUI and save locally
        # (browser can't reach ComfyUI directly)
        dest_name = f"{job_id}_{output_filename}"
        dest_path = OUTPUT_DIR / dest_name
        await download_from_comfyui(output_filename, dest_path)
        jobs[job_id]["output_url"] = f"/outputs/{dest_name}"

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["completed_at"] = datetime.now()

    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        print(f"Job {job_id} failed: {e}")


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    """Get job status."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    job = jobs[job_id]
    return {
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "output_url": job.get("output_url"),
        "error": job.get("error"),
    }


@app.get("/api/history")
async def get_history():
    """Get recent completed generations."""
    completed_jobs = [
        job
        for job in jobs.values()
        if job["status"] == "completed" and job.get("output_url")
    ]
    completed_jobs.sort(key=lambda x: x.get("completed_at", datetime.min), reverse=True)

    return {
        "history": [
            {
                "job_id": job["job_id"],
                "output_url": job["output_url"],
                "prompt": job.get("prompt", ""),
                "timestamp": job.get("completed_at", "").isoformat()
                if job.get("completed_at")
                else None,
            }
            for job in completed_jobs[:10]
        ]
    }


@app.get("/api/uploads")
async def list_uploads():
    """List all uploaded images."""
    if not UPLOAD_DIR.exists():
        return {"uploads": []}

    files = []
    for f in UPLOAD_DIR.iterdir():
        if f.is_file() and f.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            files.append(
                {
                    "filename": f.name,
                    "url": f"/uploads/{f.name}",
                    "size": f.stat().st_size,
                    "created_at": datetime.fromtimestamp(f.stat().st_ctime).isoformat(),
                }
            )
    files.sort(key=lambda x: x["created_at"], reverse=True)
    return {"uploads": files}


@app.get("/api/outputs")
async def list_outputs():
    """List all generated images."""
    if not OUTPUT_DIR.exists():
        return {"outputs": []}

    files = []
    for f in OUTPUT_DIR.iterdir():
        if f.is_file() and f.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            files.append(
                {
                    "filename": f.name,
                    "url": f"/outputs/{f.name}",
                    "size": f.stat().st_size,
                    "created_at": datetime.fromtimestamp(f.stat().st_ctime).isoformat(),
                }
            )
    files.sort(key=lambda x: x["created_at"], reverse=True)
    return {"outputs": files}


@app.get("/uploads/{filename}")
async def get_upload(filename: str):
    """Serve uploaded files."""
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(file_path)


@app.get("/outputs/{filename}")
async def get_output(filename: str):
    """Serve generated output files."""
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(file_path)


@app.get("/api/health")
async def health_check():
    """Health check with ComfyUI connectivity status."""
    comfyui_ok = await check_comfyui_connection()
    return {
        "status": "healthy",
        "comfyui_url": COMFYUI_URL,
        "comfyui_connected": comfyui_ok,
        "queued_jobs": len([j for j in jobs.values() if j["status"] == "pending"]),
        "active_jobs": len([j for j in jobs.values() if j["status"] == "processing"]),
    }


COMFYUI_SCRIPT = os.environ.get(
    "COMFYUI_SCRIPT", os.path.expanduser("~/start_comfyui.sh")
)


@app.post("/api/comfyui/start")
async def start_comfyui():
    """Start ComfyUI if not running."""
    import subprocess

    if await check_comfyui_connection():
        return {"success": True, "message": "ComfyUI is already running"}

    try:
        subprocess.Popen(
            ["bash", COMFYUI_SCRIPT],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return {"success": True, "message": "Starting ComfyUI..."}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/comfyui/status")
async def comfyui_status():
    """Get ComfyUI status."""
    connected = await check_comfyui_connection()
    return {
        "connected": connected,
        "url": COMFYUI_URL,
    }


async def check_comfyui_connection() -> bool:
    """Check if ComfyUI is running and reachable."""
    try:
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{COMFYUI_URL}/system_stats", timeout=aiohttp.ClientTimeout(total=3)
            ) as resp:
                return resp.status == 200
    except Exception:
        return False


# Serve frontend
@app.get("/")
async def serve_frontend():
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse("<h1>Frontend not found</h1>", status_code=404)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
