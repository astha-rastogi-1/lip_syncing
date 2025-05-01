from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse
from modal.functions import FunctionCall
from uuid import uuid4
import shutil
import os
import modal
import tempfile
import json
from pathlib import Path

# Initialize FastAPI app
app = FastAPI()

# Store pending jobs
pending_jobs = {}
job_log_path = Path("job_logs.json")

# Load the Modal function remotely by referencing the app and function name
# run_inference_function = modal.Function.from_name("musetalk-inference", "run_inference")

# Save job_id persistently
def save_job_metadata(job_id, status="pending"):
    if job_log_path.exists():
        with open(job_log_path, "r") as f:
            data = json.load(f)
    else:
        data = {}
    data[job_id] = status
    with open(job_log_path, "w") as f:
        json.dump(data, f)

def load_job_metadata():
    if job_log_path.exists():
        with open(job_log_path, "r") as f:
            return json.load(f)
    return {}

@app.post("/jobs/{gpu_type}")
async def create_job(gpu_type: str, video: UploadFile = File(...), audio: UploadFile = File(...)):
    """
    Endpoint to accept video and audio files, spawn the inference job asynchronously,
    and return the job ID with its status.
    """
    # Create a unique job ID for the task
    job_id = str(uuid4())

    # Read files directly into memory instead of using temp files
    video_content = await video.read()
    audio_content = await audio.read()
    # pending_jobs[job_id] = 'pending'
    save_job_metadata(job_id, status="pending")
    # Spawn Modal function with file contents
    function_name = f"run_inference_{gpu_type.lower()}"
    run_inference_function = modal.Function.from_name("musetalk-inference", function_name)
    function_call = run_inference_function.spawn(
        job_id=job_id,
        video_content=video_content,  # Pass bytes directly
        audio_content=audio_content,  # Pass bytes directly
    )
    fc_id = function_call.object_id
    print('FC ID type: ', type(fc_id))
    # Track the function call by storing it in the pending jobs dictionary
    # pending_jobs[job_id] = function_call
    save_job_metadata(job_id, status=fc_id)
    pending_jobs[job_id] = function_call
    # Return a response with the job ID and initial status
    return JSONResponse(content={
        "job_id": job_id,
        "status": "pending",
    })

@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """
    Endpoint to retrieve the status of the job using the job ID.
    """
    # Check if the job exists
    stored_fc_ids = load_job_metadata()
    # if job_id not in stored_fc_ids.keys():
    #     raise HTTPException(status_code=404, detail="Job not found")

    if job_id in pending_jobs.keys():
        function_call = pending_jobs[job_id]
    else:
        function_call = FunctionCall.from_id(stored_fc_ids[job_id])
    try:
        # Check if job is done (non-blocking)
        video_bytes = function_call.get(timeout=0)  # Timeout=0 for polling
        local_path = f"./results/{job_id}_output_video.mp4"
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(video_bytes)
        return FileResponse(local_path, media_type="video/mp4", filename="lip_synced_video.mp4")
        
    except TimeoutError:
        # Job still pending
        return JSONResponse({"job_id": job_id, "status": "pending"})
        
    except Exception as e:
        # Job failed
        raise HTTPException(500, detail=f"Job failed: {str(e)}")
