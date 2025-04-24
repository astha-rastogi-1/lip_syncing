import modal
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import uuid
import shutil
import os
import yaml

app = modal.App("musetalk-inference")

# Mount your local MuseTalk directory (including models) into the container
musetalk_mount = modal.Mount.from_local_dir(
    local_path="MuseTalk",  # your local cloned repo with models
    remote_path="/root/MuseTalk"
)

# Build the image
musetalk_image = (
    modal.Image
    .from_registry("nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04", add_python="3.10")
    .apt_install("git", "ffmpeg", "wget", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.0.1",
        "torchvision==0.15.2",
        "torchaudio==2.0.2",
        "transformers==4.37.0",
        "diffusers==0.26.0",
        "openmim",
        "fastapi[standard]"
    )
    .copy_local_dir("MuseTalk", "/root/MuseTalk")
    .run_commands(
        "mim install mmengine",
        "mim install 'mmcv==2.0.1'",
        "mim install 'mmdet==3.1.0'",
        "mim install 'mmpose==1.1.0'",
        "pip install -r /root/MuseTalk/requirements.txt"
    )
)

@app.function(
    image=musetalk_image,
    gpu="T4",
    mounts=[musetalk_mount],  # ← models will be available in /root/MuseTalk/models
    timeout=1200
)
async def run_inference(job_id: str, video_content: bytes, audio_content: bytes):
    import subprocess
    import os
    from pathlib import Path
    import yaml

    os.chdir("/root/MuseTalk")
    input_dir = Path("tmp") / job_id
    input_dir.mkdir(parents=True, exist_ok=True)
    # Write received bytes to container filesystem
    video_path = input_dir / "input_video.mp4"
    audio_path = input_dir / "input_audio.wav"
    with open(video_path, "wb") as f:
        f.write(video_content)
    with open(audio_path, "wb") as f:
        f.write(audio_content)

    # Create output directory
    result_dir = Path("results") / job_id
    result_dir.mkdir(parents=True, exist_ok=True)

    # Path to the dynamically generated YAML config file
    config_path = Path("configs/inference/test.yaml")

    # Create a YAML configuration file with paths to the video and audio files
    config_data = {
        "task_0": {  # Add this nested level
            "video_path": str(video_path),
            "audio_path": str(audio_path)
        }
    }
    
    with open(config_path, "w") as yaml_file:
        yaml.dump(config_data, yaml_file, sort_keys=False, default_flow_style=False)

    # # Ensure the paths are compatible with Linux and copy files to the container's accessible directory
    # container_video_path = f"/root/MuseTalk/tmp/{job_id}_video.mp4"
    # container_audio_path = f"/root/MuseTalk/tmp/{job_id}_audio.wav"

    # # Copy video and audio files to the accessible directory in the container
    # shutil.copy(video_path, container_video_path)
    # shutil.copy(audio_path, container_audio_path)

    # result_dir = f"results/{job_id}"
    # # Ensure the result directory exists
    # os.makedirs(result_dir, exist_ok=True)
    unet_model = "models/musetalkV15/unet.pth"
    unet_config = "models/musetalkV15/musetalk.json"
    ffmpeg_path = "/usr/bin/ffmpeg"

    cmd = [
        "python", "-m", "scripts.inference",
        "--inference_config", str(config_path),
        "--result_dir", str(result_dir),
        "--unet_model_path", unet_model,
        "--unet_config", unet_config,
        "--version", "v15",
        "--ffmpeg_path", ffmpeg_path,
    ]

    subprocess.run(cmd, check=True)
    # return f"Inference complete. Output in {result_dir}"
    output_video_path = result_dir / "v15/input_video_input_audio.mp4"
    with open(output_video_path, "rb") as f:
        video_bytes = f.read()
    return video_bytes

## FASTAPI ENDPOINTS
# @app.function(image=musetalk_image)
# @modal.fastapi_endpoint(method="POST")  # Expose as POST endpoint
# async def process_files(video: UploadFile = File(...), audio: UploadFile = File(...)):
#     # Create a unique directory to store the uploaded files
#     unique_id = uuid.uuid4().hex
#     temp_video_path = f"/tmp/{unique_id}_video.mp4"
#     temp_audio_path = f"/tmp/{unique_id}_audio.mp3"
#     # Save uploaded files
#     with open(temp_video_path, "wb") as video_file:
#         shutil.copyfileobj(video.file, video_file)
    
#     with open(temp_audio_path, "wb") as audio_file:
#         shutil.copyfileobj(audio.file, audio_file)
    
#     # Call the inference function to process the files
#     result_dir = await modal.Function(run_inference)(temp_video_path, temp_audio_path)
#     # The processed video will be saved in the result directory
#     output_video_path = os.path.join(result_dir, "output_video.mp4")

#     # Return the video file as a response
#     return FileResponse(output_video_path)

# image = modal.Image.debian_slim().pip_install("fastapi[standard]")
# @app.function(image=image)
# @modal.asgi_app()
# def app_instance():
#     from fastapi import FastAPI
#     web_app = FastAPI()

#     # Add the endpoint to FastAPI
#     web_app.post("/upload")(process_files)
#     return web_app