import modal

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
        "openmim"
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
    timeout=600
)
def run_inference(video_path: str, audio_path: str):
    import subprocess
    import os

    os.chdir("/root/MuseTalk")

    result_dir = "results/test"
    unet_model = "models/musetalkV15/unet.pth"
    unet_config = "models/musetalkV15/musetalk.json"
    ffmpeg_path = "/usr/bin/ffmpeg"

    cmd = [
        "python", "-m", "scripts.inference",
        "--inference_config", "configs/inference/test.yaml",
        "--result_dir", result_dir,
        "--unet_model_path", unet_model,
        "--unet_config", unet_config,
        "--version", "v15",
        "--ffmpeg_path", ffmpeg_path,
        "--video_path", video_path,
        "--audio_path", audio_path
    ]

    subprocess.run(cmd, check=True)
    return f"Inference complete. Output in {result_dir}"
