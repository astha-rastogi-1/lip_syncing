import requests
import time
from pathlib import Path
import os

# URL of your FastAPI server
BASE_URL = "http://localhost:8000/jobs"

# Test samples: (video_path, audio_path)
TEST_CASES = [
    (os.path.join(os.getcwd(), "testing", "woman_further_away.mp4" ), os.path.join(os.getcwd(), "testing", "ai_female.wav")),
    # (os.path.join(os.getcwd(), "testing", "testing_male.mp4"), os.path.join(os.getcwd(), "testing", "ai_male_trimmed.wav")),
]

# List of GPU types to test (must match FastAPI endpoints: t4, a10g, a100)
# GPU_TYPES = ["T4", "A10G", "A100"]
GPU_TYPES = ["A10G"]

# Output directory
Path("results").mkdir(exist_ok=True)

def run_test(video_path, audio_path, gpu_type):
    print(f"\nTesting on GPU: {gpu_type.upper()} | Video: {video_path} | Audio: {audio_path}")
    with open(video_path, "rb") as vfile, open(audio_path, "rb") as afile:
        files = {
            "video": ("video.mp4", vfile, "video/mp4"),
            "audio": ("audio.wav", afile, "audio/wav")
        }
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/{gpu_type}", files=files)
        response.raise_for_status()
        job_id = response.json()["job_id"]
        print(f"Job ID: {job_id}")

    # Poll for result
    status_url = f"{BASE_URL}/{job_id}"
    while True:
        time.sleep(30)
        status_resp = requests.get(status_url)
        if status_resp.status_code == 200 and status_resp.headers.get("content-type") == "video/mp4":
            duration = time.time() - start_time
            output_path = f"results/{job_id}_{gpu_type}.mp4"
            with open(output_path, "wb") as f:
                f.write(status_resp.content)
            print(f"Done in {duration:.2f}s → Saved to {output_path}")
            return duration
        elif status_resp.status_code == 200 and status_resp.json().get("status") == "pending":
            print("Still processing...")
        else:
            print("Failed or unknown response:", status_resp.text)
            return None

# Run all tests
results = {}
for gpu in GPU_TYPES:
    for vid_path, aud_path in TEST_CASES:
        key = f"{Path(vid_path).stem}_{gpu}"
        duration = run_test(vid_path, aud_path, gpu)
        results[key] = duration

# Summary
print("\nInference Times Summary:")
for case, time_taken in results.items():
    print(f"{case}: {time_taken:.2f}s" if time_taken else f"{case}: failed")
