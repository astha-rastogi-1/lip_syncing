# Lightweight Trade Study

| Factor | LatentSync | MuseTalk |
| ------ | ---------- | -------- |
| GPU | RTX 3090 | Tesla V100 |
| FPS | 25 | 30 | 
| Licence | Apache License 2.0 | MIT | 
| Language Support | Chinese | English, Chinese, Japanese | 
| Inference Setup | More steps for setup, instructions only for linux  | Clear instructions for both Linux and Windows |
| Use Cases | Post-processing videos (higher quality vid generation at slower speeds) | Real-Time Inference and Post-Processing |
| Active maintenance | Last updated Mar 2025 | Last updated Apr 2025 |
| Priority | High quality results with better accuracy, but takes much longer to process | Balanced between runtime and quality |

# Reasons to choose MuseTalk
- It is much easier to setup from a PoC perspective
- Documentation is clear, with no extra steps required for Windows setup or data processing
- Latest version released just last month
- Gives outputs faster than latentSync without using as much compute resource, thus giving a balanced tradeoff between speed vs quality

# Deployment
**Modal App**
- Created a modal application with docker image containing all the requirements
- The run_inference function takes in the video and audio files as bytes, converts them into files and stores them in Modal Volume, appending them to a config yaml file as is used in the repo for inference. This is an async function
- These parameters are then passed to MuseTalk's inference function, which returns an output that is converted into bytes and sent back to the API endpoint

**FastAPI Endpoint**
- Created two endpoints, one that uploads files and sends them to the modal application using modal's FunctionCall. The ID of this function call is logged so that its status can be checked, since this happens asynchronously.
- The other pings for status of the first api (pending, completed, or failed) by checking the status of the function call id.

# Observations
#### Ran on videos of around 5-10 seconds, audios ranging between 18-30 seconds
- MuseTalk does not perform well if the mouth of the subject is obscured from view
- It also does not perform as well if the subject in the original video was already speaking (lip motions before the audio are kept as the original, as well as during any pauses which make the lip syncing not great)
- It does seem to be performing pretty well for videos where the subject was not originally talking
- Takes longer time for processing for testing_nontalking_girl even though the video is only a second longer, and uses the same audio
- Running on A10G Gpus seems to take the least processing time while maintaining quality
- Quality does not differ much over GPUs


# Results
<!-- Inference Times Summary:

- Testing_other_girl_T4: 470.81s
- testing_male_T4: 292.72s
- Testing_other_girl_A10G: 228.74s
- testing_male_A10G: 132.07s
- Testing_other_girl_A100: 389.44s
- testing_male_A100: 260.63s

Inference Times Summary:
- testing_nontalking_girl_T4: failed
- testing_nontalking_girl_A10G: 777.56s
- testing_nontalking_girl_A100: 905.59s -->
<!-- 
| Test Name                     | Inference Time (s) | GPU Type   | Video Length (s) | Audio Length (s) | Result Length (s) |
|-------------------------------|--------------------|------------|------------------|------------------|-------------------|
| Testing_other_girl_T4         | 470.81             | T4         |                  |                  |                   |
| testing_male_T4               | 292.72             | T4         |                  |                  |                   |
| Testing_other_girl_A10G       | 228.74             | A10G       |                  |                  |                   |
| testing_male_A10G             | 132.07             | A10G       |                  |                  |                   |
| Testing_other_girl_A100       | 389.44             | A100       |                  |                  |                   |
| testing_male_A100             | 260.63             | A100       |                  |                  |                   |
| testing_nontalking_girl_T4    | failed             | T4         |                  |                  |                   |
| testing_nontalking_girl_A10G  | 777.56             | A10G       |                  |                  |                   |
| testing_nontalking_girl_A100  | 905.59             | A100       |                  |                  |                   | -->

| Test Name                   | Video Length (s) | Audio Length (s) | GPU Type | Inference Time (s) |
|-----------------------------|------------------|------------------|----------|--------------------|
| Testing_other_girl          |       6          |         28       | T4       | 470.81             |
|                             |                  |                  | A10G     | 228.74             | 
|                             |                  |                  | A100     | 389.44             |
| testing_male                |       4          |        18        | T4       | 292.72             |                   |
|                             |                  |                  | A10G     | 132.07             |                   |
|                             |                  |                  | A100     | 260.63             |                   |
| woman_further_away          |         11        |         25       | T4       | 357.89             |                   |
|                             |                  |                  | A10G     | 164.31             |                   |
|                             |                  |                  | A100     |  277.06            |                   |


<!-- # Evaluation
## Normalized Landmark Distance (LMD):
Measures the spatial distance between predicted and actual lip landmarks, indicating the synchronization accuracy. 

- Pros: Directly measures lip shape accuracy
- Cons: Error-prone due to landmark detection inaccuracies and inability to disentabgle sync from visual artefacts

## SyncNet Confidence Score (SyncScore):
Assesses the confidence of the synchronization between audio and visual components

### Frechet Inception Distance (FID):
Used for visual quality assessment, particularly when ground truth is unavailable, like in AI-generated lip movements. (Assesses realism of generated frames)
- Cons: Does not directly measure lip-sync accuracy

### SSIM/PSNR
Measures structural similarity between generated and real frames
- Cons: Does not directly measure lip-sync accuracy

### Cosine Similarity (CSIM):
Measures the similarity between identity embeddings of source and generated images, assessing identity preservation. 

### Lip-Sync Error (SyncNet based Metrics)
#### Lip-Sync Error Confidence (LSE-C):
Quantifies alignment confidence (higher values indicate better sync).
- Output: Distance between audio/lip embeddings (lower = better sync)

#### Lip-Sync Error Distance (LSE-D)
Measures the distance between audio and lip representations (lower values indicate better sync)
- Output: Confidence Score (higher=better sync)

**Input**: Audio waveform and video frames
**Usage**: Widely adopted for automatic evaluation

### Human Evaluation:
In some cases, human judges are used to evaluate the realism and naturalness of lip-syncing. 

## Emerging Methods:
### AV-HuBERT-Based Metrics
Proposed in recent work, these leverage robust audio-visual speech models:

- AVS_u: Unsupervised audio-visual cosine similarity.

- AVS_m: Compares generated and ground-truth sync via multimodal embeddings.

- AVS_v: Focuses on lip-shape similarity using visual features.

**Tradeoffs**:

Pros: More reliable than SyncNet, shift-invariant, and disentangle sync from visual quality.

Cons: Computationally intensive due to AV-HuBERT's size.

### Identity Preservation Metrics
Face Recognition Similarity: Measures how well the generated face retains the subject’s identity.

**Tradeoffs**:

Pros: Critical for personalized applications.

Cons: Requires pre-trained face recognition models.

## Recommendations
Combine Metrics: Use LSE-C/D or AVS metrics with FID/SSIM to balance sync accuracy and visual quality.

Prioritize Robustness: For unconstrained videos, prefer AV-HuBERT metrics over SyncNet.

Validate with Humans: Supplement automatic metrics with small-scale perceptual studies.

By leveraging these methods, researchers can holistically assess lip-sync models while addressing the limitations of individual metrics. -->

# Evaluation
## Landmark Based Metrics

### **Mouth Landmark Distance (LMD):**

Measures the spatial distance between predicted and actual lip landmarks, indicating the synchronization accuracy.

- Pros: Directly measures lip shape accuracy
- Cons: Error-prone due to landmark detection inaccuracies and inability to disentabgle sync from visual artefacts

**Input**: Predicted and ground-truth facial landmarks

**Output**: Spatial distance between landmarks (lower = better shape accuracy)

**Usage**: Common in early works to quantify lip-shape errors

**Tradeoffs**: 

- Strengths: Directly measures lip-shape fidelity
- Limitations: Errors can come from landmark detection inaccuracies

## **SyncNet Based Metrics:**

### **Lip-Sync Error (SyncNet based Metrics)**

- **Lip-Sync Error Confidence (LSE-C):** Quantifies alignment confidence (higher values indicate better sync).
    - Output: Distance between audio/lip embeddings (lower = better sync)
- **Lip-Sync Error Distance (LSE-D):** Measures the distance between audio and lip representations (lower values indicate better sync)
    - Output: Confidence Score (higher=better sync)

**Input**: Audio waveform and video frames **Usage**: Widely adopted for automatic evaluation

**Usage:** SyncNet is pretrained on contrastive audio-visual pairs to detect syncronization

**Tradeoffs**: 

- Strengths: Automated, scalable, benchmarked on datasets like LRS2
- Limitations: Sensitive to facial shifts/translations and affine transformations

## Visual Quality Assesment

### **Frechet Inception Distance (FID):**

Used for visual quality assessment, particularly when ground truth is unavailable, like in AI-generated lip movements. (Assesses realism of generated frames)

- Cons: Does not directly measure lip-sync accuracy

### **SSIM/PSNR**

Measures structural similarity between generated and real frames

- Cons: Does not directly measure lip-sync accuracy

### **Cosine Similarity (CSIM):**

Measures the similarity between identity embeddings of source and generated images, assessing identity preservation.

### **Human Evaluation:**

In some cases, human judges can be used to evaluate the realism and naturalness of lip-syncing.


