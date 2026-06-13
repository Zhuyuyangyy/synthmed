# SynthMed Technical Architecture

## System Overview

SynthMed is a dual-purpose AI platform combining sign language translation with medical image synthesis. The architecture follows a modular, microservice-oriented design.

---

## Component Architecture

### 1. Sign Language Translation Service

```
                    +-----------------+
                    |   FastAPI App   |
                    |   (main.py)     |
                    +--------+--------+
                             |
              +--------------+--------------+
              |              |              |
    +---------v--+  +--------v-----+  +-----v--------+
    |  MediaPipe  |  |  Transformer |  |  Text-to-Sign|
    |  Detector   |  |  Fusion      |  |  Generator   |
    +---------+--+  +--------+-----+  +-----+--------+
              |              |              |
    +---------v--------------v--------------v--------+
    |              Core Processing Layer              |
    +------------------------------------------------+
```

**MediaPipe Detector** (`core/vision/mediapipe_detector.py`):
- Hand landmark detection: 21 points x 2 hands
- Face mesh detection: 468 points + 52 blendshapes
- Pose detection: 33 body landmarks
- Returns `MultiModalKeypoints` dataclass

**Transformer Fusion** (`core/fusion/transformer_fusion.py`):
- Three parallel `ChannelTransformer` encoders
- Cross-channel spatial attention via `MultiheadAttention`
- MLP fusion projection
- Classification head for CSL gloss prediction

**Text-to-Sign Generator** (`core/generation/text_to_sign.py`):
- Text-to-CSL gloss mapping
- World model keypoint sequence generation
- Video/keyframe/skeleton rendering

### 2. Medical Image Synthesis Service

```
                    +-----------------+
                    |   FastAPI App   |
                    |   (app.py)      |
                    +--------+--------+
                             |
              +--------------+--------------+
              |              |              |
    +---------v--+  +--------v-----+  +-----v--------+
    |  Latent     |  |  Anatomical  |  |  Data        |
    |  Diffusion  |  |  ControlNet  |  |  Processor   |
    +---------+--+  +--------+-----+  +-----+--------+
              |              |              |
    +---------v--------------v--------------v--------+
    |              Privacy & Security Layer           |
    |  (Differential Privacy, Federated Learning)     |
    +------------------------------------------------+
```

**Latent Diffusion Model** (`core/diffusion/ldm.py`):
- Stable Diffusion 2.1 architecture
- CLIP text encoder for prompt conditioning
- DDIM scheduler for efficient inference
- VAE encoder/decoder for latent space operations
- LoRA fine-tuning support

**Anatomical ControlNet** (`core/controlnet/anatomical.py`):
- Segmentation mask-based structural constraints
- Sobel boundary extraction
- Topology consistency loss
- Multi-modal consistency loss (MRI-CT alignment)

**Data Processor** (`core/data/processor.py`):
- DICOM/NIfTI/NRRD format support
- CT windowing presets
- 3D volume rendering (MIP, orthogonal slices)
- Data augmentation pipeline

---

## Data Flow

### Sign Language Recognition Flow

```
Video Frame (RGB)
    |
    v
MediaPipe Detection
    |-- Hand: 21 landmarks x 3 coords x 2 hands = 126-dim
    |-- Face: 468 landmarks -> reduced to 300-dim
    |-- Body: 33 landmarks x 3 coords = 99-dim (upper 25 = 75-dim)
    |
    v
Feature Vector Construction (501-dim total)
    |
    v
Sequence Buffer (30 frames)
    |
    v
Tri-Channel Transformer Fusion
    |-- Hand Encoder: (batch, 30, 126) -> (batch, 256)
    |-- Face Encoder: (batch, 30, 300) -> (batch, 256)
    |-- Body Encoder: (batch, 30, 75)  -> (batch, 256)
    |-- Concatenation: (batch, 768)
    |-- Cross-Attention: (batch, 768) -> (batch, 768)
    |-- Fusion MLP: (batch, 768) -> (batch, 256)
    |
    v
Classification Head: (batch, 256) -> (batch, num_classes)
    |
    v
CSL Gloss Prediction + Confidence
```

### Medical Image Synthesis Flow

```
Text Prompt + Anatomical Mask
    |
    v
CLIP Tokenizer + Text Encoder
    |
    v
Latent Space Initialization (random noise)
    |
    v
DDIM Denoising Loop (50 steps)
    |-- UNet2D predicts noise
    |-- Classifier-free guidance (scale=7.5)
    |-- ControlNet conditioning (optional)
    |
    v
VAE Decoder: Latent -> Image
    |
    v
Post-processing + DP Noise (optional)
    |
    v
Synthetic Medical Image (512x512)
```

---

## Privacy Architecture

### Differential Privacy

- **DP-SGD Training**: Gradient clipping + Gaussian noise addition
- **Privacy Budget Tracking**: Epsilon-delta composition
- **Post-processing Noise**: Laplace/Gaussian mechanism for output perturbation

### Federated Learning

- **FedAvg Aggregation**: Weighted averaging of client model parameters
- **Secure Aggregation**: Encrypted parameter transmission
- **Privacy Amplification**: Subsampling-based epsilon reduction

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Web Framework | FastAPI 0.110+ |
| Deep Learning | PyTorch 2.0+ |
| Diffusion Models | Diffusers 0.20+ |
| Transformers | HuggingFace Transformers 4.30+ |
| Computer Vision | OpenCV 4.9+, MediaPipe 0.10+ |
| Medical Imaging | SimpleITK, nibabel, pydicom |
| Privacy | Opacus (DP-SGD) |
| Fine-tuning | PEFT (LoRA) |
| Frontend | Vue 3, TypeScript, Element Plus |
| Build Tool | Vite 5.2+ |
| Containerization | Docker, Docker Compose |
| CI/CD | GitHub Actions |
