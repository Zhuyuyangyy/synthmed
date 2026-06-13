# SynthMed API Reference

## Overview

SynthMed exposes two FastAPI services:

1. **Sign Language Translation API** (port 8000) -- `backend/main.py`
2. **Medical Image Synthesis API** (port 8015) -- `backend/app.py`

Both services provide automatic OpenAPI documentation at `/docs` (Swagger UI) and `/redoc` (ReDoc).

---

## Sign Language Translation API (Port 8000)

### Health Check

```
GET /health
```

Returns system health status, service name, version, and compute device.

**Response:**
```json
{
  "status": "healthy",
  "service": "multimodal_sign_language_system",
  "version": "1.0.0",
  "device": "cpu"
}
```

### Detailed Health Check

```
GET /health/detailed
```

Returns detailed health including model loading status.

### Sign Language Recognition

```
POST /api/v1/recognize
```

Recognize sign language from a sequence of base64-encoded video frames.

**Request Body:** `List[str]` -- Array of base64-encoded image frames (max 30).

**Response:**
```json
{
  "status": "success",
  "text": "你好",
  "confidence": 0.96,
  "translation_id": "uuid-string",
  "processing_time_ms": 150.5,
  "frames_processed": 15
}
```

### Text-to-Sign Generation

```
POST /api/v1/generate?text=你好&format=mp4
```

Generate sign language video from text input.

**Query Parameters:**
- `text` (string, required): Input text to translate
- `format` (string, optional): Output format -- `mp4`, `base64`, `keypoints` (default: `mp4`)

### Bidirectional Translation

```
POST /api/v1/translate
```

Unified endpoint for both translation directions.

**Query Parameters:**
- `direction` (string, required): `sign_to_text` or `text_to_sign`
- `frames` (List[str], optional): Video frames for sign-to-text
- `text` (string, optional): Input text for text-to-sign

### WebSocket Streaming

```
WS /ws/stream
```

Real-time bidirectional streaming translation endpoint.

**Client Messages:**
```json
{
  "type": "frame",
  "frame": "base64-encoded-frame",
  "timestamp_ms": 1234567890.0
}
```

**Server Responses:**
```json
{
  "type": "interim",
  "timestamp_ms": 1234567890.0,
  "hand_count": 2,
  "face_detected": true,
  "body_detected": true
}
```

### Model Status

```
GET /api/v1/models/status
```

### CSL Glosses

```
GET /api/v1/csl/glosses
```

Returns the list of supported CSL (Chinese Sign Language) vocabulary.

---

## Medical Image Synthesis API (Port 8015)

### Health Check

```
GET /health
```

### GPU Status

```
GET /gpu/status
```

Returns GPU utilization, memory allocation, and temperature.

### Generate Single Image

```
POST /generate
```

**Request Body:**
```json
{
  "prompt": "chest CT scan, lung window",
  "negative_prompt": "low quality, blurry, artifacts",
  "width": 512,
  "height": 512,
  "num_inference_steps": 50,
  "guidance_scale": 7.5,
  "seed": 42,
  "num_images": 1,
  "modality": "ct",
  "body_part": "chest",
  "pathology": "nodule",
  "severity": 0.5
}
```

**Response:**
```json
{
  "task_id": "uuid-string",
  "status": "completed",
  "images": ["base64-encoded-png"],
  "metadata": {
    "modality": "ct",
    "body_part": "chest",
    "pathology": "nodule",
    "seed": 42,
    "demo": true
  }
}
```

### Batch Generation

```
POST /generate/batch
```

Generate multiple images in a single request.

### Mask-Conditioned Generation

```
POST /generate/from-mask
```

Generate images conditioned on a segmentation mask using ControlNet.

### Template-Based Generation

```
GET /generate/template/list
GET /generate/template/{modality}/{body_part}
POST /generate/template/generate
```

### DICOM Processing

```
POST /process/dicom/upload
```

Upload and process a DICOM file.

### Volume Rendering

```
POST /process/volume/render
```

Render a 2D slice from a 3D volume.

### Image Augmentation

```
POST /process/augment
```

Apply data augmentation to a medical image.

### Task Management

```
GET /task/{task_id}
GET /tasks
```

---

## Error Handling

All endpoints return standard HTTP status codes:

- `200`: Success
- `400`: Bad request (invalid parameters)
- `404`: Resource not found
- `500`: Internal server error
- `503`: Service unavailable (models not loaded)

Error responses follow the format:
```json
{
  "detail": "Error description"
}
```
