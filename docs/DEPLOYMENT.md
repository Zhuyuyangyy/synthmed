# SynthMed Deployment Guide

## Deployment Options

### 1. Docker Compose (Recommended)

The simplest way to deploy SynthMed is using Docker Compose.

**Prerequisites:**
- Docker 20.10+
- Docker Compose 2.0+
- NVIDIA Container Toolkit (for GPU support)

**Steps:**

```bash
# Clone the repository
git clone https://github.com/your-org/synthmed.git
cd synthmed

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop all services
docker-compose down
```

**Services:**
- Frontend: `http://localhost:3000`
- Sign Language API: `http://localhost:8000`
- Medical Image API: `http://localhost:8015`
- Nginx Proxy: `http://localhost:80`

### 2. Manual Deployment

**Backend:**

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Start sign language service
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4

# Start medical image service (separate terminal)
cd backend
python app.py
```

**Frontend:**

```bash
cd frontend
npm install
npm run build  # Production build
# Serve with nginx or: npm run preview
```

### 3. Cloud Deployment

**AWS ECS / Azure Container Instances / Google Cloud Run:**

1. Build and push Docker images to your container registry
2. Configure environment variables
3. Deploy with appropriate GPU instance types

**Recommended Instance Types:**
- CPU-only: `t3.xlarge` (4 vCPU, 16 GB RAM)
- GPU: `g4dn.xlarge` (4 vCPU, 16 GB RAM, 1x T4 GPU)

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENVIRONMENT` | `development` | `development` or `production` |
| `DEBUG` | `true` | Enable debug mode |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server port |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU device IDs |

---

## Health Checks

All services expose health endpoints:

```bash
# Sign language service
curl http://localhost:8000/health

# Medical image service
curl http://localhost:8015/health
```

---

## Scaling

### Horizontal Scaling

The sign language service supports multiple workers:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### GPU Scaling

For medical image synthesis, deploy multiple instances with different GPU assignments:

```bash
CUDA_VISIBLE_DEVICES=0 python app.py --port 8015
CUDA_VISIBLE_DEVICES=1 python app.py --port 8016
```

---

## Monitoring

### Logs

```bash
# Docker logs
docker-compose logs -f backend-sign
docker-compose logs -f backend-medical

# Application logs
tail -f /var/log/synthmed/sign-api.log
```

### GPU Monitoring

```bash
# Check GPU status
curl http://localhost:8015/gpu/status

# NVIDIA system monitor
nvidia-smi -l 1
```

---

## Troubleshooting

### Common Issues

1. **Port already in use:** Change the port in `config.yaml` or use `--port` flag
2. **CUDA out of memory:** Reduce batch size or use CPU mode
3. **MediaPipe model not found:** The detector falls back to mock mode automatically
4. **Frontend cannot connect to backend:** Check CORS settings and proxy configuration

### Reset

```bash
# Clean Docker volumes
docker-compose down -v

# Clean Python cache
find . -type d -name __pycache__ -exec rm -rf {} +
```
