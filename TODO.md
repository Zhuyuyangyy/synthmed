# SynthMed TODO -- Innovation Task Tracker

## Priority Legend
- **P0**: Critical -- blocks core functionality
- **P1**: High -- significant value addition
- **P2**: Medium -- nice to have
- **P3**: Low -- future consideration

---

## Privacy-Preserving Synthetic Data (P0)

- [ ] **Integrate Opacus DP-SGD** into LDM fine-tuning pipeline
  - Add `PrivacyEngine` wrapper around the training loop
  - Configure `noise_multiplier`, `max_grad_norm`, `sample_rate`
  - Track privacy budget (epsilon, delta) per training run
  - File: `backend/core/diffusion/ldm.py`

- [ ] **Implement output perturbation layer** for synthetic images
  - Add configurable Laplace/Gaussian noise injection post-generation
  - Expose epsilon parameter in API (`/generate` endpoint)
  - Validate privacy guarantee via membership inference attack tests
  - File: `backend/core/privacy/dp_mechanism.py` (new)

- [ ] **Privacy budget composition tracker**
  - Implement advanced composition theorem (Kairouz et al.)
  - Persist budget state across API calls
  - Alert when budget exhausted
  - File: `backend/core/privacy/budget_tracker.py` (new)

---

## Federated Learning Framework (P0)

- [ ] **Implement FedAvg aggregation server**
  - Client registration, model distribution, parameter aggregation
  - Support weighted averaging based on client dataset sizes
  - File: `backend/core/federation/server.py` (new)

- [ ] **Implement federated client SDK**
  - Local training loop with DP-SGD
  - Secure parameter serialization and transmission
  - File: `backend/core/federation/client.py` (new)

- [ ] **Multi-institutional collaboration simulation**
  - Simulate 3-5 hospital sites with non-IID data splits
  - Benchmark convergence vs centralized training
  - File: `experiments/federated_simulation.py` (new)

---

## GAN-Based Medical Image Enhancement (P1)

- [ ] **Implement super-resolution module** for low-dose CT
  - ESRGAN/Real-ESRGAN architecture for 4x upscaling
  - Train on paired low-dose/standard-dose CT data
  - File: `backend/core/enhancement/super_resolution.py` (new)

- [ ] **Implement denoising module** for low-dose CT/MRI
  - Noise2Void or DnCNN architecture
  - Configurable noise level estimation
  - File: `backend/core/enhancement/denoiser.py` (new)

- [ ] **Style transfer between imaging modalities**
  - CycleGAN for CT-to-MRI translation
  - Preserve anatomical structure during translation
  - File: `backend/core/enhancement/style_transfer.py` (new)

---

## Diffusion Model Improvements (P1)

- [ ] **Fine-tune LDM on medical imaging datasets**
  - Collect chest CT, brain MRI, dental CBCT datasets
  - LoRA fine-tuning with domain-specific prompts
  - File: `experiments/fine_tune_medical_ldm.py` (new)

- [ ] **Implement 3D volumetric synthesis**
  - Extend LDM to generate 3D volumes (not just 2D slices)
  - Slice-by-slice generation with 3D consistency loss
  - File: `backend/core/diffusion/volumetric_ldm.py` (new)

- [ ] **Add DICOM metadata preservation**
  - Copy DICOM headers from template to generated images
  - Support DICOM anonymization pipeline
  - File: `backend/core/data/dicom_writer.py` (new)

---

## Sign Language Enhancements (P1)

- [ ] **Expand CSL vocabulary to 500+ glosses**
  - Collect additional CSL training data
  - Implement continuous sentence support
  - File: `backend/core/vision/csl_vocabulary.py` (new)

- [ ] **Add ASL (American Sign Language) support**
  - Adapt tri-channel architecture for ASL
  - Cross-lingual transfer learning
  - File: `backend/core/vision/asl_support.py` (new)

- [ ] **Implement WebRTC peer-to-peer streaming**
  - Replace WebSocket with WebRTC for lower latency
  - Add ICE/STUN/TURN server configuration
  - File: `backend/core/streaming/webrtc_handler.py` (new)

- [ ] **Active learning from user feedback**
  - Collect user corrections on recognition results
  - Online model fine-tuning with human-in-the-loop
  - File: `backend/core/learning/active_learning.py` (new)

---

## Infrastructure & DevOps (P2)

- [ ] **Add model versioning and registry**
  - MLflow or Weights & Biases integration
  - Track model artifacts, metrics, and lineage
  - File: `backend/core/registry/model_registry.py` (new)

- [ ] **Implement rate limiting and authentication**
  - API key authentication for production
  - Rate limiting per client
  - File: `backend/core/security/auth.py` (new)

- [ ] **Add Prometheus metrics export**
  - Request latency, throughput, error rates
  - GPU utilization metrics
  - File: `backend/core/monitoring/metrics.py` (new)

- [ ] **Implement log aggregation**
  - Structured JSON logging
  - ELK stack or Loki integration
  - File: `backend/core/logging/structured_logger.py` (new)

---

## Testing & Quality (P2)

- [ ] **Achieve 90%+ test coverage**
  - Add integration tests for all API endpoints
  - Add property-based tests for data processors
  - File: `tests/test_integration.py` (new)

- [ ] **Add load testing**
  - Locust or k6 load test scripts
  - Benchmark concurrent request handling
  - File: `tests/load_test.py` (new)

- [ ] **Add model accuracy benchmarks**
  - Automated accuracy regression tests
  - Benchmark against published baselines
  - File: `tests/test_benchmarks.py` (new)

---

## Documentation (P3)

- [ ] **Add interactive API examples**
  - Jupyter notebooks with API walkthroughs
  - File: `docs/notebooks/api_walkthrough.ipynb` (new)

- [ ] **Create video tutorials**
  - Setup guide
  - Feature demonstrations
  - File: `docs/tutorials/` (new)

---

## Completed

- [x] Tri-channel Transformer fusion architecture
- [x] MediaPipe three-channel keypoint detection
- [x] Latent Diffusion Model integration
- [x] ControlNet anatomical constraints
- [x] WebSocket real-time streaming
- [x] DICOM/NIfTI format support
- [x] FastAPI backend with dual services
- [x] Vue 3 frontend with Element Plus
- [x] Docker containerization
- [x] CI/CD pipeline (GitHub Actions)
- [x] Comprehensive smoke tests (50+)
- [x] Documentation structure

---

*Last updated: 2026-05-29*
