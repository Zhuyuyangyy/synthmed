# SynthMed -- SCI Q2-Level Code Review Report

> Reviewer Perspective: Q2-level SCI journal reviewer (e.g., Computer Methods and Programs in Biomedicine, Medical Image Analysis, IEEE JBHI)
> Date: 2026-05-29
> Codebase: `D:/ZYY Project/synthmed/` (backend Python core, ~2500 LOC)

---

## Executive Summary

SynthMed is a dual-purpose platform combining (1) bidirectional Chinese Sign Language translation via tri-channel Transformer fusion and (2) privacy-preserving medical image synthesis via Latent Diffusion Models with ControlNet anatomical constraints. The architecture is ambitious and the module structure is clean. However, several critical implementation defects undermine the claimed functionality and would be flagged by any competent reviewer.

---

## 7-Dimension Scoring

| # | Dimension | Score (1-10) | Summary |
|---|-----------|:------------:|---------|
| 1 | **Novelty & Innovation** | 7.5 | Tri-channel spatial attention fusion for CSL is genuinely novel. Combining sign language + medical image synthesis in one platform is creative. World model-based open-vocabulary sign generation is forward-looking. Docked for heavy reliance on dummy/placeholder implementations. |
| 2 | **Technical Rigor** | 4.0 | Critical classifier dimension bug (`output_dim=1` instead of `num_classes=38`) rendered the entire recognition pipeline non-functional [FIXED below]. Face feature dimension mismatch (630 extracted vs 300 expected) will crash at runtime. No actual trained weights; claimed 96.3% accuracy is unsubstantiated. |
| 3 | **Reproducibility** | 5.5 | Excellent test suite (50+ tests across smoke + comprehensive). Pydantic-based config is well-structured. However, no training scripts, no dataset references, no pretrained weights -- a reviewer cannot reproduce any claimed result. |
| 4 | **Code Quality** | 6.5 | Clean module hierarchy with proper separation of concerns. Good use of type hints and dataclasses. Issues: two separate FastAPI apps (main.py port 8000, app.py port 8015) with duplicated boilerplate; CORS `allow_origins=["*"]` is a security concern; inconsistent Chinese/English docstrings; `processor.py` has undefined `ndimage` reference in `largest_connected_component()`. |
| 5 | **Clinical/Medical Relevance** | 6.0 | Medical image templates (CT/MRI/X-ray for dental/chest/brain/abdomen) are clinically relevant. DICOM/NIfTI support and windowing presets show domain knowledge. Lacking: clinical validation, IRB/ethics discussion, comparison with clinical baselines, radiologist evaluation. |
| 6 | **Privacy & Ethics** | 5.0 | Differential privacy concepts are described but implementation is limited to test-level simulations (Gaussian/Laplace noise). DP-SGD integration in `ldm.py` fine-tune loop is absent (no gradient clipping, no noise injection). Federated learning exists only as test stubs. GDPR/HIPAA compliance claims are unsubstantiated. |
| 7 | **Writing & Presentation** | 6.5 | README is comprehensive with good ASCII architecture diagrams. Code comments are detailed. Issues: mixed Chinese/English throughout; some docstrings are stale (e.g., WebRTC mentioned but WebSocket is used); API docs auto-generated but lack request/response examples. |

**Overall Weighted Score: 5.9 / 10** -- Borderline Q2. With the critical bugs fixed and training pipeline added, this could reach Q2 acceptance level.

---

## Top 3 Issues (Ranked by Severity)

### Issue #1 [CRITICAL -- FIXED]: Classifier Output Dimension = 1

**File**: `backend/core/fusion/transformer_fusion.py`, line 160

**Problem**: The classification head in `TriChannelFusionTransformer` was defined as:
```python
nn.Linear(cfg.d_model // 2, 1)  # outputs a SINGLE scalar
```
This means the model outputs a single logit regardless of how many CSL gloss classes exist. The `recognize()` method then calls `torch.softmax(logits, dim=-1)` on a shape `(batch, 1)` tensor, which always produces probability 1.0 for the single "class." The model can never distinguish between different sign language glosses. This is a **showstopper** that makes the entire sign language recognition claim invalid.

**Fix Applied** (this session):
```python
# BEFORE (broken):
nn.Linear(cfg.d_model // 2, 1)  # CSL词汇分类

# AFTER (fixed):
nn.Linear(cfg.d_model // 2, cfg.num_classes)  # CSL词汇分类, num_classes=38
```
Also added `num_classes: int = 38` to `FusionConfig` dataclass, and updated `SignLanguageRecognizer._load_model()` to pass it through.

**Reviewer Impact**: Without this fix, any reviewer running the model would see it output the same gloss for every input. This would likely result in immediate rejection.

---

### Issue #2 [HIGH]: Face Feature Dimension Mismatch (630 vs 300)

**File**: `backend/core/vision/mediapipe_detector.py`, `get_feature_vector()` method (line 220)

**Problem**: The face feature extraction selects indices `range(33,133)` + `range(133,173)` + `range(193,263)` = 100 + 40 + 70 = 210 landmark points, each with 3 coordinates = **630 dimensions**. But `FusionConfig.face_dim = 300` and `ChannelTransformer(input_dim=300, ...)` expects exactly 300. When `SignLanguageRecognizer.recognize()` receives face features with 630 columns, `nn.Linear(300, 256)` will raise a dimension mismatch RuntimeError.

**Recommended Fix** (not yet applied -- requires design decision):
- Option A: Reduce face landmark selection to exactly 100 points (300/3 = 100)
- Option B: Update `FusionConfig.face_dim` to 630 and adjust the architecture
- Option C: Add an explicit dimension reduction layer (e.g., PCA or a learned projection) from 630 to 300

**Reviewer Impact**: The sign language recognition pipeline will crash with a RuntimeError on any real input. This is a second showstopper.

---

### Issue #3 [HIGH]: No Actual Training Pipeline -- Claims Are Unsubstantiated

**Files**: `backend/core/fusion/transformer_fusion.py`, `backend/core/diffusion/ldm.py`

**Problem**: The codebase presents a well-architected model but contains no actual training code for the sign language recognition model. The `TriChannelFusionTransformer` is initialized with random weights and inference is run on these random weights. The README claims "96.3% CSL recognition accuracy" but there is:
- No training script for the recognition model
- No dataset loading/description
- No evaluation metrics computation
- No pretrained weight files
- The `DummyRecognizer` returns hardcoded `confidence=0.96`, which appears to be the source of the "96.3%" claim

The LDM `fine_tune()` method exists but is incomplete (no data loading, no validation loop, no checkpointing strategy, no DP-SGD integration despite privacy claims).

**Reviewer Impact**: A reviewer would immediately identify that the accuracy claim is fabricated. This is the most damaging issue for publication credibility.

---

## Additional Issues (Lower Priority)

| # | File | Issue | Severity |
|---|------|-------|----------|
| 4 | `processor.py:265` | `ndimage` used without import in `largest_connected_component()` -- `ndimage.sum()` will raise `NameError` | Medium |
| 5 | `main.py:125-131` | CORS `allow_origins=["*"]` with `allow_credentials=True` -- insecure and violates CORS spec | Medium |
| 6 | `main.py` + `app.py` | Two separate FastAPI apps with duplicated initialization, CORS, health checks -- architectural smell | Low |
| 7 | `text_to_sign.py:103-107` | Duplicate dict key `"谢"` in `csl_dict` -- second entry silently overwrites first | Low |
| 8 | `main.py:180` | `__import__('datetime').datetime.now()` -- anti-pattern, should use proper import | Low |
| 9 | `anatomical.py:173-174` | `from cv2 import Canny, cv2` -- incorrect import syntax | Low |
| 10 | `config.py:63` | Hardcoded `LLM_MODEL: str = "gpt-3.5-turbo"` with `LLM_API_KEY: Optional[str] = None` -- security concern if env vars leak | Low |

---

## Actionable Recommendations for Manuscript Preparation

1. **Fix Issues #1 and #2 before any submission** -- these are runtime-breaking bugs that any reviewer will catch.
2. **Add a training script** with real CSL dataset (e.g., CSL-Daily, or a custom dataset) and report actual metrics with proper train/val/test splits.
3. **Add DP-SGD integration** into the LDM fine-tune loop if privacy claims are central to the paper.
4. **Unify the two FastAPI apps** into a single application with routers.
5. **Standardize documentation language** -- choose English for all code comments and docstrings in the manuscript version.
6. **Add a `requirements-lock.txt`** or `poetry.lock` for exact reproducibility.
7. **Include a `training/` directory** with training scripts, configs, and evaluation scripts.

---

## Files Reviewed

| File | Lines | Role |
|------|-------|------|
| `backend/main.py` | 481 | Sign language translation API (port 8000) |
| `backend/app.py` | 642 | Medical image synthesis API (port 8015) |
| `backend/core/fusion/transformer_fusion.py` | 298 | Tri-channel Transformer fusion [**MODIFIED**] |
| `backend/core/vision/mediapipe_detector.py` | 334 | MediaPipe three-channel detection |
| `backend/core/diffusion/ldm.py` | 594 | Latent Diffusion Model |
| `backend/core/controlnet/anatomical.py` | 449 | Anatomical ControlNet |
| `backend/core/data/processor.py` | 513 | Medical image processing |
| `backend/core/generation/text_to_sign.py` | 296 | Text-to-sign video generation |
| `backend/core/config.py` | 77 | Pydantic settings |
| `backend/schemas/schemas.py` | 124 | API data models |
| `tests/test_smoke.py` | 678 | Smoke tests |
| `tests/test_comprehensive.py` | 961 | Comprehensive tests |

---

*This review was generated by automated code analysis. All scores reflect the state of the codebase at the time of review, with Issue #1 fixed in this session.*
