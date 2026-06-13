# SynthMed -- SCI Q2-Level Code Review Report (Round 2)

> Reviewer Perspective: Q2-level SCI journal reviewer (e.g., Computer Methods and Programs in Biomedicine, Medical Image Analysis, IEEE JBHI)
> Date: 2026-05-29
> Codebase: `D:/ZYY Project/synthmed/` (backend Python core, ~2500 LOC)
> Previous Review: `SCI_REVIEW_Q2.md` (Round 1)

---

## Executive Summary

Round 1 identified 1 critical bug, 2 high-severity issues, and 7 lower-priority defects. This Round 2 review verifies the status of each fix. **Only 1 of 10 issues has been addressed** -- the classifier output dimension bug (Issue #1). The face feature dimension mismatch (Issue #2) and the absence of a training pipeline (Issue #3) remain unaddressed. All lower-priority issues persist unchanged.

**Verdict**: The codebase is **not submission-ready**. The Round 1 overall score of 5.9/10 (Borderline Q2) was contingent on completing the recommended fixes. With 9 of 10 issues unresolved, the revised score reflects the current state.

---

## Fix Verification Matrix

| # | Issue | Severity | Round 1 Status | Round 2 Status | Notes |
|---|-------|----------|----------------|----------------|-------|
| 1 | Classifier `output_dim=1` | CRITICAL | FIXED (this session) | **CONFIRMED FIXED** | `nn.Linear(cfg.d_model // 2, cfg.num_classes)` at line 160. `FusionConfig.num_classes=38` at line 26. `SignLanguageRecognizer._load_model()` passes `num_classes` at line 225. Fix is correct and complete. |
| 2 | Face feature dim mismatch (630 vs 300) | HIGH | Open | **STILL OPEN** | `get_feature_vector()` still selects indices `range(33,133)` + `range(133,173)` + `range(193,263)` = 210 points x 3 = 630 dims. `FusionConfig.face_dim = 300` unchanged. Pipeline will crash with RuntimeError on any real face input. |
| 3 | No training pipeline | HIGH | Open | **STILL OPEN** | `fine_tune()` exists for LDM (lines 291-382) but is incomplete: no validation loop, no checkpointing strategy, no DP-SGD integration, no gradient clipping. No training script for the sign language recognition model. `DummyRecognizer` still returns hardcoded `confidence=0.96` (line 298). |
| 4 | `ndimage` undefined in `processor.py` | Medium | Open | **STILL OPEN** | Line 265: `ndimage.sum(mask, labeled, range(1, n + 1))` -- `ndimage` is never imported. Will raise `NameError` at runtime. |
| 5 | CORS `allow_origins=["*"]` + `allow_credentials=True` | Medium | Open | **STILL OPEN** | Both `main.py:125-131` and `app.py:46-50` retain this insecure configuration. Violates CORS spec (credentials + wildcard origin). |
| 6 | Two separate FastAPI apps | Low | Open | **STILL OPEN** | `main.py` (port 8000) and `app.py` (port 8015) remain separate with duplicated middleware, CORS, and health check boilerplate. |
| 7 | Duplicate dict key `"谢"` | Low | Open | **STILL OPEN** | `text_to_sign.py:103`: `"谢": "谢谢"` appears twice. Second entry silently overwrites first. |
| 8 | `__import__('datetime')` anti-pattern | Low | Open | **STILL OPEN** | `main.py:180`: `__import__('datetime').datetime.now()` -- should use `from datetime import datetime` (already imported at line 11). |
| 9 | `from cv2 import Canny, cv2` | Low | Open | **STILL OPEN** | `anatomical.py:174`: Incorrect import syntax. `cv2` is a module, not an exportable name from itself. |
| 10 | Hardcoded `gpt-3.5-turbo` | Low | Open | **STILL OPEN** | `config.py:63`: `LLM_MODEL: str = "gpt-3.5-turbo"` with `LLM_API_KEY: Optional[str] = None`. Security concern if env vars leak. |

---

## Revised 7-Dimension Scoring

| # | Dimension | Round 1 | Round 2 | Change | Rationale |
|---|-----------|:-------:|:-------:|:------:|-----------|
| 1 | **Novelty & Innovation** | 7.5 | 7.5 | -- | No architectural changes. Tri-channel fusion concept remains novel. |
| 2 | **Technical Rigor** | 4.0 | 4.5 | +0.5 | Classifier dimension bug is confirmed fixed. However, face dimension mismatch (630 vs 300) is still a runtime crash, and no training pipeline exists. Score reflects one critical fix but two showstoppers remain. |
| 3 | **Reproducibility** | 5.5 | 5.5 | -- | No training scripts, no dataset references, no pretrained weights added. Test suite unchanged. |
| 4 | **Code Quality** | 6.5 | 6.5 | -- | All lower-priority issues (ndimage, CORS, dual apps, duplicate key, anti-patterns) remain. No refactoring performed. |
| 5 | **Clinical/Medical Relevance** | 6.0 | 6.0 | -- | No clinical validation, IRB discussion, or radiologist evaluation added. |
| 6 | **Privacy & Ethics** | 5.0 | 5.0 | -- | `fine_tune()` still lacks DP-SGD integration. `opacus` is in requirements.txt but unused. Federated learning still test stubs only. |
| 7 | **Writing & Presentation** | 6.5 | 6.5 | -- | No documentation changes. Mixed Chinese/English persists. |

**Revised Overall Weighted Score: 5.9 / 10** -- Unchanged. The classifier fix is necessary but insufficient; the face dimension mismatch alone prevents the pipeline from running.

---

## Critical Path to Submission

The following issues **must** be resolved before any journal submission. They are ordered by blocking severity.

### Phase 1: Make the Pipeline Runnable (Blocks all testing)

**1. Fix face feature dimension mismatch (Issue #2)**

Current state in `mediapipe_detector.py:242-248`:
```python
face_key_indices = list(range(33, 133))   # 100 points
face_key_indices.extend(range(133, 173))  # 40 points
face_key_indices.extend(range(193, 263))  # 70 points
# Total: 210 points x 3 = 630 dimensions
```

But `FusionConfig.face_dim = 300` and `ChannelTransformer(input_dim=300, ...)` expects 300.

Recommended fix -- Option C (learned projection, cleanest):
```python
# In FusionConfig:
face_dim: int = 630  # match actual extraction

# In TriChannelFusionTransformer.__init__():
self.face_projection = nn.Linear(630, 300)  # or keep 630 throughout
```

Or Option A (reduce extraction to 100 points = 300 dims):
```python
face_key_indices = list(range(33, 133))  # exactly 100 points x 3 = 300
```

**2. Fix `ndimage` import in `processor.py` (Issue #4)**

Add at top of file:
```python
from scipy import ndimage
```

Or replace `ndimage.sum()` with a numpy/scipy alternative that is already imported.

### Phase 2: Substantiate Claims (Blocks credibility)

**3. Add training script for sign language recognition**

Create `backend/training/train_recognition.py` with:
- CSL dataset loading (CSL-Daily or custom)
- Train/val/test split with proper randomization
- Training loop with logging
- Evaluation metrics (accuracy, F1, confusion matrix)
- Checkpoint saving/loading
- Hyperparameter config

**4. Integrate DP-SGD into LDM fine-tune (Issue #3 extension)**

The `opacus` library is already in `requirements.txt` but unused. Add:
```python
from opacus import PrivacyEngine
privacy_engine = PrivacyEngine()
self.unet, optimizer, dataloader = privacy_engine.make_private_with_epsilon(
    module=self.unet,
    optimizer=optimizer,
    data_loader=dataloader,
    target_epsilon=8.0,
    target_delta=1e-5,
    max_grad_norm=1.0,
)
```

### Phase 3: Code Quality (Polish for review)

**5. Fix remaining low-priority issues** (Issues #5-#10):
- Merge `main.py` and `app.py` into a single FastAPI app with routers
- Fix CORS to use specific origins
- Fix duplicate dict key in `text_to_sign.py:103`
- Replace `__import__('datetime')` with proper import
- Fix `from cv2 import Canny, cv2` syntax
- Remove hardcoded `gpt-3.5-turbo` default or document as example-only

**6. Standardize documentation**
- Choose English for all code comments and docstrings
- Add `requirements-lock.txt` or `poetry.lock`
- Add API request/response examples to docs

---

## What Would Change the Score

| Action | Score Impact |
|--------|--------------|
| Fix Issue #2 (face dim) | Technical Rigor: 4.5 -> 5.5 |
| Add training script with real metrics | Technical Rigor: 5.5 -> 7.0, Reproducibility: 5.5 -> 7.0 |
| Integrate DP-SGD with epsilon guarantee | Privacy & Ethics: 5.0 -> 7.0 |
| Fix all low-priority issues | Code Quality: 6.5 -> 7.5 |
| **All of the above** | **Overall: 5.9 -> 7.2** (solid Q2) |

---

## Summary for Authors

**Round 1 identified 10 issues. Round 2 confirms only 1 has been fixed.** The classifier output dimension bug (Issue #1) is correctly resolved -- the model now outputs `num_classes=38` logits instead of a single scalar. This was the most visible defect and its fix is appreciated.

However, the face feature dimension mismatch (Issue #2) remains a **runtime crash** that prevents the sign language recognition pipeline from executing on any real input. This, combined with the absence of a training pipeline (Issue #3), means the claimed "96.3% accuracy" is still unsubstantiated.

**Bottom line**: Fix Issue #2 and Issue #4 to make the pipeline runnable, then add a training script with real evaluation. Until then, the codebase cannot support the claims made in a manuscript.

---

*This review was generated by automated code analysis on 2026-05-29. All scores reflect the current state of the codebase after Round 1 fixes.*
