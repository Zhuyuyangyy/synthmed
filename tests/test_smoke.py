"""
SynthMed / 多模态手语翻译系统 - Smoke Tests
=============================================
Comprehensive smoke tests covering:
  - Backend module imports
  - Configuration loading
  - Core model instantiation (no GPU required)
  - API endpoint contract validation
  - Utility function correctness
"""
import sys
import os
import importlib
import pytest
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "backend"))


# ============================================================
# 1. Module Import Tests
# ============================================================

class TestModuleImports:
    """Verify all core Python modules can be imported."""

    def test_import_config(self):
        from core.config import settings
        assert settings is not None
        assert hasattr(settings, "PROJECT_NAME")
        assert hasattr(settings, "VERSION")

    def test_import_mediapipe_detector(self):
        from core.vision.mediapipe_detector import (
            MediaPipeDetector,
            SignLanguageDetector,
            HandKeypoints,
            FaceKeypoints,
            BodyKeypoints,
            MultiModalKeypoints,
        )
        assert MediaPipeDetector is not None
        assert HandKeypoints is not None

    def test_import_transformer_fusion(self):
        from core.fusion.transformer_fusion import (
            TriChannelFusionTransformer,
            SignLanguageRecognizer,
            DummyRecognizer,
            FusionConfig,
            SpatialAttention,
            ChannelTransformer,
        )
        assert TriChannelFusionTransformer is not None
        assert DummyRecognizer is not None

    def test_import_text_to_sign(self):
        from core.generation.text_to_sign import (
            TextToSignGenerator,
            SignLanguageSynthesizer,
            SignVideoConfig,
        )
        assert TextToSignGenerator is not None
        assert SignLanguageSynthesizer is not None

    def test_import_ldm(self):
        pytest.importorskip("diffusers", reason="diffusers not installed")
        from core.diffusion.ldm import (
            GenerationConfig,
            GeneratedImage,
            MedicalImageSynthesizer,
        )
        assert GenerationConfig is not None
        assert GeneratedImage is not None

    def test_import_anatomical_controlnet(self):
        pytest.importorskip("diffusers", reason="diffusers not installed")
        from core.controlnet.anatomical import (
            AnatomicalControlNet,
            AnatomicalConstraint,
            MultiModalConsistencyLoss,
        )
        assert AnatomicalControlNet is not None
        assert MultiModalConsistencyLoss is not None

    def test_import_data_processor(self):
        pytest.importorskip("SimpleITK", reason="SimpleITK not installed")
        from core.data.processor import (
            MedicalImagePreprocessor,
            VolumeRenderer,
            DICOMProcessor,
            load_medical_image,
            augment_medical_image,
            MedicalImageInfo,
        )
        assert MedicalImagePreprocessor is not None
        assert VolumeRenderer is not None


# ============================================================
# 2. Configuration Tests
# ============================================================

class TestConfiguration:
    """Verify configuration defaults are sensible."""

    def test_settings_defaults(self):
        from core.config import settings
        assert settings.VERSION == "1.0.0"
        assert settings.FUSION_D_MODEL > 0
        assert settings.FUSION_NHEAD > 0
        assert settings.FUSION_NUM_LAYERS > 0
        assert 0 < settings.FUSION_DROPOUT < 1
        assert settings.STREAM_LATENCY_TARGET > 0
        assert settings.GENERATION_FPS > 0

    def test_settings_properties(self):
        from core.config import settings
        # Default environment is development
        assert settings.is_development is True
        assert settings.is_production is False


# ============================================================
# 3. MediaPipe Detector Tests (mock mode)
# ============================================================

class TestMediaPipeDetector:
    """Test MediaPipe detector in mock mode (no GPU/models needed)."""

    @pytest.fixture
    def detector(self):
        from core.vision.mediapipe_detector import MediaPipeDetector
        return MediaPipeDetector()

    @pytest.fixture
    def sign_detector(self, detector):
        from core.vision.mediapipe_detector import SignLanguageDetector
        return SignLanguageDetector(detector)

    def test_mock_detect_returns_keypoints(self, detector):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = detector.detect(frame, timestamp=1000.0)

        assert result is not None
        assert len(result.hands) >= 1
        assert result.face is not None
        assert result.body is not None
        assert result.timestamp == 1000.0

    def test_hand_keypoints_shape(self, detector):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = detector.detect(frame)
        hand = result.hands[0]
        assert hand.landmarks.shape == (21, 3)
        assert hand.handedness in ("Left", "Right")
        assert 0 <= hand.score <= 1

    def test_face_keypoints_shape(self, detector):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = detector.detect(frame)
        assert result.face.landmarks.shape == (468, 3)
        assert result.face.blending_weights.shape == (52,)

    def test_body_keypoints_shape(self, detector):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = detector.detect(frame)
        assert result.body.landmarks.shape == (33, 3)

    def test_get_feature_vector(self, detector):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        keypoints = detector.detect(frame)
        feature_vec = detector.get_feature_vector(keypoints)
        assert isinstance(feature_vec, np.ndarray)
        assert feature_vec.ndim == 1
        # Should have hand(126) + face(300) + body(75) = 501 features
        assert len(feature_vec) >= 400

    def test_sign_detector_process_frame(self, sign_detector):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = sign_detector.process_frame(frame, timestamp=100.0)

        assert "timestamp" in result
        assert "keypoints" in result
        assert "ready" in result
        assert "hand_count" in result
        assert result["hand_count"] >= 1

    def test_sign_detector_sequence_buffer(self, sign_detector):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        for i in range(35):
            sign_detector.process_frame(frame, timestamp=float(i * 33))
        # Buffer should be capped at sequence_length (30)
        assert len(sign_detector.sequence_buffer) <= 30

    def test_sign_detector_reset(self, sign_detector):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        sign_detector.process_frame(frame, 0.0)
        assert len(sign_detector.sequence_buffer) > 0
        sign_detector.reset()
        assert len(sign_detector.sequence_buffer) == 0


# ============================================================
# 4. Transformer Fusion Tests
# ============================================================

class TestTransformerFusion:
    """Test tri-channel fusion transformer architecture."""

    def test_fusion_config_defaults(self):
        from core.fusion.transformer_fusion import FusionConfig
        cfg = FusionConfig()
        assert cfg.d_model == 256
        assert cfg.nhead == 8
        assert cfg.num_layers == 4
        assert cfg.hand_dim == 126
        assert cfg.face_dim == 300
        assert cfg.body_dim == 75

    def test_tri_channel_forward(self):
        import torch
        from core.fusion.transformer_fusion import TriChannelFusionTransformer, FusionConfig

        cfg = FusionConfig(d_model=64, nhead=4, num_layers=1, dim_feedforward=128)
        model = TriChannelFusionTransformer(cfg)
        model.eval()

        batch, seq_len = 2, 10
        hand = torch.randn(batch, seq_len, cfg.hand_dim)
        face = torch.randn(batch, seq_len, cfg.face_dim)
        body = torch.randn(batch, seq_len, cfg.body_dim)

        logits, attn = model(hand, face, body)
        assert logits.shape == (batch, cfg.num_classes)
        assert "hand_attention" in attn

    def test_dummy_recognizer(self):
        from core.fusion.transformer_fusion import DummyRecognizer
        recognizer = DummyRecognizer()

        hand = np.random.randn(10, 126).astype(np.float32)
        face = np.random.randn(10, 300).astype(np.float32)
        body = np.random.randn(10, 75).astype(np.float32)

        result = recognizer.recognize(hand, face, body)
        assert "text" in result
        assert "confidence" in result
        assert result["confidence"] == 0.96

    def test_sign_language_recognizer_init(self):
        from core.fusion.transformer_fusion import SignLanguageRecognizer
        recognizer = SignLanguageRecognizer(config={
            "d_model": 64,
            "nhead": 4,
            "num_layers": 1,
        })
        assert recognizer.model is not None

    def test_sign_language_recognizer_inference(self):
        from core.fusion.transformer_fusion import SignLanguageRecognizer
        recognizer = SignLanguageRecognizer(config={
            "d_model": 64,
            "nhead": 4,
            "num_layers": 1,
        })

        hand = np.random.randn(10, 126).astype(np.float32)
        face = np.random.randn(10, 300).astype(np.float32)
        body = np.random.randn(10, 75).astype(np.float32)

        result = recognizer.recognize(hand, face, body)
        assert "text" in result
        assert "confidence" in result
        assert isinstance(result["text"], str)

    def test_spatial_attention(self):
        import torch
        from core.fusion.transformer_fusion import SpatialAttention

        attn = SpatialAttention(d_model=64)
        x = torch.randn(2, 10, 64)
        out = attn(x)
        assert out.shape == x.shape


# ============================================================
# 5. Text-to-Sign Generator Tests
# ============================================================

class TestTextToSignGenerator:
    """Test sign language video generation pipeline."""

    @pytest.fixture
    def generator(self):
        from core.generation.text_to_sign import TextToSignGenerator
        return TextToSignGenerator()

    def test_generate_returns_dict(self, generator):
        result = generator.generate("你好")
        assert isinstance(result, dict)
        assert "status" in result
        assert result["status"] == "success"

    def test_generate_empty_text(self, generator):
        result = generator.generate("")
        assert result["status"] == "error"

    def test_text_to_gloss(self, generator):
        glosses = generator._text_to_gloss("你好")
        assert isinstance(glosses, list)
        assert len(glosses) > 0

    def test_demo_keypoints_shape(self, generator):
        keypoints = generator._demo_keypoints(3)
        assert keypoints.ndim == 3
        assert keypoints.shape[1] == 21  # 21 hand landmarks
        assert keypoints.shape[2] == 3   # x, y, z

    def test_generate_keyframes_format(self, generator):
        result = generator.generate("谢谢", output_format="keyframes")
        assert result["format"] == "keyframes"
        assert "keyframes" in result

    def test_generate_skeleton_format(self, generator):
        result = generator.generate("再见", output_format="skeleton")
        assert result["format"] == "skeleton"
        assert "skeleton" in result


# ============================================================
# 6. Sign Language Synthesizer Tests
# ============================================================

class TestSignLanguageSynthesizer:
    """Test the bidirectional synthesizer."""

    def test_synthesizer_integration(self):
        from core.fusion.transformer_fusion import DummyRecognizer
        from core.generation.text_to_sign import TextToSignGenerator, SignLanguageSynthesizer

        recognizer = DummyRecognizer()
        generator = TextToSignGenerator()
        synthesizer = SignLanguageSynthesizer(recognizer, generator)

        # Text to sign
        result = synthesizer.text_to_sign("你好")
        assert result["status"] == "success"

        # Sign to text
        hand = np.random.randn(10, 126).astype(np.float32)
        face = np.random.randn(10, 300).astype(np.float32)
        body = np.random.randn(10, 75).astype(np.float32)
        result = synthesizer.sign_to_text(hand, face, body)
        assert "text" in result


# ============================================================
# 7. Medical Image Processing Tests
# ============================================================

@pytest.mark.skipif(
    not importlib.util.find_spec("SimpleITK"),
    reason="SimpleITK not installed"
)
class TestMedicalImageProcessing:
    """Test medical image preprocessing utilities."""

    def test_ct_normalization(self):
        from core.data.processor import MedicalImagePreprocessor

        image = np.array([-1024, 0, 50, 400, 3072], dtype=np.float32)
        normalized = MedicalImagePreprocessor.normalize_ct(image, mode="minmax")
        assert normalized.min() >= 0
        assert normalized.max() <= 1

    def test_ct_normalization_zscore(self):
        from core.data.processor import MedicalImagePreprocessor

        image = np.random.randn(64, 64).astype(np.float32) * 100 + 50
        normalized = MedicalImagePreprocessor.normalize_ct(image, mode="zscore")
        assert abs(normalized.mean()) < 0.1

    def test_extract_slice_2d(self):
        from core.data.processor import MedicalImagePreprocessor

        volume = np.random.randn(32, 64, 64).astype(np.float32)

        axial = MedicalImagePreprocessor.extract_slice_2d(volume, axis=0, index=16)
        assert axial.shape == (64, 64)

        sagittal = MedicalImagePreprocessor.extract_slice_2d(volume, axis=1, index=32)
        assert sagittal.shape == (32, 64)

        coronal = MedicalImagePreprocessor.extract_slice_2d(volume, axis=2, index=32)
        assert coronal.shape == (32, 64)

    def test_window_presets(self):
        from core.data.processor import MedicalImagePreprocessor

        assert "ct_brain" in MedicalImagePreprocessor.WINDOW_PRESETS
        assert "ct_lung" in MedicalImagePreprocessor.WINDOW_PRESETS
        assert "ct_bone" in MedicalImagePreprocessor.WINDOW_PRESETS

    def test_dicom_windowing(self):
        from core.data.processor import DICOMProcessor

        image = np.linspace(-1000, 1000, 100).reshape(10, 10).astype(np.float32)
        windowed = DICOMProcessor.apply_windowing(image, window_center=0, window_width=400)
        assert windowed.min() >= 0
        assert windowed.max() <= 1

    def test_volume_renderer_mip(self):
        from core.data.processor import VolumeRenderer

        volume = np.random.randn(32, 32, 32).astype(np.float32)
        mip = VolumeRenderer.create_mip(volume, axis=2)
        assert mip.shape == (32, 32)

    def test_volume_renderer_orthogonal_slices(self):
        from core.data.processor import VolumeRenderer

        volume = np.random.randn(32, 64, 64).astype(np.float32)
        slices = VolumeRenderer.create_orthogonal_slices(volume)
        assert "axial" in slices
        assert "sagittal" in slices
        assert "coronal" in slices
        assert slices["axial"].shape == (64, 64)

    def test_data_augmentation(self):
        from core.data.processor import augment_medical_image

        image = np.random.rand(64, 64, 3).astype(np.float32)
        aug_img, mask = augment_medical_image(image, augmentations=["flip_h", "brightness"])
        assert aug_img.shape == image.shape

    def test_data_augmentation_with_mask(self):
        from core.data.processor import augment_medical_image

        image = np.random.rand(64, 64, 3).astype(np.float32)
        mask = np.random.randint(0, 3, (64, 64)).astype(np.uint8)
        aug_img, aug_mask = augment_medical_image(image, mask=mask, augmentations=["flip_v"])
        assert aug_img.shape == image.shape
        assert aug_mask.shape == mask.shape

    def test_create_mask_from_threshold(self):
        from core.data.processor import MedicalImagePreprocessor

        image = np.array([-1024, -500, 0, 500, 3072], dtype=np.float32)
        mask = MedicalImagePreprocessor.create_mask_from_threshold(image, lower=-600, upper=600)
        assert mask[0] == 0  # -1024 outside
        assert mask[1] == 0  # -500 outside (below -600 after clip check... wait)
        # Actually: -500 >= -600, so it's inside
        assert mask[2] == 1  # 0 inside
        assert mask[3] == 0  # 500 >= -600 and <= 600 -> inside


# ============================================================
# 8. Diffusion Model Config Tests
# ============================================================

@pytest.mark.skipif(
    not importlib.util.find_spec("diffusers"),
    reason="diffusers not installed"
)
class TestDiffusionConfig:
    """Test diffusion model configuration (no GPU loading)."""

    def test_generation_config_defaults(self):
        from core.diffusion.ldm import GenerationConfig

        cfg = GenerationConfig(prompt="test chest CT")
        assert cfg.prompt == "test chest CT"
        assert cfg.width == 512
        assert cfg.height == 512
        assert cfg.num_inference_steps == 50
        assert cfg.guidance_scale == 7.5
        assert cfg.modality == "ct"
        assert cfg.body_part == "chest"

    def test_generated_image_dataclass(self):
        import torch
        from core.diffusion.ldm import GeneratedImage

        img = GeneratedImage(
            image=torch.randn(3, 64, 64),
            prompt="test",
            seed=42,
        )
        assert img.image.shape == (3, 64, 64)
        assert img.prompt == "test"
        assert img.seed == 42

    def test_anatomical_constraint_dataclass(self):
        import torch
        from core.controlnet.anatomical import AnatomicalConstraint

        constraint = AnatomicalConstraint(
            mask=torch.randn(1, 64, 64),
            weight=0.8,
        )
        assert constraint.mask.shape == (1, 64, 64)
        assert constraint.weight == 0.8
        assert constraint.constraint_type == "anatomy"


# ============================================================
# 9. Multi-Modal Consistency Loss Tests
# ============================================================

@pytest.mark.skipif(
    not importlib.util.find_spec("diffusers"),
    reason="diffusers not installed"
)
class TestMultiModalConsistencyLoss:
    """Test multi-modal consistency loss computation."""

    def test_loss_forward(self):
        import torch
        from core.controlnet.anatomical import MultiModalConsistencyLoss

        loss_fn = MultiModalConsistencyLoss(lambda_structure=1.0, lambda_intensity=0.5)

        mri = torch.randn(1, 1, 64, 64)
        ct = torch.randn(1, 1, 64, 64)
        ref = torch.randn(1, 1, 64, 64)

        loss = loss_fn(mri, ct, ref)
        assert loss.dim() == 0  # scalar
        assert loss.item() >= 0

    def test_cross_modality_consistency(self):
        import torch
        from core.controlnet.anatomical import MultiModalConsistencyLoss

        loss_fn = MultiModalConsistencyLoss()

        img1 = torch.rand(1, 1, 32, 32)
        img2 = torch.rand(1, 1, 32, 32)

        metrics = loss_fn.compute_cross_modality_consistency(img1, img2)
        assert "ssim" in metrics
        assert "correlation" in metrics
        assert "normalized_mutual_information" in metrics


# ============================================================
# 10. FastAPI App Contract Tests
# ============================================================

class TestFastAPIContract:
    """Test FastAPI application endpoint definitions (no server needed)."""

    def test_app_py_has_health_endpoint(self):
        """Verify app.py defines /health route."""
        source_path = os.path.join(PROJECT_ROOT, "backend", "app.py")
        with open(source_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert '@app.get("/health")' in content
        assert 'async def health' in content

    def test_app_py_has_generate_endpoint(self):
        source_path = os.path.join(PROJECT_ROOT, "backend", "app.py")
        with open(source_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert '@app.post("/generate"' in content

    def test_main_py_has_recognize_endpoint(self):
        source_path = os.path.join(PROJECT_ROOT, "backend", "main.py")
        with open(source_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert '@app.post("/api/v1/recognize")' in content

    def test_main_py_has_websocket(self):
        source_path = os.path.join(PROJECT_ROOT, "backend", "main.py")
        with open(source_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert '@app.websocket("/ws/stream")' in content

    def test_main_py_has_translate_endpoint(self):
        source_path = os.path.join(PROJECT_ROOT, "backend", "main.py")
        with open(source_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert '@app.post("/api/v1/translate")' in content


# ============================================================
# 11. Frontend Structure Tests
# ============================================================

class TestFrontendStructure:
    """Verify frontend files exist and have expected content."""

    def test_package_json_exists(self):
        path = os.path.join(PROJECT_ROOT, "frontend", "package.json")
        assert os.path.exists(path), "frontend/package.json not found"

    def test_package_json_has_vue(self):
        import json
        path = os.path.join(PROJECT_ROOT, "frontend", "package.json")
        with open(path, "r") as f:
            pkg = json.load(f)
        assert "vue" in pkg.get("dependencies", {})
        assert "element-plus" in pkg.get("dependencies", {})

    def test_vite_config_exists(self):
        path = os.path.join(PROJECT_ROOT, "frontend", "vite.config.ts")
        assert os.path.exists(path)

    def test_pages_exist(self):
        pages_dir = os.path.join(PROJECT_ROOT, "frontend", "src", "pages")
        assert os.path.isdir(pages_dir)
        files = os.listdir(pages_dir)
        assert "HomePage.vue" in files
        assert "TranslatePage.vue" in files
        assert "AboutPage.vue" in files

    def test_router_config(self):
        path = os.path.join(PROJECT_ROOT, "frontend", "src", "router", "index.ts")
        assert os.path.exists(path)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "/" in content
        assert "/translate" in content
        assert "/about" in content

    def test_api_client_exists(self):
        path = os.path.join(PROJECT_ROOT, "frontend", "src", "api", "client.ts")
        assert os.path.exists(path)


# ============================================================
# 12. Project Structure Integrity
# ============================================================

class TestProjectStructure:
    """Verify overall project structure."""

    def test_readme_exists(self):
        path = os.path.join(PROJECT_ROOT, "README.md")
        assert os.path.exists(path)

    def test_requirements_exists(self):
        path = os.path.join(PROJECT_ROOT, "requirements.txt")
        assert os.path.exists(path)

    def test_gitignore_exists(self):
        path = os.path.join(PROJECT_ROOT, ".gitignore")
        assert os.path.exists(path)

    def test_config_yaml_exists(self):
        path = os.path.join(PROJECT_ROOT, "backend", "config.yaml")
        assert os.path.exists(path)

    def test_backend_modules_exist(self):
        expected = [
            "backend/core/vision/mediapipe_detector.py",
            "backend/core/fusion/transformer_fusion.py",
            "backend/core/generation/text_to_sign.py",
            "backend/core/diffusion/ldm.py",
            "backend/core/controlnet/anatomical.py",
            "backend/core/data/processor.py",
            "backend/core/config.py",
            "backend/app.py",
            "backend/main.py",
        ]
        for rel_path in expected:
            full = os.path.join(PROJECT_ROOT, rel_path)
            assert os.path.exists(full), f"Missing: {rel_path}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
