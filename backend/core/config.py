"""
多模态手语翻译系统 - 核心配置
"""
from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    # 项目信息
    PROJECT_NAME: str = "多模态手语翻译系统"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"
    DEBUG: bool = True
    ENVIRONMENT: str = "development"

    # 服务器配置
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # CORS
    CORS_ORIGINS: list = ["*"]

    # MediaPipe配置
    MEDIAPIPE_MAXHands: int = 2
    MEDIAPIPE_MAXFaces: int = 1
    MEDIAPIPE_STATIC_IMAGE_MODE: bool = False
    MEDIAPIPE_MIN_DETECTION_CONFIDENCE: float = 0.5
    MEDIAPIPE_MIN_TRACKING_CONFIDENCE: float = 0.5

    # 模型配置
    POSE_DETECTION_ENABLED: bool = True
    FACE_DETECTION_ENABLED: bool = True
    HAND_DETECTION_ENABLED: bool = True

    # 三通道Transformer融合配置
    FUSION_D_MODEL: int = 256
    FUSION_NHEAD: int = 8
    FUSION_NUM_LAYERS: int = 4
    FUSION_DIM_FEEDFORWARD: int = 1024
    FUSION_DROPOUT: float = 0.1

    # 手语识别配置
    RECOGNITION_BATCH_SIZE: int = 8
    RECOGNITION_SEQUENCE_LENGTH: int = 30  # 帧数
    CSL_DATASET_PATH: Optional[str] = None

    # 手语视频生成配置
    GENERATION_MODEL: str = "stable-diffusion"
    GENERATION_FPS: int = 25
    GENERATION_VIDEO_LENGTH: int = 50  # 帧
    GENERATION_RESOLUTION: tuple = (512, 512)

    # WebRTC配置
    WEBRTC_ENABLED: bool = True
    WEBRTC_PORT: int = 8080

    # 流式处理配置
    STREAM_BUFFER_SIZE: int = 30  # 帧缓冲
    STREAM_LATENCY_TARGET: float = 1.5  # 秒

    # LLM配置（用于语言生成）
    LLM_MODEL: str = "gpt-3.5-turbo"
    LLM_API_KEY: Optional[str] = None
    LLM_API_BASE: str = "https://api.openai.com/v1"

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"

    @property
    def is_development(self) -> bool:
        return self.ENVIRONMENT == "development"


settings = Settings()
