"""
Pydantic models for the Sign Language Translation API
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime
from enum import Enum


class TranslationDirection(str, Enum):
    SIGN_TO_TEXT = "sign_to_text"
    TEXT_TO_SIGN = "text_to_sign"


class VideoFormat(str, Enum):
    MP4 = "mp4"
    BASE64_FRAMES = "base64_frames"
    KEYPOINTS = "keypoints"
    SKELETON = "skeleton"


class SignRecognitionRequest(BaseModel):
    """手语识别请求"""
    frames: List[str] = Field(..., description="Base64编码的视频帧列表")
    language: str = Field(default="csl", description="手语类型 (csl/asl/bsl)")


class SignRecognitionResponse(BaseModel):
    """手语识别响应"""
    status: str
    text: Optional[str] = None
    confidence: Optional[float] = None
    translation_id: str
    processing_time_ms: Optional[float] = None
    frames_processed: Optional[int] = None
    error: Optional[str] = None


class TextToSignRequest(BaseModel):
    """文字转手语请求"""
    text: str = Field(..., description="输入文本")
    format: VideoFormat = Field(default=VideoFormat.BASE64_FRAMES)
    fps: int = Field(default=25, ge=1, le=60)
    resolution: tuple = Field(default=(512, 512))


class KeypointData(BaseModel):
    """关键点数据"""
    hand_landmarks: Optional[List[List[float]]] = None
    face_landmarks: Optional[List[List[float]]] = None
    body_landmarks: Optional[List[List[float]]] = None
    handedness: Optional[str] = None
    score: Optional[float] = None


class SignVideoResponse(BaseModel):
    """手语视频响应"""
    status: str
    translation_id: str
    format: Optional[str] = None
    frames: Optional[List[str]] = None
    keyframes: Optional[List] = None
    fps: int = 25
    length: int = 0
    processing_time_ms: Optional[float] = None
    error: Optional[str] = None


class BidirectionalTranslationRequest(BaseModel):
    """双向翻译请求"""
    direction: TranslationDirection
    frames: Optional[List[str]] = None
    text: Optional[str] = None


class BidirectionalTranslationResponse(BaseModel):
    """双向翻译响应"""
    status: str
    direction: TranslationDirection
    translation_id: str
    recognized_text: Optional[str] = None
    confidence: Optional[float] = None
    sign_video: Optional[List[str]] = None
    format: Optional[str] = None
    processing_time_ms: Optional[float] = None
    error: Optional[str] = None


class StreamFrameData(BaseModel):
    """流式帧数据"""
    type: str = "frame"
    frame: str
    timestamp_ms: float


class StreamTranslationResult(BaseModel):
    """流式翻译结果"""
    type: str  # "interim" | "final" | "pong"
    timestamp_ms: Optional[float] = None
    hand_count: Optional[int] = None
    face_detected: Optional[bool] = None
    body_detected: Optional[bool] = None
    text: Optional[str] = None
    confidence: Optional[float] = None
    translation_id: Optional[str] = None


class ModelStatus(BaseModel):
    """模型状态"""
    recognizer_loaded: bool
    generator_loaded: bool
    device: str
    cuda_available: bool
    mediapipe_available: bool
    fusion_model_ready: bool


class SystemHealth(BaseModel):
    """系统健康状态"""
    status: str
    timestamp: datetime
    models: ModelStatus
    latency_ms: Optional[float] = None
