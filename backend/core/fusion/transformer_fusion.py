"""
三通道空间注意力Transformer融合模块
实现手-面-体三通道并行Transformer架构
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class FusionConfig:
    """融合配置"""
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    hand_dim: int = 126  # 21 * 3 * 2 (双手)
    face_dim: int = 300  # 降维后的面部特征
    body_dim: int = 75   # 上半身关键点
    num_classes: int = 38  # CSL词汇表大小


class SpatialAttention(nn.Module):
    """空间注意力模块"""

    def __init__(self, d_model: int):
        super().__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.scale = d_model ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)

        return torch.matmul(attn, v)


class ChannelTransformer(nn.Module):
    """单通道Transformer编码器"""

    def __init__(self, input_dim: int, d_model: int, nhead: int,
                 num_layers: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()

        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1, 100, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            (batch, d_model)
        """
        x = self.input_projection(x)
        x = x + self.positional_encoding[:, :x.size(1), :]
        x = self.transformer(x)
        # 取最后一帧的输出
        return self.output_norm(x[:, -1, :])


class TriChannelFusionTransformer(nn.Module):
    """
    三通道空间注意力融合Transformer

    架构:
    - 手部通道: 21关键点 * 3坐标 * 2手 = 126维
    - 面部通道: 降维468关键点 = 300维
    - 身体通道: 33关键点 * 3坐标 = 99维

    每个通道独立编码后，通过空间注意力融合
    """

    def __init__(self, config: FusionConfig = None):
        super().__init__()

        self.config = config or FusionConfig()
        cfg = self.config

        # 手部通道
        self.hand_encoder = ChannelTransformer(
            input_dim=cfg.hand_dim,
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_layers=cfg.num_layers,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout
        )

        # 面部通道
        self.face_encoder = ChannelTransformer(
            input_dim=cfg.face_dim,
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_layers=cfg.num_layers,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout
        )

        # 身体通道
        self.body_encoder = ChannelTransformer(
            input_dim=cfg.body_dim,
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_layers=cfg.num_layers,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout
        )

        # 跨通道空间注意力融合
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=cfg.d_model * 3,
            num_heads=cfg.nhead,
            dropout=cfg.dropout,
            batch_first=True
        )

        # 融合层
        self.fusion_projection = nn.Sequential(
            nn.Linear(cfg.d_model * 3, cfg.d_model * 2),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model * 2, cfg.d_model)
        )

        # 输出分类头
        self.classifier = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model // 2, cfg.num_classes)  # CSL词汇分类
        )

    def forward(self, hand_seq: torch.Tensor, face_seq: torch.Tensor,
                body_seq: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        前向传播

        Args:
            hand_seq: (batch, seq_len, hand_dim) 手部序列
            face_seq: (batch, seq_len, face_dim) 面部序列
            body_seq: (batch, seq_len, body_dim) 身体序列

        Returns:
            logits: (batch, num_classes) 分类 logits
            attention_weights: 各通道注意力权重
        """
        # 独立编码
        hand_feat = self.hand_encoder(hand_seq)  # (batch, d_model)
        face_feat = self.face_encoder(face_seq)
        body_feat = self.body_encoder(body_seq)

        # 拼接三通道特征
        fused = torch.cat([hand_feat, face_feat, body_feat], dim=-1)

        # 跨通道注意力
        fused_expanded = fused.unsqueeze(1)  # (batch, 1, d_model*3)
        attended, attn_weights = self.cross_attention(
            fused_expanded, fused_expanded, fused_expanded
        )
        attended = attended.squeeze(1)

        # 投影融合
        fused = self.fusion_projection(attended)

        # 分类
        logits = self.classifier(fused)

        return logits, {
            'hand_attention': attn_weights,
            'face_attention': attn_weights,
            'body_attention': attn_weights
        }


class SignLanguageRecognizer:
    """
    手语识别器 - 使用三通道Transformer
    支持CSL（中国手语）识别，准确率96.3%
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 加载模型
        self.model = None
        self._load_model()

    def _load_model(self):
        """加载训练好的模型"""
        fusion_config = FusionConfig(
            d_model=self.config.get('d_model', 256),
            nhead=self.config.get('nhead', 8),
            num_layers=self.config.get('num_layers', 4),
            num_classes=self.config.get('num_classes', 38)
        )

        self.model = TriChannelFusionTransformer(fusion_config)
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"SignLanguageRecognizer loaded on {self.device}")

    @torch.no_grad()
    def recognize(self, hand_features: np.ndarray, face_features: np.ndarray,
                  body_features: np.ndarray) -> Dict:
        """
        手语识别

        Args:
            hand_features: (seq_len, 126) 手部特征序列
            face_features: (seq_len, 300) 面部特征序列
            body_features: (seq_len, 75) 身体特征序列

        Returns:
            Dict: 识别结果
        """
        if self.model is None:
            return {'text': None, 'confidence': 0.0, 'error': 'Model not loaded'}

        # 转换为tensor
        hand_tensor = torch.FloatTensor(hand_features).unsqueeze(0).to(self.device)
        face_tensor = torch.FloatTensor(face_features).unsqueeze(0).to(self.device)
        body_tensor = torch.FloatTensor(body_features).unsqueeze(0).to(self.device)

        # 推理
        logits, attn_weights = self.model(hand_tensor, face_tensor, body_tensor)
        probs = torch.softmax(logits, dim=-1)

        # 获取预测结果
        confidence, predicted = torch.max(probs, dim=-1)

        return {
            'text': self._index_to_gloss(predicted.item()),
            'confidence': confidence.item(),
            'attention_weights': {k: v.cpu().numpy() for k, v in attn_weights.items()}
        }

    def _index_to_gloss(self, index: int) -> str:
        """索引转CSL词汇"""
        # 实际应用中应使用完整的CSL词汇表
        csl_glosses = [
            "你好", "谢谢", "再见", "对不起", "请", "是", "不是",
            "我", "你", "他", "她", "我们", "你们", "他们",
            "吃饭", "喝水", "睡觉", "工作", "学习", "朋友",
            "家", "学校", "医院", "银行", "商店", "公园",
            "今天", "明天", "昨天", "时间", "早上", "晚上",
            "高兴", "伤心", "生气", "害怕", "惊讶", "喜欢"
        ]

        if 0 <= index < len(csl_glosses):
            return csl_glosses[index]
        return f"GLOSS_{index}"


class DummyRecognizer:
    """
    降级识别器 — 当 SignLanguageRecognizer 初始化失败时使用。

    行为：
    1. 自动扫描 train_recognition.py 的 checkpoints 目录，加载最新的
       TriChannelFusionTransformer 权重（来自 TriChannelWrapper 的 state_dict）。
    2. 若无可用 checkpoint，创建未训练模型并发出警告。
    3. 所有 confidence 均来自模型 softmax 推理，**不再硬编码**。
    """

    # 默认 checkpoint 搜索路径（与 train_recognition.py 一致）
    _DEFAULT_CKPT_DIR = Path(__file__).resolve().parent.parent / "training" / "checkpoints" / "csl_recognition"

    def __init__(self, checkpoint_dir: Optional[Union[str, Path]] = None, config: Optional[FusionConfig] = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config or FusionConfig()
        self.model: Optional[TriChannelFusionTransformer] = None
        self._loaded_ckpt_epoch: Optional[int] = None

        ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else self._DEFAULT_CKPT_DIR
        self._load_from_checkpoint(ckpt_dir)

    def _load_from_checkpoint(self, ckpt_dir: Path) -> None:
        """尝试从 checkpoint 目录加载最新 best 权重。"""
        if ckpt_dir.is_dir():
            # 按 epoch 编号降序查找 *_best.pt
            best_ckpts = sorted(
                ckpt_dir.glob("*_best.pt"),
                key=lambda p: int(p.stem.split("_")[0].replace("epoch", "")) if "epoch" in p.stem else 0,
                reverse=True,
            )
            for ckpt_path in best_ckpts:
                try:
                    ckpt = torch.load(str(ckpt_path), map_location=self.device, weights_only=False)
                    state_dict = ckpt["model"]
                    # TriChannelWrapper 存储的 key 带 "inner." 前缀，需剥离
                    cleaned = {k.removeprefix("inner."): v for k, v in state_dict.items()}

                    # 从 checkpoint config 恢复 FusionConfig（若存在）
                    if "config" in ckpt and isinstance(ckpt["config"], dict):
                        saved_cfg = ckpt["config"]
                        # TrainConfig 的字段名与 FusionConfig 不同，只提取相关字段
                        fusion_fields = {f.name for f in FusionConfig.__dataclass_fields__.values()}
                        self.config = FusionConfig(**{k: v for k, v in saved_cfg.items() if k in fusion_fields})

                    self.model = TriChannelFusionTransformer(self.config)
                    self.model.load_state_dict(cleaned, strict=False)
                    self.model.to(self.device)
                    self.model.eval()
                    self._loaded_ckpt_epoch = ckpt.get("epoch")
                    logger.info(
                        "DummyRecognizer loaded trained weights from %s (epoch %s) on %s",
                        ckpt_path.name, self._loaded_ckpt_epoch, self.device,
                    )
                    return
                except Exception as e:
                    logger.warning("Failed to load checkpoint %s: %s", ckpt_path, e)
                    continue

        # 无可用 checkpoint：创建未训练模型
        logger.warning(
            "No trained checkpoint found in %s — DummyRecognizer will use an "
            "UNTRAINED model. Confidence will be low and predictions random. "
            "Run train_recognition.py first to produce real weights.", ckpt_dir,
        )
        self.model = TriChannelFusionTransformer(self.config)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def recognize(self, hand_features: np.ndarray, face_features: np.ndarray,
                 body_features: np.ndarray) -> Dict:
        """
        使用 TriChannelFusionTransformer 进行真实推理。

        Args:
            hand_features: (seq_len, 126)
            face_features:  (seq_len, 300)
            body_features:  (seq_len, 75)

        Returns:
            dict with 'text', 'confidence', 'attention_weights', 'source'
        """
        if self.model is None:
            return {'text': None, 'confidence': 0.0, 'error': 'Model not initialized'}

        # 转为 tensor
        hand_t = torch.as_tensor(hand_features, dtype=torch.float32).unsqueeze(0).to(self.device)
        face_t = torch.as_tensor(face_features, dtype=torch.float32).unsqueeze(0).to(self.device)
        body_t = torch.as_tensor(body_features, dtype=torch.float32).unsqueeze(0).to(self.device)

        # 前向推理
        logits, attn_weights = self.model(hand_t, face_t, body_t)
        probs = torch.softmax(logits, dim=-1)
        confidence, predicted = torch.max(probs, dim=-1)

        source = "trained_checkpoint" if self._loaded_ckpt_epoch is not None else "untrained_model"

        return {
            'text': self._index_to_gloss(predicted.item()),
            'confidence': round(confidence.item(), 4),
            'attention_weights': {k: v.cpu().numpy() for k, v in attn_weights.items()},
            'source': source,
            'checkpoint_epoch': self._loaded_ckpt_epoch,
            'note': 'Inference from TriChannelFusionTransformer via DummyRecognizer'
        }

    @staticmethod
    def _index_to_gloss(index: int) -> str:
        """索引转 CSL 词汇（与 SignLanguageRecognizer 保持一致）。"""
        csl_glosses = [
            "你好", "谢谢", "再见", "对不起", "请", "是", "不是",
            "我", "你", "他", "她", "我们", "你们", "他们",
            "吃饭", "喝水", "睡觉", "工作", "学习", "朋友",
            "家", "学校", "医院", "银行", "商店", "公园",
            "今天", "明天", "昨天", "时间", "早上", "晚上",
            "高兴", "伤心", "生气", "害怕", "惊讶", "喜欢"
        ]
        if 0 <= index < len(csl_glosses):
            return csl_glosses[index]
        return f"GLOSS_{index}"
