"""
连接 TrainPipeline 到 TriChannelFusionTransformer 的训练脚本。
使用模拟 CSL 数据（38 个类别）验证训练管线，运行 5 个 epoch。

用法:
    python -m backend.core.training.train_recognition
"""
import json
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# 项目内部导入
# ---------------------------------------------------------------------------
from backend.core.fusion.transformer_fusion import (
    FusionConfig,
    TriChannelFusionTransformer,
    SignLanguageRecognizer,
    DummyRecognizer,
)
from backend.core.training.train_pipeline import TrainConfig, TrainPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CSL 词汇表（38 类）
# ---------------------------------------------------------------------------
CSL_GLOSSES = [
    "你好", "谢谢", "再见", "对不起", "请", "是", "不是",
    "我", "你", "他", "她", "我们", "你们", "他们",
    "吃饭", "喝水", "睡觉", "工作", "学习", "朋友",
    "家", "学校", "医院", "银行", "商店", "公园",
    "今天", "明天", "昨天", "时间", "早上", "晚上",
    "高兴", "伤心", "生气", "害怕", "惊讶", "喜欢",
]
NUM_CLASSES = len(CSL_GLOSSES)  # 38

# 特征维度（与 FusionConfig 一致）
HAND_DIM = 126   # 21 * 3 * 2
FACE_DIM = 300   # 降维面部特征
BODY_DIM = 75    # 上半身关键点


# ===========================================================================
# 1. 模拟 CSL 数据集
# ===========================================================================
class SimulatedCSLDataset(Dataset):
    """
    生成模拟的手语识别数据。
    每个样本由三通道时序特征 + 类别标签组成。
    不同类别的数据使用不同的随机种子偏移，使模型可以学到区分信号。
    """

    def __init__(self, num_samples: int, seq_len: int = 16, num_classes: int = NUM_CLASSES,
                 seed: int = 42):
        super().__init__()
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.num_classes = num_classes

        rng = np.random.RandomState(seed)
        self.labels = rng.randint(0, num_classes, size=num_samples)

        # 为每个类别生成一个独特的"签名"偏移，使模型可以区分
        self.class_signatures = rng.randn(num_classes, HAND_DIM + FACE_DIM + BODY_DIM).astype(np.float32) * 0.5

        # 预生成所有数据
        self.hand_data = np.zeros((num_samples, seq_len, HAND_DIM), dtype=np.float32)
        self.face_data = np.zeros((num_samples, seq_len, FACE_DIM), dtype=np.float32)
        self.body_data = np.zeros((num_samples, seq_len, BODY_DIM), dtype=np.float32)

        for i in range(num_samples):
            label = self.labels[i]
            sig = self.class_signatures[label]
            noise_h = rng.randn(seq_len, HAND_DIM).astype(np.float32) * 0.1
            noise_f = rng.randn(seq_len, FACE_DIM).astype(np.float32) * 0.1
            noise_b = rng.randn(seq_len, BODY_DIM).astype(np.float32) * 0.1
            self.hand_data[i] = sig[:HAND_DIM] + noise_h
            self.face_data[i] = sig[HAND_DIM:HAND_DIM + FACE_DIM] + noise_f
            self.body_data[i] = sig[HAND_DIM + FACE_DIM:] + noise_b

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        返回 (packed_input, label)。
        packed_input 将三通道拼接为一个 tensor，由 WrapperModel 拆分。
        """
        hand = torch.from_numpy(self.hand_data[idx])   # (seq_len, 126)
        face = torch.from_numpy(self.face_data[idx])    # (seq_len, 300)
        body = torch.from_numpy(self.body_data[idx])    # (seq_len, 75)
        packed = torch.cat([hand, face, body], dim=-1)   # (seq_len, 501)
        return packed, int(self.labels[idx])


# ===========================================================================
# 2. 模型包装器 —— 让 TrainPipeline 的单一 inputs 接口适配三通道模型
# ===========================================================================
class TriChannelWrapper(nn.Module):
    """
    包装 TriChannelFusionTransformer，接收拼接的单一 tensor，
    内部拆分为 hand / face / body 三路。
    """

    def __init__(self, config: FusionConfig):
        super().__init__()
        self.inner = TriChannelFusionTransformer(config)
        self.hand_dim = config.hand_dim
        self.face_dim = config.face_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hand_dim + face_dim + body_dim)
        Returns:
            logits: (batch, num_classes)
        """
        hand = x[:, :, :self.hand_dim]
        face = x[:, :, self.hand_dim:self.hand_dim + self.face_dim]
        body = x[:, :, self.hand_dim + self.face_dim:]
        logits, _ = self.inner(hand, face, body)
        return logits


# ===========================================================================
# 3. 训练主流程
# ===========================================================================
def main(epochs: int = 5, num_train: int = 600, num_val: int = 150,
         seq_len: int = 16, batch_size: int = 16, lr: float = 3e-4):
    """运行训练管线验证。"""

    logger.info("=" * 60)
    logger.info("SynthMed CSL Recognition — Training Pipeline Validation")
    logger.info("=" * 60)

    # --- 模型 ---
    cfg = FusionConfig(num_classes=NUM_CLASSES)
    model = TriChannelWrapper(cfg)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: TriChannelFusionTransformer  |  params={total_params:,}  |  classes={NUM_CLASSES}")

    # --- 数据 ---
    train_ds = SimulatedCSLDataset(num_train, seq_len=seq_len, seed=42)
    val_ds = SimulatedCSLDataset(num_val, seq_len=seq_len, seed=123)
    logger.info(f"Dataset: train={len(train_ds)}  val={len(val_ds)}  seq_len={seq_len}")

    # --- 训练配置 ---
    train_cfg = TrainConfig(
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        num_workers=0,        # Windows 兼容
        scheduler="cosine",
        grad_clip=1.0,
        save_every=2,
        early_stop_patience=10,
        ckpt_dir=str(Path(__file__).resolve().parent / "checkpoints" / "csl_recognition"),
        log_dir=str(Path(__file__).resolve().parent / "runs" / "csl_recognition"),
    )

    # --- 训练 ---
    criterion = nn.CrossEntropyLoss()
    pipeline = TrainPipeline(model, criterion, train_ds, val_ds, cfg=train_cfg)
    logger.info(f"Starting training: {epochs} epochs, device={train_cfg.resolve_device()}")
    history = pipeline.run()

    # --- 评估：计算 accuracy ---
    device = train_cfg.resolve_device()
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, targets in pipeline.val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            preds = model(inputs).argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(targets.cpu().tolist())

    accuracy = correct / max(total, 1)
    logger.info(f"Validation Accuracy: {accuracy:.4f}  ({correct}/{total})")

    # --- 保存结果摘要 ---
    results = {
        "model": "TriChannelFusionTransformer",
        "num_classes": NUM_CLASSES,
        "train_samples": num_train,
        "val_samples": num_val,
        "seq_len": seq_len,
        "epochs": epochs,
        "best_val_loss": history["best_val_loss"] if "best_val_loss" in history else min(history["val_loss"]),
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "val_accuracy": round(accuracy, 4),
        "total_params": total_params,
    }
    results_path = Path(train_cfg.ckpt_dir) / "train_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    logger.info(f"Results saved to {results_path}")

    # --- 替换 DummyRecognizer 提示 ---
    best_ckpt = Path(train_cfg.ckpt_dir) / "epoch*_best.pt"
    logger.info("")
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE — DummyRecognizer can now be replaced.")
    logger.info("To use the trained model in SignLanguageRecognizer:")
    logger.info("  1. Load checkpoint: pipeline.load_ckpt('checkpoints/csl_recognition/epochX_best.pt')")
    logger.info("  2. Extract inner model: model.inner (TriChannelFusionTransformer)")
    logger.info("  3. Assign to SignLanguageRecognizer.model")
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    main()
