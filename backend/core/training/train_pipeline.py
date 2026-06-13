"""基础训练流水线 — train / validate / test + checkpoint + TensorBoard"""
import json, time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


@dataclass
class TrainConfig:
    epochs: int = 50; lr: float = 1e-4; weight_decay: float = 1e-5
    batch_size: int = 8; num_workers: int = 4
    log_dir: str = "runs"; ckpt_dir: str = "checkpoints"
    save_every: int = 5; early_stop_patience: int = 10; grad_clip: float = 1.0
    scheduler: str = "cosine"  # cosine / step / none
    step_size: int = 10; gamma: float = 0.5; device: str = ""

    def resolve_device(self) -> torch.device:
        return torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))


class TrainPipeline:
    """通用训练循环，注入 model / criterion / datasets 即可运行。"""

    def __init__(self, model: nn.Module, criterion: nn.Module,
                 train_ds, val_ds, test_ds=None, cfg: Optional[TrainConfig] = None):
        self.cfg = cfg or TrainConfig()
        self.device = self.cfg.resolve_device()
        self.model = model.to(self.device)
        self.criterion = criterion
        kw = dict(batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers, pin_memory=True)
        self.train_loader = DataLoader(train_ds, shuffle=True, **kw)
        self.val_loader = DataLoader(val_ds, shuffle=False, **kw)
        self.test_loader = DataLoader(test_ds, shuffle=False, **kw) if test_ds else None
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        self.scheduler = self._build_scheduler()
        self.writer = SummaryWriter(self.cfg.log_dir)
        self.ckpt_dir = Path(self.cfg.ckpt_dir); self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_loss, self.no_improve = float("inf"), 0

    def _build_scheduler(self):
        c = self.cfg
        if c.scheduler == "cosine": return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=c.epochs)
        if c.scheduler == "step": return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=c.step_size, gamma=c.gamma)
        return None

    def train_epoch(self, epoch: int) -> float:
        self.model.train(); total, n = 0.0, 0
        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(inputs), targets)
            loss.backward()
            if self.cfg.grad_clip > 0: nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.optimizer.step(); total += loss.item(); n += 1
        avg = total / max(n, 1); self.writer.add_scalar("loss/train", avg, epoch); return avg

    @torch.no_grad()
    def _eval_loop(self, loader, tag: str, epoch: int) -> float:
        self.model.eval(); total, n = 0.0, 0
        for inputs, targets in loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            total += self.criterion(self.model(inputs), targets).item(); n += 1
        avg = total / max(n, 1); self.writer.add_scalar(f"loss/{tag}", avg, epoch); return avg

    def validate(self, epoch: int) -> float: return self._eval_loop(self.val_loader, "val", epoch)

    def test(self) -> float:
        if not self.test_loader: print("No test set, skipping."); return 0.0
        loss = self._eval_loop(self.test_loader, "test", 0); print(f"Test loss: {loss:.6f}"); return loss

    def save_ckpt(self, epoch: int, tag: str = "") -> Path:
        path = self.ckpt_dir / f"epoch{epoch}{'_' + tag if tag else ''}.pt"
        torch.save({"epoch": epoch, "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                     "scheduler": self.scheduler.state_dict() if self.scheduler else None,
                     "best_val_loss": self.best_val_loss, "config": asdict(self.cfg)}, path)
        return path

    def load_ckpt(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"]); self.optimizer.load_state_dict(ckpt["optimizer"])
        if self.scheduler and ckpt.get("scheduler"): self.scheduler.load_state_dict(ckpt["scheduler"])
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Loaded checkpoint from epoch {ckpt['epoch']}"); return ckpt["epoch"]

    def run(self) -> dict:
        """完整训练 -> 验证 -> 测试，返回历史记录。"""
        history = {"train_loss": [], "val_loss": []}; t0 = time.time()
        for epoch in range(1, self.cfg.epochs + 1):
            tl = self.train_epoch(epoch); vl = self.validate(epoch)
            if self.scheduler: self.scheduler.step()
            history["train_loss"].append(tl); history["val_loss"].append(vl)
            print(f"[{epoch}/{self.cfg.epochs}] train={tl:.6f}  val={vl:.6f}")
            if vl < self.best_val_loss:
                self.best_val_loss = vl; self.no_improve = 0; self.save_ckpt(epoch, "best")
            else: self.no_improve += 1
            if epoch % self.cfg.save_every == 0: self.save_ckpt(epoch)
            if self.no_improve >= self.cfg.early_stop_patience: print(f"Early stop @ epoch {epoch}"); break
        test_loss = self.test(); elapsed = time.time() - t0; self.writer.close()
        summary = {**history, "test_loss": test_loss, "elapsed_sec": round(elapsed, 1)}
        (self.ckpt_dir / "history.json").write_text(json.dumps(summary, indent=2))
        print(f"Done in {elapsed:.1f}s. Best val: {self.best_val_loss:.6f}"); return summary
