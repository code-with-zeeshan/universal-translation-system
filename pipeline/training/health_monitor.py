import logging
import torch
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

EXPECTED_LOSS = {
    1: {"center": 8.5, "warn_above": 10.0, "label": "random init"},
    3: {"center": 5.5, "warn_above": 7.5, "label": "early learning"},
    5: {"center": 4.0, "warn_above": 6.0, "label": "converging"},
    10: {"center": 3.5, "warn_above": 5.0, "label": "near convergence"},
}

class TrainingHealthMonitor:
    def __init__(self, checkpoint_dir: Path, experiment_name: str):
        self.checkpoint_dir = checkpoint_dir
        self.experiment_name = experiment_name
        self.epoch_losses: list[float] = []
        self.issues: list[str] = []

    def on_epoch_end(
        self,
        epoch: int,
        num_epochs: int,
        train_loss: float,
        val_loss: Optional[float],
        lr_val: float,
        _unused: float,
    ) -> None:
        loss = val_loss if val_loss is not None else train_loss
        self.epoch_losses.append(loss)
        self._validate_loss(epoch + 1, loss)
        self._verify_checkpoint(epoch + 1)
        self._print_summary(epoch + 1, num_epochs, loss, lr_val)

    def _validate_loss(self, epoch: int, loss: float) -> None:
        if epoch in EXPECTED_LOSS:
            spec = EXPECTED_LOSS[epoch]
            if loss > spec["warn_above"]:
                msg = (
                    f"Loss {loss:.2f} at epoch {epoch} exceeds expected {spec['warn_above']:.1f} "
                    f"({spec['label']}, expected ~{spec['center']}). "
                )
                if epoch == 1:
                    msg += "If >20.0, check data pipeline / tokenization."
                else:
                    msg += "Check learning rate, data quality, or model initialization."
                self.issues.append(msg)
                logger.warning("⚠️  " + msg)

    def _verify_checkpoint(self, epoch: int) -> None:
        candidates = [
            self.checkpoint_dir / "best_model.pt",
            self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt",
        ]
        for path in candidates:
            if path.exists():
                size = path.stat().st_size
                if size < 1024:
                    msg = f"Checkpoint {path.name} is suspiciously small ({size} bytes)"
                    self.issues.append(msg)
                    logger.warning("⚠️  " + msg)
                else:
                    try:
                        ckpt = torch.load(path, map_location="cpu", weights_only=True)
                        if "epoch" not in ckpt:
                            msg = f"Checkpoint {path.name} missing 'epoch' key — may be corrupt"
                            self.issues.append(msg)
                            logger.warning("⚠️  " + msg)
                    except Exception as e:
                        msg = f"Checkpoint {path.name} failed to load: {e}"
                        self.issues.append(msg)
                        logger.warning("⚠️  " + msg)

    def _print_summary(self, epoch: int, num_epochs: int, loss: float, lr: float) -> None:
        trend = ""
        if len(self.epoch_losses) >= 2:
            prev = self.epoch_losses[-2]
            delta = prev - loss
            if delta > 0.5:
                trend = "↓ good drop"
            elif delta > 0.1:
                trend = "↓ improving"
            elif delta > -0.1:
                trend = "→ plateau"
            else:
                trend = "↑ rising"
        logger.info(
            f"[health] epoch {epoch}/{num_epochs} | loss {loss:.3f} | lr {lr:.2e} | {trend}"
        )

    def final_report(self) -> dict:
        return {
            "experiment": self.experiment_name,
            "epochs_trained": len(self.epoch_losses),
            "final_loss": self.epoch_losses[-1] if self.epoch_losses else None,
            "issues": self.issues,
            "healthy": len(self.issues) == 0,
        }
