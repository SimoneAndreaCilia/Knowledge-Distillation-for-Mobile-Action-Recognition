# -*- coding: utf-8 -*-
"""
Training Metrics & TensorBoard Logging
=======================================

Provides two core utilities:

1. **AverageMeter** — tracks running mean/sum/count for any scalar metric
   during training epochs (loss, accuracy, etc.).

2. **TensorBoardLogger** — structured wrapper around ``SummaryWriter`` that
   enforces a consistent tagging convention for all experiments. Logs are
   written to ``./runs/<experiment_name>/`` so they can be mounted locally
   via sshfs and viewed with ``tensorboard --logdir ./runs``.

Tagging convention::

    Loss/train, Loss/val          — per-epoch losses
    Accuracy/train, Accuracy/val  — per-epoch accuracies
    LearningRate                  — scheduler LR
    KD/total_loss, KD/ce_loss, KD/kl_div_loss  — KD-specific (Phase 2)
    Deployment/param_count, Deployment/model_size_mb, Deployment/latency_ms
"""

import time
import logging
from typing import Optional, Dict

from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


# ======================================================================
# AVERAGE METER
# ======================================================================
class AverageMeter:
    """Computes and stores the running average of a scalar metric.

    Typical usage in a training loop::

        loss_meter = AverageMeter("Loss")
        for inputs, targets in dataloader:
            loss = criterion(model(inputs), targets)
            loss_meter.update(loss.item(), n=inputs.size(0))
        print(f"Epoch loss: {loss_meter.avg:.4f}")

    Args:
        name: Human-readable name for logging/printing.
    """

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.reset()

    def reset(self) -> None:
        """Reset all counters to zero."""
        self.val: float = 0.0   # Most recent value
        self.avg: float = 0.0   # Running average
        self.sum: float = 0.0   # Running sum
        self.count: int = 0     # Total sample count

    def update(self, val: float, n: int = 1) -> None:
        """Record a new measurement.

        Args:
            val: The metric value (e.g., batch loss).
            n: Number of samples this value represents (for weighted avg).
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0

    def __str__(self) -> str:
        return f"{self.name}: {self.avg:.4f} (latest: {self.val:.4f})"


# ======================================================================
# TENSORBOARD LOGGER
# ======================================================================
class TensorBoardLogger:
    """Structured experiment logger backed by TensorBoard.

    All logs are written to ``log_dir`` (typically ``./runs/<name>/``).
    The directory is created automatically by SummaryWriter.

    Args:
        log_dir: Absolute or relative path for TensorBoard event files.
            Critical: must be under ``./runs/`` for sshfs compatibility.
        experiment_name: Descriptive label for this run (used in log messages).
    """

    def __init__(self, log_dir: str, experiment_name: str = "experiment") -> None:
        self.writer = SummaryWriter(log_dir=log_dir)
        self.experiment_name = experiment_name
        self.start_time = time.time()
        logger.info(
            f"TensorBoard logger initialized: {experiment_name} → {log_dir}"
        )

    # ---- Generic scalar logging ----

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a single scalar value.

        Args:
            tag: TensorBoard tag (e.g., ``"Loss/train"``).
            value: Scalar value to record.
            step: Global step (typically epoch number).
        """
        self.writer.add_scalar(tag, value, step)

    def log_scalars(
        self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int
    ) -> None:
        """Log multiple scalars under a shared main tag.

        Useful for overlaying train/val curves on the same plot.

        Args:
            main_tag: Group name (e.g., ``"Loss"``).
            tag_scalar_dict: Mapping of sub-tag → value.
            step: Global step.
        """
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    # ---- Structured epoch logging ----

    def log_training_epoch(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        lr: float,
    ) -> None:
        """Log all standard metrics for one training epoch.

        Writes to tags: ``Loss/train``, ``Loss/val``, ``Accuracy/train``,
        ``Accuracy/val``, ``LearningRate``.

        Args:
            epoch: The completed epoch number (0-indexed).
            train_loss: Average training loss for this epoch.
            train_acc: Training accuracy (0–100 scale).
            val_loss: Average validation loss for this epoch.
            val_acc: Validation accuracy (0–100 scale).
            lr: Current learning rate from the scheduler.
        """
        self.log_scalar("Loss/train", train_loss, epoch)
        self.log_scalar("Loss/val", val_loss, epoch)
        self.log_scalar("Accuracy/train", train_acc, epoch)
        self.log_scalar("Accuracy/val", val_acc, epoch)
        self.log_scalar("LearningRate", lr, epoch)

    # ---- KD-specific logging (used in Phase 2) ----

    def log_kd_losses(
        self,
        epoch: int,
        total_loss: float,
        ce_loss: float,
        kd_loss: float,
    ) -> None:
        """Log the decomposed knowledge distillation loss components.

        Args:
            epoch: The completed epoch number.
            total_loss: Combined KD loss (alpha * CE + (1-alpha) * T²·KLDiv).
            ce_loss: Hard-label cross-entropy component.
            kd_loss: Soft-label KL divergence component.
        """
        self.log_scalar("KD/total_loss", total_loss, epoch)
        self.log_scalar("KD/ce_loss", ce_loss, epoch)
        self.log_scalar("KD/kl_div_loss", kd_loss, epoch)

    # ---- Deployment / profiling metrics ----

    def log_deployment_metrics(
        self,
        model_name: str,
        param_count: int,
        model_size_mb: float,
        inference_latency_ms: Optional[float] = None,
    ) -> None:
        """Log model deployment characteristics as scalars and text.

        Args:
            model_name: Identifier string (e.g., ``"teacher"``, ``"student"``).
            param_count: Total number of trainable parameters.
            model_size_mb: On-disk model size in megabytes.
            inference_latency_ms: Mean single-sample inference time in ms.
        """
        # Log as scalars for easy comparison across experiments
        self.log_scalar(f"Deployment/{model_name}/param_count", param_count, 0)
        self.log_scalar(f"Deployment/{model_name}/model_size_mb", model_size_mb, 0)

        if inference_latency_ms is not None:
            self.log_scalar(
                f"Deployment/{model_name}/latency_ms", inference_latency_ms, 0
            )

        # Also log a human-readable text summary
        summary = (
            f"**{model_name}**\n\n"
            f"| Metric | Value |\n"
            f"|--------|-------|\n"
            f"| Parameters | {param_count:,} |\n"
            f"| Model Size | {model_size_mb:.2f} MB |\n"
        )
        if inference_latency_ms is not None:
            summary += f"| Inference Latency | {inference_latency_ms:.2f} ms |\n"

        self.writer.add_text(f"Deployment/{model_name}", summary, 0)

    # ---- Lifecycle ----

    def close(self) -> None:
        """Flush pending events and close the writer."""
        elapsed = time.time() - self.start_time
        logger.info(
            f"TensorBoard logger closed for '{self.experiment_name}' "
            f"(session duration: {elapsed / 60:.1f} min)"
        )
        self.writer.flush()
        self.writer.close()
