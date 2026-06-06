# -*- coding: utf-8 -*-
"""
Checkpoint Management for Cluster Training
===========================================

Designed for shared university GPU clusters that enforce hard time limits
(e.g., 12 hours). This module provides:

  • **Per-epoch checkpointing** — saves model weights, optimizer state,
    scheduler state, current epoch, and best accuracy after every epoch.
  • **Automatic resume** — on startup, scans the checkpoint directory and
    loads the latest valid ``.pth`` file to continue training seamlessly.
  • **Corruption recovery** — if the latest checkpoint is corrupt (e.g.,
    killed mid-write), falls back to the second-most-recent checkpoint.
  • **Best-model tracking** — saves a separate ``best_model.pth`` whenever
    validation accuracy improves.
  • **Disk cleanup** — retains only the N most recent checkpoints to avoid
    filling cluster quotas.

Usage::

    ckpt_mgr = CheckpointManager(save_dir="./checkpoints/baseline", max_keep=5)

    # Attempt to resume
    checkpoint = ckpt_mgr.load_latest()
    if checkpoint is not None:
        start_epoch = CheckpointManager.resume_training(
            checkpoint, model, optimizer, scheduler, device
        )
        best_acc = checkpoint["best_acc"]
    else:
        start_epoch, best_acc = 0, 0.0

    # After each epoch
    ckpt_mgr.save(epoch, model, optimizer, scheduler, best_acc, is_best=improved)
"""

import os
import glob
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages model checkpoints with automatic save, resume, and cleanup.

    Args:
        save_dir: Directory where checkpoint files are stored.
            Created automatically if it does not exist.
        max_keep: Maximum number of epoch checkpoints to retain on disk.
            Oldest checkpoints are deleted when the limit is exceeded.
            Set to ``-1`` to keep all checkpoints indefinitely.
    """

    def __init__(self, save_dir: str, max_keep: int = 5) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_keep = max_keep
        logger.info(f"CheckpointManager initialized at: {self.save_dir}")

    # ==================================================================
    # SAVE
    # ==================================================================
    def save(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[LRScheduler] = None,
        best_acc: float = 0.0,
        metrics: Optional[Dict[str, Any]] = None,
        is_best: bool = False,
    ) -> str:
        """Persist the full training state to disk.

        The checkpoint contains everything needed to resume training:
        model weights, optimizer buffers/momentum, scheduler state,
        epoch counter, and best validation accuracy.

        Args:
            epoch: Current (completed) epoch number (0-indexed).
            model: The ``nn.Module`` whose ``state_dict`` will be saved.
            optimizer: The optimizer whose ``state_dict`` will be saved.
            scheduler: Optional LR scheduler to save.
            best_acc: Best validation accuracy achieved so far.
            metrics: Optional dictionary of additional metrics (e.g.,
                per-class accuracy, loss breakdown).
            is_best: If ``True``, an additional copy is saved as
                ``best_model.pth`` for easy retrieval.

        Returns:
            Absolute path to the saved checkpoint file.
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_acc": best_acc,
            "metrics": metrics or {},
        }

        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        # ---- Save epoch checkpoint ----
        ckpt_path = self.save_dir / f"checkpoint_epoch_{epoch:04d}.pth"
        torch.save(checkpoint, ckpt_path)
        logger.info(f"Checkpoint saved → {ckpt_path}")

        # ---- Save best model separately ----
        if is_best:
            best_path = self.save_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"★ Best model updated (acc={best_acc:.4f}) → {best_path}")

        # ---- Cleanup old checkpoints ----
        self._cleanup()

        return str(ckpt_path)

    # ==================================================================
    # LOAD LATEST
    # ==================================================================
    def load_latest(self) -> Optional[Dict[str, Any]]:
        """Scan the checkpoint directory and load the most recent checkpoint.

        Files are matched by the naming pattern ``checkpoint_epoch_NNNN.pth``
        and sorted numerically so the highest epoch is loaded first.

        Returns:
            A checkpoint dictionary if a valid checkpoint exists, or
            ``None`` if the directory is empty or all checkpoints are corrupt.
        """
        ckpt_files = sorted(
            glob.glob(str(self.save_dir / "checkpoint_epoch_*.pth"))
        )

        if not ckpt_files:
            logger.info("No checkpoints found — training will start from scratch.")
            return None

        # Try the most recent checkpoint first
        latest_path = ckpt_files[-1]
        logger.info(f"Found {len(ckpt_files)} checkpoint(s). Loading: {latest_path}")

        try:
            checkpoint = torch.load(
                latest_path, map_location="cpu", weights_only=False
            )
            return checkpoint
        except Exception as exc:
            logger.warning(
                f"Latest checkpoint is corrupt ({latest_path}): {exc}. "
                f"Trying fallback..."
            )
            # Fallback: try second-most-recent checkpoint
            if len(ckpt_files) >= 2:
                fallback_path = ckpt_files[-2]
                try:
                    checkpoint = torch.load(
                        fallback_path, map_location="cpu", weights_only=False
                    )
                    logger.info(f"Fallback checkpoint loaded: {fallback_path}")
                    return checkpoint
                except Exception as exc2:
                    logger.error(f"Fallback also corrupt ({fallback_path}): {exc2}")

            return None

    # ==================================================================
    # LOAD BEST
    # ==================================================================
    def load_best(self) -> Optional[Dict[str, Any]]:
        """Load the best-accuracy model checkpoint.

        Returns:
            Checkpoint dict, or ``None`` if ``best_model.pth`` doesn't exist.
        """
        best_path = self.save_dir / "best_model.pth"
        if not best_path.exists():
            logger.warning("No best_model.pth found.")
            return None

        try:
            checkpoint = torch.load(
                best_path, map_location="cpu", weights_only=False
            )
            logger.info(
                f"Loaded best model (epoch={checkpoint['epoch']}, "
                f"acc={checkpoint['best_acc']:.4f})"
            )
            return checkpoint
        except Exception as exc:
            logger.error(f"Failed to load best model: {exc}")
            return None

    # ==================================================================
    # RESUME TRAINING
    # ==================================================================
    @staticmethod
    def resume_training(
        checkpoint: Dict[str, Any],
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[LRScheduler] = None,
        device: Optional[torch.device] = None,
    ) -> int:
        """Restore the full training state from a loaded checkpoint.

        After calling this method, the model, optimizer, and scheduler
        are all in the exact state they were at the end of the checkpointed
        epoch. Training should resume from ``returned_epoch``.

        Args:
            checkpoint: Dict returned by ``load_latest()`` or ``load_best()``.
            model: Model to load weights into. Must have the same architecture.
            optimizer: Optimizer to restore state into.
            scheduler: Optional LR scheduler to restore.
            device: Device to move the model (and optimizer tensors) to.
                If ``None``, keeps everything on CPU.

        Returns:
            The epoch number to **start** the next training iteration from
            (i.e., ``checkpoint_epoch + 1``).
        """
        if device is None:
            device = torch.device("cpu")

        # ---- Restore model weights ----
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)

        # ---- Restore optimizer state ----
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # Move optimizer buffers (e.g., momentum) to the correct device
        for state in optimizer.state.values():
            for key, val in state.items():
                if isinstance(val, torch.Tensor):
                    state[key] = val.to(device)

        # ---- Restore scheduler state ----
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        resume_epoch = checkpoint["epoch"] + 1
        best_acc = checkpoint.get("best_acc", 0.0)
        logger.info(
            f"Training state restored from epoch {checkpoint['epoch']} "
            f"(best_acc={best_acc:.4f}). Resuming at epoch {resume_epoch}."
        )
        return resume_epoch

    # ==================================================================
    # INTERNAL: DISK CLEANUP
    # ==================================================================
    def _cleanup(self) -> None:
        """Delete old epoch checkpoints, keeping only the most recent N."""
        if self.max_keep < 0:
            return  # Keep everything

        ckpt_files = sorted(
            glob.glob(str(self.save_dir / "checkpoint_epoch_*.pth"))
        )

        if len(ckpt_files) > self.max_keep:
            to_delete = ckpt_files[: -self.max_keep]
            for filepath in to_delete:
                os.remove(filepath)
                logger.debug(f"Removed old checkpoint: {filepath}")
