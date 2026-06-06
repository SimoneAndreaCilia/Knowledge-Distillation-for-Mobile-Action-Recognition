# -*- coding: utf-8 -*-
"""
Baseline Training Script
=========================

Main entry point for training either the Teacher (3D ResNet-50) or the
Student (3D MobileNet) on HMDB-51 using standard Cross-Entropy loss.

This establishes the **upper-bound** (Teacher) and **lower-bound**
(Student baseline) accuracy baselines for the knowledge distillation
pipeline.

**Cluster-ready features**:
  • Automatic checkpoint resume (scans checkpoint dir on startup).
  • Per-epoch checkpointing (model, optimizer, scheduler, epoch, best_acc).
  • Hard time-limit awareness (12h default, with configurable safety buffer).
  • All paths configurable via CLI arguments (no hardcoded paths).
  • TensorBoard logging to ``./runs/<experiment_name>/``.

**Usage**::

    # Train student baseline
    python -m src.training.train_baseline \\
        --model student \\
        --data_dir ./data/hmdb51 \\
        --annotation_dir ./data/hmdb51_splits \\
        --epochs 100 \\
        --batch_size 16 \\
        --lr 0.01 \\
        --checkpoint_dir ./checkpoints/baseline \\
        --log_dir ./runs/baseline

    # Fine-tune teacher (with pre-trained ImageNet weights)
    python -m src.training.train_baseline \\
        --model teacher \\
        --pretrained \\
        --data_dir ./data/hmdb51 \\
        --annotation_dir ./data/hmdb51_splits \\
        --epochs 50 \\
        --batch_size 8 \\
        --lr 0.001 \\
        --checkpoint_dir ./checkpoints/teacher \\
        --log_dir ./runs/teacher

    # Load from YAML config (CLI args override config values)
    python -m src.training.train_baseline \\
        --config experiments/configs/baseline.yaml
"""

import argparse
import logging
import os
import sys
import time
from typing import Tuple, Dict, Optional

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---- Project imports ----
from src.utils.seed import set_seed
from src.utils.checkpoint import CheckpointManager
from src.utils.metrics import AverageMeter, TensorBoardLogger
from src.models import build_model
from src.datasets import build_dataset
from src.training.losses import CrossEntropyLoss

# ---- Logging configuration ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_baseline")


# ======================================================================
# ARGUMENT PARSING
# ======================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments with optional YAML config loading.

    YAML config values are used as defaults; explicit CLI arguments
    override them. This enables both quick CLI experiments and
    reproducible config-file-based runs.

    Returns:
        Parsed ``argparse.Namespace`` with all training parameters.
    """
    parser = argparse.ArgumentParser(
        description="Baseline training for Knowledge Distillation pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Config file (optional) ----
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file. CLI args override config values.",
    )

    # ---- Experiment ----
    parser.add_argument("--experiment_name", type=str, default="baseline")
    parser.add_argument("--seed", type=int, default=42)

    # ---- Model ----
    parser.add_argument(
        "--model", type=str, default="student",
        choices=["teacher", "student", "resnet3d_50", "mobilenet3d"],
        help="Model architecture to train.",
    )
    parser.add_argument("--num_classes", type=int, default=51)
    parser.add_argument(
        "--pretrained", action="store_true",
        help="Load ImageNet-inflated pre-trained weights (teacher only).",
    )
    parser.add_argument(
        "--width_mult", type=float, default=1.0,
        help="Width multiplier for the student model.",
    )
    parser.add_argument("--dropout", type=float, default=0.2)

    # ---- Data ----
    parser.add_argument(
        "--data_dir", type=str,
        default=os.environ.get("HMDB51_DATA_DIR", "./data/hmdb51"),
        help="Path to HMDB-51 video directory. Can also set via "
             "HMDB51_DATA_DIR environment variable.",
    )
    parser.add_argument(
        "--annotation_dir", type=str,
        default=os.environ.get("HMDB51_ANNO_DIR", "./data/hmdb51_splits"),
        help="Path to HMDB-51 split annotation files.",
    )
    parser.add_argument("--split", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--frame_size", type=int, default=112)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--dataset_type", type=str, default="video",
        choices=["video", "features"],
    )
    parser.add_argument("--feature_dir", type=str, default=None)

    # ---- Training ----
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument(
        "--scheduler", type=str, default="cosine",
        choices=["cosine", "step", "none"],
    )
    parser.add_argument(
        "--warmup_epochs", type=int, default=5,
        help="Number of linear warmup epochs before the main scheduler.",
    )
    parser.add_argument("--label_smoothing", type=float, default=0.0)

    # ---- Checkpointing ----
    parser.add_argument(
        "--checkpoint_dir", type=str, default="./checkpoints/baseline",
    )
    parser.add_argument("--max_keep", type=int, default=5)
    parser.add_argument(
        "--resume", action="store_true", default=True,
        help="Automatically resume from the latest checkpoint if available.",
    )
    parser.add_argument(
        "--no_resume", action="store_true",
        help="Disable automatic resume (start fresh).",
    )

    # ---- Logging ----
    parser.add_argument("--log_dir", type=str, default="./runs/baseline")
    parser.add_argument("--log_interval", type=int, default=10)

    # ---- Cluster time limit ----
    parser.add_argument(
        "--time_limit_hours", type=float, default=12.0,
        help="Hard time limit in hours (cluster kills the job after this).",
    )
    parser.add_argument(
        "--time_buffer_minutes", type=float, default=15.0,
        help="Safety buffer in minutes before the time limit to allow "
             "graceful checkpoint saving.",
    )

    args = parser.parse_args()

    # ---- Load YAML config and merge ----
    if args.config is not None:
        args = _merge_yaml_config(args)

    # Handle --no_resume flag
    if args.no_resume:
        args.resume = False

    return args


def _merge_yaml_config(args: argparse.Namespace) -> argparse.Namespace:
    """Merge YAML config into argparse namespace.

    Values from the YAML file are used as defaults; any explicitly
    provided CLI arguments take priority.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Updated namespace with YAML defaults applied.
    """
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Flatten nested YAML structure
    flat_config = {}
    for section_name, section_dict in config.items():
        if isinstance(section_dict, dict):
            flat_config.update(section_dict)
        else:
            flat_config[section_name] = section_dict

    # Apply YAML values only where CLI didn't provide an explicit value
    # (argparse defaults are overridden by YAML; explicit CLI overrides YAML)
    for key, value in flat_config.items():
        if hasattr(args, key):
            setattr(args, key, value)

    logger.info(f"Loaded config from: {args.config}")
    return args


# ======================================================================
# TRAINING FUNCTIONS
# ======================================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_interval: int = 10,
) -> Tuple[float, float]:
    """Train the model for one epoch.

    Args:
        model: The model to train (set to train mode internally).
        dataloader: Training data loader.
        criterion: Loss function (``CrossEntropyLoss`` wrapper).
        optimizer: Optimizer instance.
        device: Device for computation.
        epoch: Current epoch number (for progress bar labeling).
        log_interval: Log every N batches within the epoch.

    Returns:
        Tuple of (average_loss, accuracy_percent).
    """
    model.train()
    loss_meter = AverageMeter("Loss")
    acc_meter = AverageMeter("Acc")

    pbar = tqdm(
        dataloader,
        desc=f"Epoch {epoch:03d} [Train]",
        leave=False,
        dynamic_ncols=True,
    )

    for batch_idx, (clips, labels) in enumerate(pbar):
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Forward pass
        logits = model(clips)
        loss, loss_dict = criterion(logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute accuracy
        _, predicted = logits.max(dim=1)
        correct = predicted.eq(labels).sum().item()
        batch_acc = 100.0 * correct / labels.size(0)

        # Update meters
        loss_meter.update(loss.item(), labels.size(0))
        acc_meter.update(batch_acc, labels.size(0))

        # Update progress bar
        pbar.set_postfix(
            loss=f"{loss_meter.avg:.4f}",
            acc=f"{acc_meter.avg:.1f}%",
        )

    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> Tuple[float, float]:
    """Evaluate the model on the validation/test set.

    Args:
        model: The model to evaluate (set to eval mode internally).
        dataloader: Validation data loader.
        criterion: Loss function.
        device: Device for computation.
        epoch: Current epoch number (for progress bar).

    Returns:
        Tuple of (average_loss, accuracy_percent).
    """
    model.eval()
    loss_meter = AverageMeter("Val Loss")
    acc_meter = AverageMeter("Val Acc")

    pbar = tqdm(
        dataloader,
        desc=f"Epoch {epoch:03d} [Val  ]",
        leave=False,
        dynamic_ncols=True,
    )

    for clips, labels in pbar:
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(clips)
        loss, _ = criterion(logits, labels)

        _, predicted = logits.max(dim=1)
        correct = predicted.eq(labels).sum().item()
        batch_acc = 100.0 * correct / labels.size(0)

        loss_meter.update(loss.item(), labels.size(0))
        acc_meter.update(batch_acc, labels.size(0))

        pbar.set_postfix(
            loss=f"{loss_meter.avg:.4f}",
            acc=f"{acc_meter.avg:.1f}%",
        )

    return loss_meter.avg, acc_meter.avg


# ======================================================================
# SCHEDULER FACTORY
# ======================================================================

def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    epochs: int,
    warmup_epochs: int = 0,
) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    """Create a learning rate scheduler.

    If ``warmup_epochs > 0``, a ``SequentialLR`` is returned that chains
    a linear warmup with the main scheduler.

    Args:
        optimizer: The optimizer to schedule.
        scheduler_type: ``"cosine"``, ``"step"``, or ``"none"``.
        epochs: Total number of training epochs.
        warmup_epochs: Number of linear warmup epochs.

    Returns:
        A scheduler instance, or ``None`` if ``scheduler_type="none"``.
    """
    if scheduler_type == "none":
        return None

    # Main scheduler (after warmup)
    if scheduler_type == "cosine":
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs - warmup_epochs
        )
    elif scheduler_type == "step":
        main_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.1
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")

    # Optional warmup
    if warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=warmup_epochs
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_epochs],
        )
        return scheduler

    return main_scheduler


# ======================================================================
# MAIN TRAINING LOOP
# ======================================================================

def main() -> None:
    """Main training entry point.

    Orchestrates the full training pipeline:
      1. Parse arguments and load config.
      2. Set random seed for reproducibility.
      3. Build dataset and data loaders.
      4. Build model, optimizer, and scheduler.
      5. Attempt to resume from checkpoint.
      6. Run the training loop with per-epoch validation.
      7. Save checkpoints and log to TensorBoard.
      8. Respect the cluster time limit.
    """
    args = parse_args()

    # ---- Reproducibility ----
    set_seed(args.seed, deterministic=True)

    # ---- Device setup ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ---- Build dataset and data loaders ----
    logger.info("Building datasets...")
    train_ds, val_ds, _ = build_dataset(
        dataset_type=args.dataset_type,
        data_dir=args.data_dir,
        annotation_dir=args.annotation_dir,
        split=args.split,
        num_frames=args.num_frames,
        frame_size=args.frame_size,
        feature_dir=args.feature_dir,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    logger.info(
        f"Data: {len(train_ds)} train / {len(val_ds)} val samples, "
        f"batch_size={args.batch_size}"
    )

    # ---- Build model ----
    logger.info(f"Building model: {args.model}")
    model = build_model(
        model_name=args.model,
        num_classes=args.num_classes,
        pretrained=args.pretrained,
        width_mult=args.width_mult,
        dropout=args.dropout,
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")

    # ---- Build optimizer ----
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # ---- Build scheduler ----
    scheduler = build_scheduler(
        optimizer, args.scheduler, args.epochs, args.warmup_epochs
    )

    # ---- Build loss function ----
    criterion = CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # ---- Checkpoint management ----
    ckpt_mgr = CheckpointManager(
        save_dir=args.checkpoint_dir, max_keep=args.max_keep
    )

    start_epoch = 0
    best_acc = 0.0

    if args.resume:
        checkpoint = ckpt_mgr.load_latest()
        if checkpoint is not None:
            start_epoch = CheckpointManager.resume_training(
                checkpoint, model, optimizer, scheduler, device
            )
            best_acc = checkpoint.get("best_acc", 0.0)

    # ---- TensorBoard logger ----
    tb_logger = TensorBoardLogger(
        log_dir=args.log_dir,
        experiment_name=args.experiment_name,
    )

    # ---- Time limit tracking ----
    wall_start = time.time()
    time_limit_sec = (
        args.time_limit_hours * 3600 - args.time_buffer_minutes * 60
    )
    epoch_durations = []  # Track epoch durations for time estimation

    # ---- Training loop ----
    logger.info(
        f"Starting training from epoch {start_epoch} to {args.epochs - 1} "
        f"(time limit: {args.time_limit_hours}h, "
        f"buffer: {args.time_buffer_minutes}min)"
    )
    logger.info("=" * 72)

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        # ---- Time limit check ----
        elapsed = time.time() - wall_start
        if elapsed >= time_limit_sec:
            logger.warning(
                f"⚠ Time limit approaching ({elapsed / 3600:.2f}h elapsed). "
                f"Saving checkpoint and exiting gracefully."
            )
            break

        # Estimate if we have time for another epoch
        if epoch_durations:
            avg_epoch_time = sum(epoch_durations) / len(epoch_durations)
            remaining_time = time_limit_sec - elapsed
            if remaining_time < avg_epoch_time * 1.5:
                logger.warning(
                    f"⚠ Not enough time for another epoch "
                    f"(remaining: {remaining_time / 60:.1f}min, "
                    f"avg epoch: {avg_epoch_time / 60:.1f}min). "
                    f"Saving and exiting."
                )
                break

        # ---- Train one epoch ----
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch, args.log_interval,
        )

        # ---- Validate ----
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, epoch
        )

        # ---- Step scheduler ----
        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()

        # ---- Track best model ----
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc

        # ---- Log to TensorBoard ----
        tb_logger.log_training_epoch(
            epoch, train_loss, train_acc, val_loss, val_acc, current_lr
        )

        # ---- Save checkpoint ----
        ckpt_mgr.save(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_acc=best_acc,
            metrics={
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            },
            is_best=is_best,
        )

        # ---- Epoch summary ----
        epoch_time = time.time() - epoch_start
        epoch_durations.append(epoch_time)

        star = " ★" if is_best else ""
        logger.info(
            f"Epoch {epoch:03d}/{args.epochs - 1} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.1f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.1f}%{star} | "
            f"LR: {current_lr:.6f} | "
            f"Time: {epoch_time:.0f}s"
        )

    # ---- Training complete ----
    logger.info("=" * 72)
    total_time = time.time() - wall_start
    logger.info(
        f"Training complete! Best validation accuracy: {best_acc:.2f}% "
        f"(total time: {total_time / 3600:.2f}h)"
    )

    tb_logger.close()


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    main()
