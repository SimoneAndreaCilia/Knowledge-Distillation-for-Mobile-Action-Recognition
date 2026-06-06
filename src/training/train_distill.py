# -*- coding: utf-8 -*-
"""
Knowledge Distillation Training Script
========================================

Trains the Student (3D MobileNet) using knowledge distilled from a
frozen Teacher (3D ResNet-50). Supports two distillation mechanisms:

  1. **Logit-level KD** (Hinton et al., 2015): temperature-scaled KL
     divergence between Teacher and Student softmax distributions,
     combined with standard hard-label cross-entropy.

  2. **Attention Transfer** (Zagoruyko & Komodakis, 2017): optional
     intermediate-layer attention map matching that transfers spatial
     attention patterns from Teacher to Student at zero extra parameter
     cost. Controlled via ``--use_attention_transfer``.

  3. **Hint-based Learning** (Romero et al., 2015): optional adapter-
     based feature matching. Adds lightweight ``1×1×1`` convolutions
     to project Student features into Teacher space. Controlled via
     ``--use_hint_learning``.

**Temperature ablation**: Easily sweep T ∈ {1, 5, 10, 20} by passing
different config files or ``--temperature <T>``.

**Cluster features**: Inherits all cluster-ready features from the
baseline script (automatic checkpoint resume, 12h time limit, etc.).

Usage::

    # Standard KD with T=5
    python -m src.training.train_distill \\
        --config experiments/configs/distill_T5.yaml

    # KD + Attention Transfer
    python -m src.training.train_distill \\
        --config experiments/configs/distill_T5.yaml \\
        --use_attention_transfer

    # KD + Hint-based learning
    python -m src.training.train_distill \\
        --config experiments/configs/distill_T5.yaml \\
        --use_hint_learning

    # Temperature ablation via CLI
    python -m src.training.train_distill \\
        --teacher_checkpoint ./checkpoints/teacher/best_model.pth \\
        --temperature 10 \\
        --checkpoint_dir ./checkpoints/distilled_T10 \\
        --log_dir ./runs/distilled_T10
"""

import argparse
import logging
import os
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
from src.models.attention_adapter import build_hint_adapters, HintLoss
from src.datasets import build_dataset
from src.training.losses import KnowledgeDistillationLoss, AttentionTransferLoss

# ---- Logging configuration ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_distill")


# ======================================================================
# ARGUMENT PARSING
# ======================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for KD training.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Knowledge Distillation training for Student model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Config file ----
    parser.add_argument("--config", type=str, default=None)

    # ---- Experiment ----
    parser.add_argument("--experiment_name", type=str, default="distilled_T5")
    parser.add_argument("--seed", type=int, default=42)

    # ---- Teacher ----
    parser.add_argument(
        "--teacher_checkpoint", type=str, required=False,
        default="./checkpoints/teacher/best_model.pth",
        help="Path to the trained Teacher model checkpoint.",
    )
    parser.add_argument("--teacher_num_classes", type=int, default=51)

    # ---- Student ----
    parser.add_argument("--num_classes", type=int, default=51)
    parser.add_argument("--width_mult", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.2)

    # ---- KD hyperparameters ----
    parser.add_argument(
        "--temperature", type=float, default=5.0,
        help="Temperature T for softmax smoothing. Higher T reveals more "
             "inter-class structure. Ablate with T ∈ {1, 5, 10, 20}.",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.3,
        help="Balance factor: alpha * CE + (1-alpha) * T²·KLDiv. "
             "Lower alpha gives more weight to the teacher's soft targets.",
    )

    # ---- Attention Transfer ----
    parser.add_argument(
        "--use_attention_transfer", action="store_true",
        help="Enable parameter-free attention map matching (AT).",
    )
    parser.add_argument(
        "--at_beta", type=float, default=1000.0,
        help="Weight for the Attention Transfer loss term. AT losses are "
             "typically much smaller than CE/KD, so a large β is needed.",
    )

    # ---- Hint-based Learning ----
    parser.add_argument(
        "--use_hint_learning", action="store_true",
        help="Enable adapter-based hint learning (FitNets). "
             "Adds trainable 1×1×1 conv adapters.",
    )
    parser.add_argument(
        "--hint_beta", type=float, default=0.5,
        help="Weight for the hint-based MSE loss term.",
    )

    # ---- Data ----
    parser.add_argument(
        "--data_dir", type=str,
        default=os.environ.get("HMDB51_DATA_DIR", "./data/hmdb51"),
    )
    parser.add_argument(
        "--annotation_dir", type=str,
        default=os.environ.get("HMDB51_ANNO_DIR", "./data/hmdb51_splits"),
    )
    parser.add_argument("--split", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--frame_size", type=int, default=112)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--dataset_type", type=str, default="video")
    parser.add_argument("--feature_dir", type=str, default=None)

    # ---- Training ----
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="cosine")
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--label_smoothing", type=float, default=0.0)

    # ---- Checkpointing ----
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/distilled_T5")
    parser.add_argument("--max_keep", type=int, default=5)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no_resume", action="store_true")

    # ---- Logging ----
    parser.add_argument("--log_dir", type=str, default="./runs/distilled_T5")
    parser.add_argument("--log_interval", type=int, default=10)

    # ---- Cluster ----
    parser.add_argument("--time_limit_hours", type=float, default=12.0)
    parser.add_argument("--time_buffer_minutes", type=float, default=15.0)

    args = parser.parse_args()

    # ---- Load YAML config ----
    if args.config is not None:
        args = _merge_yaml_config(args)

    if args.no_resume:
        args.resume = False

    return args


def _merge_yaml_config(args: argparse.Namespace) -> argparse.Namespace:
    """Merge YAML config values into argparse namespace."""
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    flat_config = {}
    for section_name, section_dict in config.items():
        if isinstance(section_dict, dict):
            flat_config.update(section_dict)
        else:
            flat_config[section_name] = section_dict

    for key, value in flat_config.items():
        if hasattr(args, key):
            setattr(args, key, value)

    logger.info(f"Loaded config from: {args.config}")
    return args


# ======================================================================
# TRAINING FUNCTIONS
# ======================================================================

def train_one_epoch_kd(
    student: nn.Module,
    teacher: nn.Module,
    dataloader: DataLoader,
    kd_criterion: KnowledgeDistillationLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    at_criterion: Optional[AttentionTransferLoss] = None,
    at_beta: float = 1000.0,
    hint_criterion: Optional[HintLoss] = None,
    log_interval: int = 10,
) -> Tuple[float, float, Dict[str, float]]:
    """Train the Student for one epoch using Knowledge Distillation.

    The Teacher is kept frozen in eval mode throughout. Only the Student
    (and optional hint adapters) receive gradient updates.

    Args:
        student: Student model (trainable).
        teacher: Teacher model (frozen, eval mode).
        dataloader: Training data loader.
        kd_criterion: KD loss function (CE + KLDiv).
        optimizer: Optimizer for Student (+ adapter) parameters.
        device: Computation device.
        epoch: Current epoch number.
        at_criterion: Optional Attention Transfer loss.
        at_beta: Weight for the AT loss term.
        hint_criterion: Optional hint-based learning loss (with adapters).
        log_interval: Log every N batches.

    Returns:
        Tuple of (avg_loss, accuracy, loss_breakdown_dict).
    """
    student.train()
    teacher.eval()  # Teacher is ALWAYS in eval mode

    loss_meter = AverageMeter("Loss")
    acc_meter = AverageMeter("Acc")
    ce_meter = AverageMeter("CE")
    kd_meter = AverageMeter("KD")
    at_meter = AverageMeter("AT")
    hint_meter = AverageMeter("Hint")

    pbar = tqdm(
        dataloader,
        desc=f"Epoch {epoch:03d} [KD Train]",
        leave=False,
        dynamic_ncols=True,
    )

    for batch_idx, (clips, labels) in enumerate(pbar):
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # ---- Teacher forward pass (no gradients) ----
        with torch.no_grad():
            teacher_logits = teacher(clips)
            teacher_features = teacher.get_features()

        # ---- Student forward pass ----
        student_logits = student(clips)
        student_features = student.get_features()

        # ---- KD loss (CE + KLDiv) ----
        kd_loss, kd_dict = kd_criterion(student_logits, teacher_logits, labels)
        total_loss = kd_loss

        # ---- Optional: Attention Transfer loss ----
        at_loss_val = 0.0
        if at_criterion is not None:
            at_loss, at_dict = at_criterion(student_features, teacher_features)
            total_loss = total_loss + at_beta * at_loss
            at_loss_val = at_dict.get("at_loss_total", 0.0)

        # ---- Optional: Hint-based learning loss ----
        hint_loss_val = 0.0
        if hint_criterion is not None:
            hint_loss, hint_dict = hint_criterion(
                student_features, teacher_features
            )
            total_loss = total_loss + hint_loss  # Already weighted internally
            hint_loss_val = hint_dict.get("hint_loss_total", 0.0)

        # ---- Backward pass ----
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # ---- Compute accuracy ----
        _, predicted = student_logits.max(dim=1)
        correct = predicted.eq(labels).sum().item()
        batch_acc = 100.0 * correct / labels.size(0)

        # ---- Update meters ----
        bs = labels.size(0)
        loss_meter.update(total_loss.item(), bs)
        acc_meter.update(batch_acc, bs)
        ce_meter.update(kd_dict.get("ce_loss", 0.0), bs)
        kd_meter.update(kd_dict.get("kd_loss", 0.0), bs)
        at_meter.update(at_loss_val, bs)
        hint_meter.update(hint_loss_val, bs)

        pbar.set_postfix(
            loss=f"{loss_meter.avg:.4f}",
            acc=f"{acc_meter.avg:.1f}%",
            ce=f"{ce_meter.avg:.3f}",
            kd=f"{kd_meter.avg:.3f}",
        )

    loss_breakdown = {
        "total_loss": loss_meter.avg,
        "ce_loss": ce_meter.avg,
        "kd_loss": kd_meter.avg,
        "at_loss": at_meter.avg,
        "hint_loss": hint_meter.avg,
    }

    return loss_meter.avg, acc_meter.avg, loss_breakdown


@torch.no_grad()
def validate_kd(
    student: nn.Module,
    teacher: nn.Module,
    dataloader: DataLoader,
    kd_criterion: KnowledgeDistillationLoss,
    device: torch.device,
    epoch: int,
) -> Tuple[float, float]:
    """Validate the distilled Student on the validation set.

    Uses the same KD loss for validation to track distillation progress.

    Args:
        student: Student model.
        teacher: Teacher model (frozen).
        dataloader: Validation data loader.
        kd_criterion: KD loss function.
        device: Computation device.
        epoch: Current epoch number.

    Returns:
        Tuple of (avg_loss, accuracy).
    """
    student.eval()
    teacher.eval()

    loss_meter = AverageMeter("Val Loss")
    acc_meter = AverageMeter("Val Acc")

    pbar = tqdm(
        dataloader,
        desc=f"Epoch {epoch:03d} [KD Val  ]",
        leave=False,
        dynamic_ncols=True,
    )

    for clips, labels in pbar:
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        teacher_logits = teacher(clips)
        student_logits = student(clips)

        kd_loss, _ = kd_criterion(student_logits, teacher_logits, labels)

        _, predicted = student_logits.max(dim=1)
        correct = predicted.eq(labels).sum().item()
        batch_acc = 100.0 * correct / labels.size(0)

        loss_meter.update(kd_loss.item(), labels.size(0))
        acc_meter.update(batch_acc, labels.size(0))

        pbar.set_postfix(
            loss=f"{loss_meter.avg:.4f}",
            acc=f"{acc_meter.avg:.1f}%",
        )

    return loss_meter.avg, acc_meter.avg


# ======================================================================
# SCHEDULER FACTORY (reused from baseline)
# ======================================================================

def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    epochs: int,
    warmup_epochs: int = 0,
) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    """Create a learning rate scheduler with optional warmup."""
    if scheduler_type == "none":
        return None

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
    """Main KD training entry point.

    Orchestrates the full distillation pipeline:
      1. Load the frozen Teacher from checkpoint.
      2. Create/resume the Student model.
      3. Build the KD loss (+ optional AT / Hint losses).
      4. Run the training loop with KD-specific metric logging.
    """
    args = parse_args()

    # ---- Reproducibility ----
    set_seed(args.seed, deterministic=True)

    # ---- Device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ==================================================================
    # TEACHER: Load and freeze
    # ==================================================================
    logger.info(f"Loading Teacher from: {args.teacher_checkpoint}")
    teacher = build_model(
        model_name="teacher",
        num_classes=args.teacher_num_classes,
        pretrained=False,
    )

    teacher_ckpt = torch.load(
        args.teacher_checkpoint, map_location="cpu", weights_only=False
    )
    teacher.load_state_dict(teacher_ckpt["model_state_dict"])
    teacher.to(device)
    teacher.eval()

    # Freeze ALL teacher parameters — no gradients ever flow through
    for param in teacher.parameters():
        param.requires_grad = False

    teacher_acc = teacher_ckpt.get("best_acc", "N/A")
    logger.info(
        f"Teacher loaded (epoch={teacher_ckpt.get('epoch', '?')}, "
        f"best_acc={teacher_acc}). All parameters frozen."
    )

    # ==================================================================
    # STUDENT: Create or resume
    # ==================================================================
    logger.info("Building Student model...")
    student = build_model(
        model_name="student",
        num_classes=args.num_classes,
        width_mult=args.width_mult,
        dropout=args.dropout,
    )
    student.to(device)

    # ==================================================================
    # LOSS FUNCTIONS
    # ==================================================================
    logger.info(
        f"KD Configuration: T={args.temperature}, α={args.alpha}"
    )

    kd_criterion = KnowledgeDistillationLoss(
        temperature=args.temperature,
        alpha=args.alpha,
        label_smoothing=args.label_smoothing,
    )

    # ---- Optional: Attention Transfer ----
    at_criterion = None
    if args.use_attention_transfer:
        at_criterion = AttentionTransferLoss()
        logger.info(
            f"Attention Transfer enabled (β={args.at_beta}). "
            f"Pairs: {at_criterion.feature_pairs}"
        )

    # ---- Optional: Hint-based Learning ----
    hint_criterion = None
    hint_adapters = None
    if args.use_hint_learning:
        logger.info("Building hint adapters...")
        hint_adapters = build_hint_adapters(teacher, student, device=device)
        hint_criterion = HintLoss(
            adapters=hint_adapters, beta=args.hint_beta
        )
        logger.info(
            f"Hint-based learning enabled (β={args.hint_beta}). "
            f"Adapter params: "
            f"{sum(p.numel() for p in hint_adapters.parameters()):,}"
        )

    # ==================================================================
    # OPTIMIZER (Student params + optional adapter params)
    # ==================================================================
    param_groups = list(student.parameters())
    if hint_adapters is not None:
        param_groups = param_groups + list(hint_adapters.parameters())

    optimizer = torch.optim.SGD(
        param_groups,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # ==================================================================
    # SCHEDULER
    # ==================================================================
    scheduler = build_scheduler(
        optimizer, args.scheduler, args.epochs, args.warmup_epochs
    )

    # ==================================================================
    # DATASET
    # ==================================================================
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
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    logger.info(
        f"Data: {len(train_ds)} train / {len(val_ds)} val, "
        f"batch_size={args.batch_size}"
    )

    # ==================================================================
    # CHECKPOINT MANAGEMENT
    # ==================================================================
    ckpt_mgr = CheckpointManager(
        save_dir=args.checkpoint_dir, max_keep=args.max_keep
    )

    start_epoch = 0
    best_acc = 0.0

    if args.resume:
        checkpoint = ckpt_mgr.load_latest()
        if checkpoint is not None:
            start_epoch = CheckpointManager.resume_training(
                checkpoint, student, optimizer, scheduler, device
            )
            best_acc = checkpoint.get("best_acc", 0.0)

            # Resume hint adapters if present
            if hint_adapters is not None and "adapter_state_dict" in checkpoint:
                hint_adapters.load_state_dict(checkpoint["adapter_state_dict"])
                logger.info("Hint adapter state restored from checkpoint.")

    # ==================================================================
    # TENSORBOARD
    # ==================================================================
    tb_logger = TensorBoardLogger(
        log_dir=args.log_dir, experiment_name=args.experiment_name
    )

    # Log hyperparameters as text
    hparams_text = (
        f"| Param | Value |\n|---|---|\n"
        f"| Temperature | {args.temperature} |\n"
        f"| Alpha | {args.alpha} |\n"
        f"| Attention Transfer | {args.use_attention_transfer} |\n"
        f"| AT Beta | {args.at_beta} |\n"
        f"| Hint Learning | {args.use_hint_learning} |\n"
        f"| Hint Beta | {args.hint_beta} |\n"
        f"| LR | {args.lr} |\n"
        f"| Width Mult | {args.width_mult} |\n"
    )
    tb_logger.writer.add_text("Hyperparameters", hparams_text, 0)

    # ==================================================================
    # TRAINING LOOP
    # ==================================================================
    wall_start = time.time()
    time_limit_sec = (
        args.time_limit_hours * 3600 - args.time_buffer_minutes * 60
    )
    epoch_durations = []

    logger.info(
        f"\nStarting KD training from epoch {start_epoch} to {args.epochs - 1}"
    )
    logger.info(
        f"  Teacher: frozen, {sum(p.numel() for p in teacher.parameters()):,} params"
    )
    logger.info(
        f"  Student: trainable, {sum(p.numel() for p in student.parameters()):,} params"
    )
    logger.info(f"  KD: T={args.temperature}, α={args.alpha}")
    if args.use_attention_transfer:
        logger.info(f"  AT: β={args.at_beta}")
    if args.use_hint_learning:
        logger.info(f"  Hint: β={args.hint_beta}")
    logger.info("=" * 72)

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        # ---- Time limit check ----
        elapsed = time.time() - wall_start
        if elapsed >= time_limit_sec:
            logger.warning(
                f"⚠ Time limit approaching ({elapsed / 3600:.2f}h). "
                f"Saving and exiting."
            )
            break

        if epoch_durations:
            avg_epoch_time = sum(epoch_durations) / len(epoch_durations)
            remaining_time = time_limit_sec - elapsed
            if remaining_time < avg_epoch_time * 1.5:
                logger.warning(
                    f"⚠ Insufficient time for another epoch. "
                    f"Saving and exiting."
                )
                break

        # ---- Train ----
        train_loss, train_acc, loss_breakdown = train_one_epoch_kd(
            student=student,
            teacher=teacher,
            dataloader=train_loader,
            kd_criterion=kd_criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            at_criterion=at_criterion,
            at_beta=args.at_beta,
            hint_criterion=hint_criterion,
            log_interval=args.log_interval,
        )

        # ---- Validate ----
        val_loss, val_acc = validate_kd(
            student, teacher, val_loader, kd_criterion, device, epoch
        )

        # ---- Step scheduler ----
        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()

        # ---- Track best ----
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc

        # ---- TensorBoard logging ----
        tb_logger.log_training_epoch(
            epoch, train_loss, train_acc, val_loss, val_acc, current_lr
        )
        tb_logger.log_kd_losses(
            epoch,
            total_loss=loss_breakdown["total_loss"],
            ce_loss=loss_breakdown["ce_loss"],
            kd_loss=loss_breakdown["kd_loss"],
        )
        if args.use_attention_transfer:
            tb_logger.log_scalar("AT/loss", loss_breakdown["at_loss"], epoch)
        if args.use_hint_learning:
            tb_logger.log_scalar("Hint/loss", loss_breakdown["hint_loss"], epoch)

        # ---- Checkpoint ----
        save_dict = {
            "kd_config": {
                "temperature": args.temperature,
                "alpha": args.alpha,
                "use_attention_transfer": args.use_attention_transfer,
                "use_hint_learning": args.use_hint_learning,
            },
        }
        if hint_adapters is not None:
            save_dict["adapter_state_dict"] = hint_adapters.state_dict()

        ckpt_mgr.save(
            epoch=epoch,
            model=student,
            optimizer=optimizer,
            scheduler=scheduler,
            best_acc=best_acc,
            metrics={
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                **loss_breakdown,
                **save_dict,
            },
            is_best=is_best,
        )

        # ---- Epoch summary ----
        epoch_time = time.time() - epoch_start
        epoch_durations.append(epoch_time)

        star = " ★" if is_best else ""
        logger.info(
            f"Epoch {epoch:03d}/{args.epochs - 1} | "
            f"Train: {train_loss:.4f} ({train_acc:.1f}%) | "
            f"Val: {val_loss:.4f} ({val_acc:.1f}%){star} | "
            f"CE: {loss_breakdown['ce_loss']:.3f} "
            f"KD: {loss_breakdown['kd_loss']:.3f} | "
            f"LR: {current_lr:.6f} | {epoch_time:.0f}s"
        )

    # ---- Done ----
    logger.info("=" * 72)
    total_time = time.time() - wall_start
    logger.info(
        f"KD training complete! Best val accuracy: {best_acc:.2f}% "
        f"(T={args.temperature}, α={args.alpha}, "
        f"time: {total_time / 3600:.2f}h)"
    )
    tb_logger.close()


if __name__ == "__main__":
    main()
