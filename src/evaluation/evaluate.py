# -*- coding: utf-8 -*-
"""
Evaluation & Comparison Script
================================

Evaluates and compares test-set accuracy across multiple models:
  • **Teacher** (3D ResNet-50) — upper-bound baseline
  • **Student Baseline** (3D MobileNet, CE-trained) — lower-bound
  • **Student Distilled** (3D MobileNet, KD-trained) — Phase 2 result

Produces:
  1. Terminal output with a formatted comparison table.
  2. Matplotlib comparison figures (accuracy bars, deployment dashboard,
     per-class heatmap, compression scatter) saved to disk.
  3. TensorBoard logs with accuracy comparison scalars, text summaries,
     and embedded figures.
  4. Optional per-class accuracy breakdown.
  5. Optional temperature ablation visualization (requires distilled
     checkpoints at multiple T values).

Usage::

    # Evaluate a single model
    python -m src.evaluation.evaluate \\
        --model teacher \\
        --checkpoint ./checkpoints/teacher/best_model.pth \\
        --data_dir ./data/hmdb51 \\
        --annotation_dir ./data/hmdb51_splits

    # Compare all three models (with figures)
    python -m src.evaluation.evaluate \\
        --compare \\
        --teacher_ckpt ./checkpoints/teacher/best_model.pth \\
        --baseline_ckpt ./checkpoints/baseline/best_model.pth \\
        --distilled_ckpt ./checkpoints/distilled/best_model.pth \\
        --output_dir ./figures \\
        --log_dir ./runs/comparison

    # Temperature ablation comparison
    python -m src.evaluation.evaluate \\
        --temp_ablation \\
        --teacher_ckpt ./checkpoints/teacher/best_model.pth \\
        --baseline_ckpt ./checkpoints/baseline/best_model.pth \\
        --distilled_T1_ckpt ./checkpoints/distilled_T1/best_model.pth \\
        --distilled_T5_ckpt ./checkpoints/distilled_T5/best_model.pth \\
        --distilled_T10_ckpt ./checkpoints/distilled_T10/best_model.pth \\
        --distilled_T20_ckpt ./checkpoints/distilled_T20/best_model.pth
"""

import argparse
import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.seed import set_seed
from src.utils.metrics import AverageMeter, TensorBoardLogger
from src.models import build_model
from src.datasets import build_dataset
from src.evaluation.profile import profile_model
from src.evaluation.comparison import (
    plot_accuracy_comparison,
    plot_deployment_dashboard,
    plot_per_class_heatmap,
    plot_compression_scatter,
    plot_temperature_ablation,
    save_all_figures,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("evaluate")


# ======================================================================
# EVALUATION FUNCTIONS
# ======================================================================

@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int = 51,
) -> Dict[str, float]:
    """Run full test-set evaluation.

    Computes overall top-1 accuracy and per-class accuracy.

    Args:
        model: Trained model to evaluate (set to eval mode internally).
        dataloader: Test data loader.
        device: Device for computation.
        num_classes: Number of action classes.

    Returns:
        Dict containing:
          - ``"top1_acc"``: Overall top-1 accuracy (%).
          - ``"per_class_acc"``: List of per-class accuracies (%).
          - ``"num_samples"``: Total number of test samples.
    """
    model.eval()
    model.to(device)

    correct_total = 0
    total_samples = 0
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    pbar = tqdm(dataloader, desc="Evaluating", leave=False, dynamic_ncols=True)

    for clips, labels in pbar:
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(clips)
        _, predicted = logits.max(dim=1)

        # Overall accuracy
        correct = predicted.eq(labels).sum().item()
        correct_total += correct
        total_samples += labels.size(0)

        # Per-class accuracy
        for i in range(labels.size(0)):
            label = labels[i].item()
            class_total[label] += 1
            if predicted[i].item() == label:
                class_correct[label] += 1

        pbar.set_postfix(acc=f"{100.0 * correct_total / total_samples:.1f}%")

    top1_acc = 100.0 * correct_total / total_samples if total_samples > 0 else 0.0
    per_class_acc = [
        100.0 * class_correct[c] / class_total[c]
        if class_total[c] > 0 else 0.0
        for c in range(num_classes)
    ]

    return {
        "top1_acc": top1_acc,
        "per_class_acc": per_class_acc,
        "num_samples": total_samples,
    }


def load_model_from_checkpoint(
    model_name: str,
    checkpoint_path: str,
    num_classes: int = 51,
    width_mult: float = 1.0,
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    """Instantiate a model and load weights from a checkpoint file.

    Args:
        model_name: ``"teacher"`` or ``"student"``.
        checkpoint_path: Path to the ``.pth`` checkpoint file.
        num_classes: Number of output classes.
        width_mult: Width multiplier (student only).
        device: Device to load the model onto.

    Returns:
        Model with loaded weights, ready for evaluation.
    """
    model = build_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=False,
        width_mult=width_mult,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    epoch = checkpoint.get("epoch", "?")
    best_acc = checkpoint.get("best_acc", 0.0)
    logger.info(
        f"Loaded {model_name} from {checkpoint_path} "
        f"(epoch {epoch}, train best_acc={best_acc:.2f}%)"
    )

    return model


# ======================================================================
# COMPARISON REPORTING
# ======================================================================

def print_comparison_table(results: Dict[str, Dict[str, float]]) -> str:
    """Format a rich comparison table for terminal output.

    Args:
        results: Dict mapping model names to their evaluation results.

    Returns:
        Formatted table string.
    """
    header = (
        "\n"
        "+----------------------+--------------+---------------+---------------+--------------+\n"
        "|       Model          | Test Acc (%) |  Params       | Size (MB)     | Latency (ms) |\n"
        "+----------------------+--------------+---------------+---------------+--------------+"
    )
    rows = []
    for name, res in results.items():
        acc = res.get("top1_acc", 0.0)
        params = res.get("param_count", 0)
        size_mb = res.get("model_size_mb", 0.0)
        latency = res.get("latency_ms", None)
        lat_str = f"{latency:>10.1f}" if latency else "        N/A"
        rows.append(
            f"| {name:<20s} | {acc:>10.2f}%  | {params:>13,} | {size_mb:>11.2f}  | {lat_str}  |"
        )
    footer = (
        "+----------------------+--------------+---------------+---------------+--------------+"
    )

    # Add summary stats
    accs = [r.get("top1_acc", 0) for r in results.values()]
    params_list = [r.get("param_count", 0) for r in results.values()]
    if len(accs) >= 2:
        summary = "\n  Summary:"
        # Find teacher and best student
        names = list(results.keys())
        teacher_acc = None
        for n in names:
            if "teacher" in n.lower():
                teacher_acc = results[n]["top1_acc"]
                teacher_params = results[n]["param_count"]
                break

        for n in names:
            if "distill" in n.lower() and teacher_acc is not None:
                student_acc = results[n]["top1_acc"]
                student_params = results[n]["param_count"]
                gap = teacher_acc - student_acc
                compression = teacher_params / student_params if student_params > 0 else 0
                summary += (
                    f"\n    KD Gap:         {gap:+.2f}% (Teacher - Distilled)"
                    f"\n    Compression:    {compression:.1f}x parameter reduction"
                    f"\n    Acc Retained:   {student_acc / teacher_acc * 100:.1f}% "
                    f"of Teacher accuracy"
                )

    table = header + "\n" + "\n".join(rows) + "\n" + footer
    if len(accs) >= 2:
        table += summary

    return table


# ======================================================================
# FIGURE LOGGING TO TENSORBOARD
# ======================================================================

def log_figures_to_tensorboard(
    tb_logger: TensorBoardLogger,
    figures: Dict[str, "matplotlib.figure.Figure"],
) -> None:
    """Log all comparison figures to TensorBoard.

    Args:
        tb_logger: Active TensorBoard logger instance.
        figures: Dict mapping figure names to Figure objects.
    """
    for name, fig in figures.items():
        try:
            tb_logger.writer.add_figure(f"Figures/{name}", fig, 0)
            logger.info(f"Logged figure '{name}' to TensorBoard")
        except Exception as e:
            logger.warning(f"Failed to log figure '{name}': {e}")


# ======================================================================
# MAIN
# ======================================================================

def main() -> None:
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate and compare model accuracies",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Mode ----
    parser.add_argument(
        "--compare", action="store_true",
        help="Compare multiple models (requires --teacher_ckpt etc.).",
    )
    parser.add_argument(
        "--temp_ablation", action="store_true",
        help="Run temperature ablation comparison.",
    )

    # ---- Single model evaluation ----
    parser.add_argument("--model", type=str, default="student")
    parser.add_argument("--checkpoint", type=str, default=None)

    # ---- Multi-model comparison ----
    parser.add_argument("--teacher_ckpt", type=str, default=None)
    parser.add_argument("--baseline_ckpt", type=str, default=None)
    parser.add_argument("--distilled_ckpt", type=str, default=None)

    # ---- Temperature ablation checkpoints ----
    parser.add_argument("--distilled_T1_ckpt", type=str, default=None)
    parser.add_argument("--distilled_T5_ckpt", type=str, default=None)
    parser.add_argument("--distilled_T10_ckpt", type=str, default=None)
    parser.add_argument("--distilled_T20_ckpt", type=str, default=None)

    # ---- Data ----
    parser.add_argument("--data_dir", type=str, default="./data/hmdb51")
    parser.add_argument("--annotation_dir", type=str, default="./data/hmdb51_splits")
    parser.add_argument("--split", type=int, default=1)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--frame_size", type=int, default=112)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=51)
    parser.add_argument("--width_mult", type=float, default=1.0)
    parser.add_argument("--dataset_type", type=str, default="video")
    parser.add_argument("--feature_dir", type=str, default=None)

    # ---- Output ----
    parser.add_argument("--log_dir", type=str, default="./runs/evaluation")
    parser.add_argument("--output_dir", type=str, default="./figures")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Build test dataset ----
    _, _, test_ds = build_dataset(
        dataset_type=args.dataset_type,
        data_dir=args.data_dir,
        annotation_dir=args.annotation_dir,
        split=args.split,
        num_frames=args.num_frames,
        frame_size=args.frame_size,
        feature_dir=args.feature_dir,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    class_names = getattr(test_ds, "classes", None)

    tb_logger = TensorBoardLogger(
        log_dir=args.log_dir,
        experiment_name="evaluation",
    )

    if args.compare or args.temp_ablation:
        # ==============================================================
        # Multi-model comparison
        # ==============================================================
        all_results = {}

        # Standard comparison models
        models_to_eval = []
        if args.teacher_ckpt:
            models_to_eval.append(("Teacher (ResNet3D-50)", "teacher", args.teacher_ckpt))
        if args.baseline_ckpt:
            models_to_eval.append(("Student Baseline", "student", args.baseline_ckpt))
        if args.distilled_ckpt:
            models_to_eval.append(("Student Distilled", "student", args.distilled_ckpt))

        for name, model_type, ckpt_path in models_to_eval:
            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluating: {name}")
            logger.info(f"{'='*60}")

            model = load_model_from_checkpoint(
                model_type, ckpt_path, args.num_classes, args.width_mult, device
            )

            # Test accuracy
            eval_result = evaluate_model(
                model, test_loader, device, args.num_classes
            )

            # Model profiling
            profile_result = profile_model(
                model, device=device,
                input_shape=(1, 3, args.num_frames, args.frame_size, args.frame_size),
            )

            result = {**eval_result, **profile_result}
            all_results[name] = result

            # Log to TensorBoard
            tb_logger.log_scalar(
                f"Comparison/accuracy", result["top1_acc"],
                len(all_results) - 1,
            )
            tb_logger.log_deployment_metrics(
                name,
                result["param_count"],
                result["model_size_mb"],
                result.get("latency_ms"),
            )

            logger.info(f"{name}: Test Accuracy = {result['top1_acc']:.2f}%")

            del model  # Free GPU memory

        # ---- Temperature ablation ----
        temp_results = None
        if args.temp_ablation:
            temp_results = {}
            temp_ckpts = {
                1: args.distilled_T1_ckpt,
                5: args.distilled_T5_ckpt,
                10: args.distilled_T10_ckpt,
                20: args.distilled_T20_ckpt,
            }

            for temp, ckpt_path in temp_ckpts.items():
                if ckpt_path is None:
                    continue

                name = f"Distilled T={temp}"
                logger.info(f"\n{'='*60}")
                logger.info(f"Evaluating: {name}")
                logger.info(f"{'='*60}")

                model = load_model_from_checkpoint(
                    "student", ckpt_path, args.num_classes,
                    args.width_mult, device,
                )

                eval_result = evaluate_model(
                    model, test_loader, device, args.num_classes
                )

                temp_results[temp] = eval_result["top1_acc"]
                logger.info(f"T={temp}: Accuracy = {eval_result['top1_acc']:.2f}%")

                del model

        # ---- Print comparison table ----
        table = print_comparison_table(all_results)
        logger.info(table)
        tb_logger.writer.add_text("Comparison/summary", f"```\n{table}\n```", 0)

        # ---- Generate and save comparison figures ----
        try:
            import matplotlib
            matplotlib.use("Agg")

            figures = save_all_figures(
                results=all_results,
                output_dir=args.output_dir,
                class_names=class_names,
                temp_results=temp_results,
            )

            # Log figures to TensorBoard
            log_figures_to_tensorboard(tb_logger, figures)

            logger.info(
                f"\n📊 {len(figures)} comparison figures saved to "
                f"{args.output_dir}/ and logged to TensorBoard"
            )

            # Close all matplotlib figures to free memory
            import matplotlib.pyplot as plt
            plt.close("all")

        except Exception as e:
            logger.warning(f"Figure generation failed: {e}")
            logger.warning("Comparison table was still printed above.")

    else:
        # ==============================================================
        # Single model evaluation
        # ==============================================================
        if args.checkpoint is None:
            logger.error("--checkpoint is required for single model evaluation.")
            return

        model = load_model_from_checkpoint(
            args.model, args.checkpoint, args.num_classes, args.width_mult, device
        )

        eval_result = evaluate_model(
            model, test_loader, device, args.num_classes
        )
        profile_result = profile_model(
            model, device=device,
            input_shape=(1, 3, args.num_frames, args.frame_size, args.frame_size),
        )

        result = {**eval_result, **profile_result}

        logger.info(f"\nTest Accuracy: {result['top1_acc']:.2f}%")
        logger.info(f"Parameters:    {result['param_count']:,}")
        logger.info(f"Model Size:    {result['model_size_mb']:.2f} MB")
        if "latency_ms" in result:
            logger.info(f"Latency:       {result['latency_ms']:.2f} ms")

        # Per-class accuracy summary
        per_class = result.get("per_class_acc", [])
        if per_class:
            logger.info(f"\nPer-class accuracy range: "
                        f"{min(per_class):.1f}% — {max(per_class):.1f}%")
            logger.info(f"Mean per-class accuracy: "
                        f"{sum(per_class) / len(per_class):.1f}%")

        tb_logger.log_deployment_metrics(
            args.model,
            result["param_count"],
            result["model_size_mb"],
            result.get("latency_ms"),
        )

    tb_logger.close()


if __name__ == "__main__":
    main()
