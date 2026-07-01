"""
evaluate_top5_all.py
--------------------
Evaluates Top-1 and Top-5 accuracy on HMDB-51 split-1 validation set
for the three main models:
  1. Teacher  (ResNet3D-50)
  2. Student Baseline (MobileNet3D width=1.0, trained with hard labels)
  3. Student KD+AT    (MobileNet3D width=1.0, best distilled checkpoint)

Results are saved to: results/top5_comparison.txt
Run from the project root:
    python src/evaluation/evaluate_top5_all.py
"""

import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.models import build_model
from src.datasets import build_dataset

# -----------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------
DATA_DIR        = "./data/hmdb51"
ANNOTATION_DIR  = "./data/hmdb51_splits"
OUTPUT_FILE     = "./results/top5_comparison.txt"

MODELS = [
    {
        "name":       "Teacher (ResNet3D-50)",
        "type":       "teacher",
        "width_mult": None,
        "checkpoint": "checkpoints/teacher/best_model.pth",
    },
    {
        "name":       "Student Baseline (Hard Labels)",
        "type":       "student",
        "width_mult": 1.0,
        "checkpoint": "checkpoints/student_baseline/best_model.pth",
    },
    {
        "name":       "Student KD+AT (beta=1000, T=10)",
        "type":       "student",
        "width_mult": 1.0,
        "checkpoint": "checkpoints/distilled_AT_T10_seed1234/best_model.pth",
    },
]

BATCH_SIZE  = 16
NUM_FRAMES  = 16
FRAME_SIZE  = 112
SPLIT       = 1


def load_model(cfg, device):
    """Build and load a model from a checkpoint."""
    if cfg["type"] == "teacher":
        model = build_model("teacher", num_classes=51)
    else:
        model = build_model("student", num_classes=51, width_mult=cfg["width_mult"])

    ckpt = torch.load(cfg["checkpoint"], map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def evaluate(model, loader, device):
    """Return (top1_acc, top5_acc) over the full validation set."""
    y_true, y_prob = [], []

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="  Evaluating", leave=False):
            inputs = inputs.to(device)
            logits = model(inputs)
            probs  = torch.softmax(logits, dim=1)

            y_true.extend(labels.numpy())
            y_prob.extend(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = np.argmax(y_prob, axis=1)

    top1_acc = (y_pred == y_true).mean()

    top5_preds = np.argsort(y_prob, axis=1)[:, -5:]
    top5_acc   = np.mean([y_true[i] in top5_preds[i] for i in range(len(y_true))])

    return top1_acc, top5_acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}\n{'='*55}")

    # Load validation dataset once (shared across all models)
    print("Loading HMDB-51 split-1 validation set ...")
    _, val_ds, _ = build_dataset(
        dataset_type   = "video",
        data_dir       = DATA_DIR,
        annotation_dir = ANNOTATION_DIR,
        split          = SPLIT,
        num_frames     = NUM_FRAMES,
        frame_size     = FRAME_SIZE,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = BATCH_SIZE,
        shuffle     = False,
        num_workers = 0,
        pin_memory  = (device.type == "cuda"),
    )
    print(f"Validation samples: {len(val_ds)}\n{'='*55}")

    results = []

    for cfg in MODELS:
        print(f"\nModel: {cfg['name']}")
        print(f"  Checkpoint : {cfg['checkpoint']}")

        if not os.path.isfile(cfg["checkpoint"]):
            print(f"  [SKIP] Checkpoint not found: {cfg['checkpoint']}")
            results.append((cfg["name"], None, None))
            continue

        model = load_model(cfg, device)
        top1, top5 = evaluate(model, val_loader, device)

        print(f"  Top-1 Accuracy : {top1*100:.2f}%")
        print(f"  Top-5 Accuracy : {top5*100:.2f}%")
        results.append((cfg["name"], top1, top5))

        # Free GPU / CPU memory before next model
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*55}")
    print("SUMMARY")
    print(f"{'='*55}")
    header = f"{'Model':<40} {'Top-1':>8} {'Top-5':>8}"
    print(header)
    print("-" * len(header))
    for name, t1, t5 in results:
        t1_str = f"{t1*100:.2f}%" if t1 is not None else "  SKIP"
        t5_str = f"{t5*100:.2f}%" if t5 is not None else "  SKIP"
        print(f"{name:<40} {t1_str:>8} {t5_str:>8}")

    # Save to file
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("Top-1 and Top-5 Accuracy -- HMDB-51 Split 1 Validation\n")
        f.write("=" * 55 + "\n")
        f.write(f"{'Model':<40} {'Top-1':>8} {'Top-5':>8}\n")
        f.write("-" * 55 + "\n")
        for name, t1, t5 in results:
            t1_str = f"{t1*100:.2f}%" if t1 is not None else "  SKIP"
            t5_str = f"{t5*100:.2f}%" if t5 is not None else "  SKIP"
            f.write(f"{name:<40} {t1_str:>8} {t5_str:>8}\n")

    print(f"\nResults saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
