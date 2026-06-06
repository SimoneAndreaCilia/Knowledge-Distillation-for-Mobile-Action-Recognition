# -*- coding: utf-8 -*-
"""Smoke test: verifies models, losses, checkpointing, profiling, KD, and comparison figures."""

import os
import sys
import shutil

# Add the project root to sys.path so we can import 'src' easily
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

print("=" * 60)
print("SMOKE TEST: Full Pipeline Verification (Phase 1+2+3)")
print("=" * 60)

# === Phase 1: Models ===
from src.models.teacher import resnet3d_50
from src.models.student import mobilenet3d

teacher = resnet3d_50(num_classes=51, pretrained=False)
student = mobilenet3d(num_classes=51, width_mult=1.0)

x = torch.randn(2, 3, 16, 112, 112)
t_out = teacher(x)
s_out = student(x)

t_params = sum(p.numel() for p in teacher.parameters())
s_params = sum(p.numel() for p in student.parameters())
print(f"\nTeacher: {t_params:,} params | Student: {s_params:,} params")
print(f"Compression: {t_params / s_params:.1f}x")

t_feats = teacher.get_features()
s_feats = student.get_features()

# === Phase 1: Losses ===
from src.training.losses import CrossEntropyLoss, KnowledgeDistillationLoss

targets = torch.randint(0, 51, (2,))
ce = CrossEntropyLoss()
ce_loss, ce_dict = ce(s_out, targets)
print(f"\nCE Loss: {ce_loss.item():.4f}")

# === Phase 2: KD Losses ===
print("\n--- Temperature Ablation ---")
for T in [1, 5, 10, 20]:
    kd = KnowledgeDistillationLoss(temperature=T, alpha=0.3)
    kd_loss, kd_dict = kd(s_out, t_out.detach(), targets)
    print(f"  T={T:2d}: total={kd_dict['total_loss']:.4f} "
          f"ce={kd_dict['ce_loss']:.4f} kd={kd_dict['kd_loss']:.4f}")

# AT + Hints
from src.training.losses import AttentionTransferLoss
at_loss_fn = AttentionTransferLoss()
at_loss, at_dict = at_loss_fn(s_feats, t_feats)
print(f"\nAT Loss: {at_dict['at_loss_total']:.6f}")

from src.models.attention_adapter import build_hint_adapters, HintLoss
adapters = build_hint_adapters(teacher, student, device=torch.device("cpu"))
hint_loss_fn = HintLoss(adapters=adapters, beta=0.5)
_ = teacher(x); _ = student(x)
hint_loss, hint_dict = hint_loss_fn(student.get_features(), teacher.get_features())
print(f"Hint Loss: {hint_dict['hint_loss_weighted']:.4f}")

# === Phase 1: Checkpoint ===
from src.utils.checkpoint import CheckpointManager
tmpdir = os.path.join(".", "_test_ckpt")
mgr = CheckpointManager(save_dir=tmpdir, max_keep=2)
optimizer = torch.optim.SGD(student.parameters(), lr=0.01)
mgr.save(0, student, optimizer, best_acc=50.0, is_best=True)
ckpt = mgr.load_latest()
print(f"\nCheckpoint: epoch={ckpt['epoch']}, best_acc={ckpt['best_acc']}")
shutil.rmtree(tmpdir)

# === Phase 1: Profiling ===
from src.evaluation.profile import profile_model
tp = profile_model(teacher, device=torch.device("cpu"), measure_latency_flag=False)
sp = profile_model(student, device=torch.device("cpu"), measure_latency_flag=False)

# === Phase 3: Comparison Figures ===
print("\n--- Phase 3: Comparison Figures ---")

import matplotlib
matplotlib.use("Agg")

from src.evaluation.comparison import (
    plot_accuracy_comparison,
    plot_deployment_dashboard,
    plot_per_class_heatmap,
    plot_compression_scatter,
    plot_temperature_ablation,
    save_all_figures,
)

# Build mock results for figure generation
mock_results = {
    "Teacher (ResNet3D-50)": {
        "top1_acc": 65.2,
        "param_count": t_params,
        "model_size_mb": tp["model_size_mb"],
        "latency_ms": 45.3,
        "per_class_acc": list(np.random.uniform(40, 90, 51)),
    },
    "Student Baseline": {
        "top1_acc": 42.1,
        "param_count": s_params,
        "model_size_mb": sp["model_size_mb"],
        "latency_ms": 8.7,
        "per_class_acc": list(np.random.uniform(20, 70, 51)),
    },
    "Student Distilled": {
        "top1_acc": 55.3,
        "param_count": s_params,
        "model_size_mb": sp["model_size_mb"],
        "latency_ms": 8.7,
        "per_class_acc": list(np.random.uniform(30, 80, 51)),
    },
}

# Test individual figures
fig1 = plot_accuracy_comparison(mock_results)
print(f"  [1/5] Accuracy bar chart: OK ({type(fig1).__name__})")

fig2 = plot_deployment_dashboard(mock_results)
print(f"  [2/5] Deployment dashboard: OK ({type(fig2).__name__})")

fig3 = plot_per_class_heatmap(mock_results)
print(f"  [3/5] Per-class heatmap: OK ({type(fig3).__name__})")

fig4 = plot_compression_scatter(mock_results)
print(f"  [4/5] Compression scatter: OK ({type(fig4).__name__})")

temp_results = {1: 48.5, 5: 55.3, 10: 53.1, 20: 50.2}
fig5 = plot_temperature_ablation(
    temp_results, baseline_acc=42.1, teacher_acc=65.2
)
print(f"  [5/5] Temperature ablation: OK ({type(fig5).__name__})")

# Test save_all_figures
test_output_dir = os.path.join(".", "_test_figures")
figures = save_all_figures(
    mock_results, output_dir=test_output_dir, temp_results=temp_results
)
saved_files = os.listdir(test_output_dir)
print(f"\n  Saved {len(saved_files)} figures to {test_output_dir}/: {saved_files}")
shutil.rmtree(test_output_dir)

import matplotlib.pyplot as plt
plt.close("all")

# === Phase 3: Comparison table ===
from src.evaluation.evaluate import print_comparison_table
table = print_comparison_table(mock_results)
print(table)

# === Phase 3: TensorBoard figure logging ===
from src.utils.metrics import TensorBoardLogger
test_log_dir = os.path.join(".", "_test_tb")
tb = TensorBoardLogger(log_dir=test_log_dir, experiment_name="test")
tb.log_training_epoch(0, 1.5, 45.0, 1.2, 50.0, 0.01)
tb.log_kd_losses(0, 1.2, 0.8, 0.4)
tb.log_deployment_metrics("student", s_params, sp["model_size_mb"], 8.7)
tb.close()
shutil.rmtree(test_log_dir)
print("\nTensorBoard logging: OK")

# === Import chain ===
from src.utils.visualize import extract_penultimate_features, compute_tsne, plot_tsne
print("t-SNE utilities: OK")

print("\n" + "=" * 60)
print("ALL SMOKE TESTS PASSED (Phase 1 + Phase 2 + Phase 3)")
print("=" * 60)
