# -*- coding: utf-8 -*-
"""Smoke test: verifies models, losses, checkpointing, profiling, and KD components."""

import os
import shutil
import torch
import logging

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

print("=" * 60)
print("SMOKE TEST: Full Pipeline Verification")
print("=" * 60)

# === Phase 1: Models ===
from src.models.teacher import resnet3d_50
from src.models.student import mobilenet3d

teacher = resnet3d_50(num_classes=51, pretrained=False)
student = mobilenet3d(num_classes=51, width_mult=1.0)

x = torch.randn(2, 3, 16, 112, 112)
t_out = teacher(x)
s_out = student(x)

print(f"\nTeacher: {tuple(x.shape)} -> {tuple(t_out.shape)}, "
      f"{sum(p.numel() for p in teacher.parameters()):,} params")
print(f"Student: {tuple(x.shape)} -> {tuple(s_out.shape)}, "
      f"{sum(p.numel() for p in student.parameters()):,} params")
print(f"Compression: {sum(p.numel() for p in teacher.parameters()) / sum(p.numel() for p in student.parameters()):.1f}x")

# Check features
t_feats = teacher.get_features()
s_feats = student.get_features()
print(f"\nTeacher features: { {k: tuple(v.shape) for k, v in t_feats.items()} }")
print(f"Student features: { {k: tuple(v.shape) for k, v in s_feats.items()} }")

# === Phase 1: Losses ===
from src.training.losses import CrossEntropyLoss, KnowledgeDistillationLoss

targets = torch.randint(0, 51, (2,))
ce = CrossEntropyLoss()
ce_loss, ce_dict = ce(s_out, targets)
print(f"\nCE Loss: {ce_loss.item():.4f}")

# === Phase 2: KD Loss with Temperature Ablation ===
print("\n--- Temperature Ablation ---")
for T in [1, 5, 10, 20]:
    kd = KnowledgeDistillationLoss(temperature=T, alpha=0.3)
    kd_loss, kd_dict = kd(s_out, t_out.detach(), targets)
    print(f"  T={T:2d}: total={kd_dict['total_loss']:.4f} "
          f"ce={kd_dict['ce_loss']:.4f} kd={kd_dict['kd_loss']:.4f}")

# === Phase 2: Attention Transfer Loss ===
from src.training.losses import AttentionTransferLoss

at_loss_fn = AttentionTransferLoss(
    feature_pairs=[("layer2", "stage_2"), ("layer3", "stage_4")]
)
at_loss, at_dict = at_loss_fn(s_feats, t_feats)
print(f"\nAttention Transfer Loss: {at_dict['at_loss_total']:.6f}")
for k, v in at_dict.items():
    if k != "at_loss_total":
        print(f"  {k}: {v:.6f}")

# === Phase 2: Hint Adapters ===
from src.models.attention_adapter import build_hint_adapters, HintLoss

adapters = build_hint_adapters(teacher, student, device=torch.device("cpu"))
print(f"\nHint Adapters created:")
for k, adapter in adapters.items():
    adapter_params = sum(p.numel() for p in adapter.parameters())
    print(f"  [{k}]: {adapter_params:,} params")

# Test hint loss
hint_loss_fn = HintLoss(adapters=adapters, beta=0.5)

# Need fresh forward passes to populate features
_ = teacher(x)
_ = student(x)
t_feats2 = teacher.get_features()
s_feats2 = student.get_features()

hint_loss, hint_dict = hint_loss_fn(s_feats2, t_feats2)
print(f"\nHint Loss (weighted): {hint_dict['hint_loss_weighted']:.4f}")
print(f"Hint Loss (raw):     {hint_dict['hint_loss_total']:.4f}")

# === Checkpoint Manager ===
from src.utils.checkpoint import CheckpointManager

tmpdir = os.path.join(".", "_test_ckpt")
mgr = CheckpointManager(save_dir=tmpdir, max_keep=2)
optimizer = torch.optim.SGD(student.parameters(), lr=0.01)
mgr.save(0, student, optimizer, best_acc=50.0, is_best=True)
ckpt = mgr.load_latest()
print(f"\nCheckpoint: epoch={ckpt['epoch']}, best_acc={ckpt['best_acc']}")
shutil.rmtree(tmpdir)

# === Model Profiling ===
from src.evaluation.profile import profile_model

print("\n--- Profiling ---")
tp = profile_model(teacher, device=torch.device("cpu"), measure_latency_flag=False)
sp = profile_model(student, device=torch.device("cpu"), measure_latency_flag=False)
print(f"Teacher: {tp['param_count']:,} params, {tp['model_size_mb']:.2f} MB")
print(f"Student: {sp['param_count']:,} params, {sp['model_size_mb']:.2f} MB")

# === Model Factory ===
from src.models import build_model

t2 = build_model("teacher", num_classes=51)
s2 = build_model("student", num_classes=51, width_mult=0.75)
print(f"\nbuild_model('teacher'): {sum(p.numel() for p in t2.parameters()):,} params")
print(f"build_model('student', a=0.75): {sum(p.numel() for p in s2.parameters()):,} params")

# === t-SNE utility (import check only, no data needed) ===
from src.utils.visualize import extract_penultimate_features, compute_tsne, plot_tsne
print("\nt-SNE utilities imported successfully")

print("\n" + "=" * 60)
print("ALL SMOKE TESTS PASSED (Phase 1 + Phase 2)")
print("=" * 60)
