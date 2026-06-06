# -*- coding: utf-8 -*-
"""Smoke test: verifies models, losses, checkpointing, and profiling."""

import os
import shutil
import torch
import logging

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

print("=" * 60)
print("SMOKE TEST: Model Instantiation & Forward Pass")
print("=" * 60)

# --- Teacher ---
from src.models.teacher import resnet3d_50

teacher = resnet3d_50(num_classes=51, pretrained=False)
x = torch.randn(2, 3, 16, 112, 112)
out = teacher(x)
print(f"\nTeacher (3D ResNet-50):")
print(f"  Input:  {tuple(x.shape)}")
print(f"  Output: {tuple(out.shape)}")
total_t = sum(p.numel() for p in teacher.parameters())
print(f"  Params: {total_t:,}")
size_t = sum(p.numel() * p.element_size() for p in teacher.parameters()) / (1024**2)
print(f"  Size:   {size_t:.2f} MB")

feats = teacher.get_features()
for k, v in feats.items():
    print(f"  Feature [{k}]: {tuple(v.shape)}")

# --- Student ---
from src.models.student import mobilenet3d

student = mobilenet3d(num_classes=51, width_mult=1.0)
out_s = student(x)
print(f"\nStudent (3D MobileNet, alpha=1.0):")
print(f"  Input:  {tuple(x.shape)}")
print(f"  Output: {tuple(out_s.shape)}")
total_s = sum(p.numel() for p in student.parameters())
print(f"  Params: {total_s:,}")
size_s = sum(p.numel() * p.element_size() for p in student.parameters()) / (1024**2)
print(f"  Size:   {size_s:.2f} MB")

feats_s = student.get_features()
for k, v in feats_s.items():
    print(f"  Feature [{k}]: {tuple(v.shape)}")

# --- Compression ratio ---
ratio = total_t / total_s
print(f"\nCompression ratio: {ratio:.1f}x")
print(f"Parameter reduction: {total_t:,} -> {total_s:,}")

# --- Student width_mult=0.5 ---
student_small = mobilenet3d(num_classes=51, width_mult=0.5)
total_ss = sum(p.numel() for p in student_small.parameters())
print(f"\nStudent (alpha=0.5): {total_ss:,} params ({total_t/total_ss:.1f}x compression)")

# --- Loss functions ---
from src.training.losses import CrossEntropyLoss, KnowledgeDistillationLoss

ce = CrossEntropyLoss()
targets = torch.randint(0, 51, (2,))
loss, loss_dict = ce(out_s, targets)
print(f"\nCrossEntropyLoss: {loss.item():.4f} | {loss_dict}")

kd = KnowledgeDistillationLoss(temperature=5.0, alpha=0.3)
kd_loss, kd_dict = kd(out_s, out.detach(), targets)
print(f"KD Loss (T=5): {kd_loss.item():.4f} | {kd_dict}")

# --- Profiling ---
from src.evaluation.profile import profile_model

print("\n--- Teacher Profile ---")
tp = profile_model(teacher, device=torch.device("cpu"), measure_latency_flag=False)
print(tp)
print("\n--- Student Profile ---")
sp = profile_model(student, device=torch.device("cpu"), measure_latency_flag=False)
print(sp)

# --- Checkpoint manager ---
from src.utils.checkpoint import CheckpointManager

tmpdir = os.path.join(".", "_test_ckpt")
mgr = CheckpointManager(save_dir=tmpdir, max_keep=2)
optimizer = torch.optim.SGD(student.parameters(), lr=0.01)
mgr.save(0, student, optimizer, best_acc=50.0, is_best=True)
mgr.save(1, student, optimizer, best_acc=55.0, is_best=True)
ckpt = mgr.load_latest()
epoch = ckpt["epoch"]
best = ckpt["best_acc"]
print(f"\nCheckpoint resume: epoch={epoch}, best_acc={best}")
shutil.rmtree(tmpdir)

# --- Model factory ---
from src.models import build_model

t2 = build_model("teacher", num_classes=51)
s2 = build_model("student", num_classes=51, width_mult=0.75)
print(f"\nbuild_model('teacher'): {sum(p.numel() for p in t2.parameters()):,} params")
print(f"build_model('student', alpha=0.75): {sum(p.numel() for p in s2.parameters()):,} params")

print("\n" + "=" * 60)
print("ALL SMOKE TESTS PASSED")
print("=" * 60)
