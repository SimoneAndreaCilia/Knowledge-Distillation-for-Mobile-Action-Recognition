# -*- coding: utf-8 -*-
"""
Knowledge Distillation for Mobile Action Recognition
=====================================================

Compresses a 3D ResNet-50 (Teacher) into an ultra-lightweight 3D MobileNet
(Student) via knowledge distillation on the HMDB-51 dataset.

Project structure:
    src/
    ├── models/      — Teacher (3D ResNet-50) and Student (3D MobileNet)
    ├── datasets/    — HMDB-51 video loaders and feature loaders
    ├── training/    — Loss functions and training loops
    ├── evaluation/  — Accuracy evaluation and model profiling
    └── utils/       — Seed, checkpointing, metrics, TensorBoard logging
"""

__version__ = "1.0.0"
