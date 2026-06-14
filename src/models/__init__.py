# -*- coding: utf-8 -*-
"""Model factory for creating Teacher and Student architectures.

Usage::

    from src.models import build_model

    teacher = build_model("teacher", num_classes=51, pretrained=True)
    student = build_model("student", num_classes=51, width_mult=1.0)
"""

from typing import Optional

import torch.nn as nn

from .teacher import ResNet3D, resnet3d_50
from .student import MobileNet3D, mobilenet3d

__all__ = [
    "build_model",
    "ResNet3D",
    "resnet3d_50",
    "MobileNet3D",
    "mobilenet3d",
]

# ==================================================================
# MODEL REGISTRY
# ==================================================================
_MODEL_REGISTRY = {
    "teacher": "resnet3d_50",
    "resnet3d_50": "resnet3d_50",
    "student": "mobilenet3d",
    "mobilenet3d": "mobilenet3d",
}


def build_model(
    model_name: str,
    num_classes: int = 51,
    pretrained: bool = False,
    pretrained_source: str = "kinetics",
    pretrained_path: Optional[str] = None,
    width_mult: float = 1.0,
    dropout: float = 0.2,
) -> nn.Module:
    """Factory function to instantiate a model by name.

    Provides a single entry point for creating either the Teacher (3D
    ResNet-50) or Student (3D MobileNet) with consistent configuration.

    Args:
        model_name: One of ``"teacher"``, ``"resnet3d_50"``, ``"student"``,
            or ``"mobilenet3d"``.
        num_classes: Number of output action classes (51 for HMDB-51).
        pretrained: If ``True``, load pre-trained weights.
        pretrained_source: Source of pre-trained weights for the teacher.
            ``"kinetics"`` (recommended) or ``"imagenet"`` (fallback).
            Ignored for the student.
        pretrained_path: Path to the Kinetics ``.pth`` checkpoint file.
            Required when ``pretrained_source="kinetics"`` and
            ``pretrained=True``. Ignored for the student.
        width_mult: Width multiplier for the student model. Controls the
            number of channels in each layer. Default 1.0. Ignored for
            the teacher.
        dropout: Dropout probability before the final classifier.

    Returns:
        An ``nn.Module`` ready for training or evaluation.

    Raises:
        ValueError: If ``model_name`` is not recognized.
    """
    canonical = _MODEL_REGISTRY.get(model_name.lower())
    if canonical is None:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available: {list(_MODEL_REGISTRY.keys())}"
        )

    if canonical == "resnet3d_50":
        return resnet3d_50(
            num_classes=num_classes,
            pretrained=pretrained,
            pretrained_source=pretrained_source,
            pretrained_path=pretrained_path,
            dropout=dropout,
        )
    elif canonical == "mobilenet3d":
        return mobilenet3d(
            num_classes=num_classes,
            width_mult=width_mult,
            dropout=dropout,
        )
    else:
        raise ValueError(f"No builder for canonical model: {canonical}")

