# -*- coding: utf-8 -*-
"""Model configuration registry — typed, frozen descriptors for every checkpoint.

Replaces the raw ``Dict[str, Any]`` dictionaries ``MODELS_MAIN`` and
``MODELS_ADVANCED`` with strongly-typed ``ModelConfig`` dataclasses.

Adding a new checkpoint requires only a new ``ModelConfig`` entry — no
other file needs to change (Open/Closed Principle).

Usage::

    from src.config.model_configs import MODELS_MAIN, MODELS_ADVANCED, ModelConfig
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from .settings import get_settings
from src.domain.models.model_metadata import ModelMetadata as ModelConfig
from src.i18n.keys import TranslationKey


def _ckpt(relative: str) -> Path:
    """Resolve a checkpoint path relative to the checkpoints directory."""
    return get_settings().checkpoints_dir / relative


# ---------------------------------------------------------------------------
# Main models — shown by default in the UI
# ---------------------------------------------------------------------------

MODELS_MAIN: Dict[str, ModelConfig] = {
    cfg.name: cfg
    for cfg in [
        ModelConfig(
            name="Teacher (ResNet3D-50) — 62.94%",
            checkpoint=_ckpt("teacher/best_model.pth"),
            model_name="teacher",
            width_mult=1.0,
            params_m=46.30,
            accuracy=62.94,
            size_mb=176.6,
            description_key=TranslationKey.DESC_TEACHER,
        ),
        ModelConfig(
            name="Student Baseline — 20.13%",
            checkpoint=_ckpt("student_baseline/best_model.pth"),
            model_name="student",
            width_mult=1.0,
            params_m=2.42,
            accuracy=20.13,
            size_mb=9.23,
            description_key=TranslationKey.DESC_STUDENT,
        ),
        ModelConfig(
            name="Student Distilled (KD T=10) — 29.15%",
            checkpoint=_ckpt("distilled_T10/best_model.pth"),
            model_name="student",
            width_mult=1.0,
            params_m=2.42,
            accuracy=29.15,
            size_mb=9.23,
            description_key=TranslationKey.DESC_STUDENT_DISTILLED_T10,
        ),
        ModelConfig(
            name="Student Distilled + AT — 47.19%",
            checkpoint=_ckpt("distilled_AT_T10_seed1234/best_model.pth"),
            model_name="student",
            width_mult=1.0,
            params_m=2.42,
            accuracy=47.19,
            size_mb=9.23,
            description_key=TranslationKey.DESC_STUDENT_AT,
        ),
    ]
}

# ---------------------------------------------------------------------------
# Advanced / ablation models — shown when "show all variants" is toggled
# ---------------------------------------------------------------------------

MODELS_ADVANCED: Dict[str, ModelConfig] = {
    cfg.name: cfg
    for cfg in [
        ModelConfig(
            name="Student KD T=1",
            checkpoint=_ckpt("distilled_T1/best_model.pth"),
            model_name="student",
            width_mult=1.0,
            params_m=2.42,
            accuracy=None,
            size_mb=9.23,
            description_key=TranslationKey.DESC_STUDENT_KD_T1,
        ),
        ModelConfig(
            name="Student KD T=5",
            checkpoint=_ckpt("distilled_T5/best_model.pth"),
            model_name="student",
            width_mult=1.0,
            params_m=2.42,
            accuracy=None,
            size_mb=9.23,
            description_key=TranslationKey.DESC_STUDENT_KD_T5,
        ),
        ModelConfig(
            name="Student KD T=20",
            checkpoint=_ckpt("distilled_T20/best_model.pth"),
            model_name="student",
            width_mult=1.0,
            params_m=2.42,
            accuracy=None,
            size_mb=9.23,
            description_key=TranslationKey.DESC_STUDENT_KD_T20,
        ),
        ModelConfig(
            name="Student KD + AT (seed 42)",
            checkpoint=_ckpt("distilled_AT_T10/best_model.pth"),
            model_name="student",
            width_mult=1.0,
            params_m=2.42,
            accuracy=None,
            size_mb=9.23,
            description_key=TranslationKey.DESC_STUDENT_AT_SEED42,
        ),
        ModelConfig(
            name="Student KD + AT (β=10)",
            checkpoint=_ckpt("distilled_AT_T10_beta10/best_model.pth"),
            model_name="student",
            width_mult=1.0,
            params_m=2.42,
            accuracy=40.00,
            size_mb=9.23,
            description_key=TranslationKey.DESC_STUDENT_AT_BETA10,
        ),
        ModelConfig(
            name="Student KD + AT (β=100)",
            checkpoint=_ckpt("distilled_AT_T10_beta100/best_model.pth"),
            model_name="student",
            width_mult=1.0,
            params_m=2.42,
            accuracy=None,
            size_mb=9.23,
            description_key=TranslationKey.DESC_STUDENT_AT_BETA100,
        ),
        ModelConfig(
            name="Student KD T=10 (α=1.5)",
            checkpoint=_ckpt("distilled_final_T10_w1.5/best_model.pth"),
            model_name="student",
            width_mult=1.5,
            params_m=4.70,
            accuracy=None,
            size_mb=40.2,
            description_key=TranslationKey.DESC_STUDENT_KD_T10_W1_5,
        ),
    ]
}
