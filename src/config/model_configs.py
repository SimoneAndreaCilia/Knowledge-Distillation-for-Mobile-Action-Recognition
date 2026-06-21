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
            description=(
                "3D ResNet-50 fine-tuned from Kinetics-400 weights. "
                "Upper-bound reference for the KD pipeline."
            ),
        ),
        ModelConfig(
            name="Student Baseline — 20.13%",
            checkpoint=_ckpt("student_baseline/best_model.pth"),
            model_name="student",
            width_mult=1.0,
            params_m=2.42,
            accuracy=20.13,
            size_mb=9.23,
            description=(
                "3D MobileNet trained from scratch on HMDB-51. "
                "Lower-bound baseline without any knowledge transfer."
            ),
        ),
        ModelConfig(
            name="Student Distilled (KD T=10) — 29.15%",
            checkpoint=_ckpt("distilled_T10/best_model.pth"),
            model_name="student",
            width_mult=1.0,
            params_m=2.42,
            accuracy=29.15,
            size_mb=9.23,
            description=(
                "3D MobileNet distilled with logit-matching KD at T=10. "
                "Soft targets transfer 'dark knowledge' from the Teacher."
            ),
        ),
        ModelConfig(
            name="Student Distilled + AT — 47.19%",
            checkpoint=_ckpt("distilled_AT_T10_seed1234/best_model.pth"),
            model_name="student",
            width_mult=1.0,
            params_m=2.42,
            accuracy=47.19,
            size_mb=9.23,
            description=(
                "3D MobileNet with KD (T=10) + Attention Transfer (β=1000). "
                "Best student model — bridges 68% of the Teacher-Baseline gap."
            ),
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
            description="KD with T=1 (hard targets only, equivalent to label smoothing).",
        ),
        ModelConfig(
            name="Student KD T=5",
            checkpoint=_ckpt("distilled_T5/best_model.pth"),
            model_name="student",
            width_mult=1.0,
            params_m=2.42,
            accuracy=None,
            size_mb=9.23,
            description="KD with T=5 (moderately softened distributions).",
        ),
        ModelConfig(
            name="Student KD T=20",
            checkpoint=_ckpt("distilled_T20/best_model.pth"),
            model_name="student",
            width_mult=1.0,
            params_m=2.42,
            accuracy=None,
            size_mb=9.23,
            description="KD with T=20 (highly softened, near-uniform distributions).",
        ),
        ModelConfig(
            name="Student KD + AT (seed 42)",
            checkpoint=_ckpt("distilled_AT_T10/best_model.pth"),
            model_name="student",
            width_mult=1.0,
            params_m=2.42,
            accuracy=None,
            size_mb=9.23,
            description="KD + AT (β=1000) with seed 42 (original run).",
        ),
        ModelConfig(
            name="Student KD + AT (β=10)",
            checkpoint=_ckpt("distilled_AT_T10_beta10/best_model.pth"),
            model_name="student",
            width_mult=1.0,
            params_m=2.42,
            accuracy=40.00,
            size_mb=9.23,
            description="KD + AT with reduced β=10 (balanced loss weighting).",
        ),
        ModelConfig(
            name="Student KD + AT (β=100)",
            checkpoint=_ckpt("distilled_AT_T10_beta100/best_model.pth"),
            model_name="student",
            width_mult=1.0,
            params_m=2.42,
            accuracy=None,
            size_mb=9.23,
            description="KD + AT with intermediate β=100 (ablation).",
        ),
        ModelConfig(
            name="Student KD T=10 (α=1.5)",
            checkpoint=_ckpt("distilled_final_T10_w1.5/best_model.pth"),
            model_name="student",
            width_mult=1.5,
            params_m=4.70,
            accuracy=None,
            size_mb=40.2,
            description=(
                "Wider student (α=1.5, ~4.7M params) with KD T=10. "
                "~7× compression vs Teacher."
            ),
        ),
    ]
}
