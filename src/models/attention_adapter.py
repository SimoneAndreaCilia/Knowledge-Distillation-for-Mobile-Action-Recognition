# -*- coding: utf-8 -*-
"""
Attention Adapters for Hint-Based Knowledge Distillation
=========================================================

Provides lightweight ``1×1×1`` convolutional adapters that project the
Student's intermediate feature maps into the Teacher's feature space for
hint-based KD (FitNets, Romero et al. 2015).

**Why adapters are needed**:
  Teacher and Student have different channel counts *and* spatial resolutions
  at matching intermediate layers:

  +-----------+---------+---------+-------------------+-------------------+
  | Pair      | T key   | S key   | Teacher shape     | Student shape     |
  +===========+=========+=========+===================+===================+
  | Pair A    | layer2  | stage_2 | (B, 512, 8,28,28) | (B, 32, 8,14,14) |
  | Pair B    | layer3  | stage_4 | (B,1024, 4,14,14) | (B, 96, 4, 7, 7) |
  +-----------+---------+---------+-------------------+-------------------+

  The adapter:
    1. Uses a ``1×1×1 Conv3d`` to project Student channels → Teacher channels.
    2. Uses ``trilinear`` interpolation to align spatial/temporal dims.

**When to use adapters vs. pure Attention Transfer (AT)**:
  • **AT** (``AttentionTransferLoss``): Parameter-free; matches *where* both
    models attend (spatial attention maps). Lighter, lower risk of overfitting.
  • **Hint-based** (this module + MSE loss): Adds trainable adapter params;
    matches *what* features are computed. Stronger signal, but needs careful
    learning rate tuning.

Usage::

    from src.models.attention_adapter import build_hint_adapters

    adapters = build_hint_adapters(teacher, student, device)
    # adapters is an nn.ModuleDict with keys like "layer2_stage_2"

    # During training, include adapter params in the optimizer:
    optimizer = torch.optim.SGD(
        list(student.parameters()) + list(adapters.parameters()),
        lr=0.01,
    )

    # In the training loop:
    adapted = adapters["layer2_stage_2"](student_feats["stage_2"])
    hint_loss = F.mse_loss(adapted, teacher_feats["layer2"].detach())

References:
    [1] Romero et al., "FitNets: Hints for Thin Deep Nets", ICLR 2015.
    [2] Zagoruyko & Komodakis, "Paying More Attention...", ICLR 2017.
"""

import logging
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Default feature pair mapping: (teacher_key, student_key)
DEFAULT_FEATURE_PAIRS = [
    ("layer2", "stage_2"),   # Teacher layer2 ↔ Student stage 2
    ("layer3", "stage_4"),   # Teacher layer3 ↔ Student stage 4
]


class HintAdapter(nn.Module):
    """Lightweight adapter that projects Student features into Teacher space.

    Performs two operations:
      1. **Channel projection**: ``1×1×1 Conv3d`` maps Student channels
         to Teacher channels.
      2. **Spatial alignment**: ``trilinear`` interpolation resizes the
         Student's ``(T, H, W)`` to match the Teacher's dimensions.

    The adapter is trained jointly with the Student during KD.

    Args:
        student_channels: Number of channels in the Student feature map.
        teacher_channels: Number of channels in the Teacher feature map.
        teacher_spatial: Target ``(T, H, W)`` dimensions from the Teacher.
            If ``None``, no spatial interpolation is performed (assumes
            dimensions already match).
    """

    def __init__(
        self,
        student_channels: int,
        teacher_channels: int,
        teacher_spatial: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        super().__init__()

        # 1×1×1 pointwise conv: student channels → teacher channels
        self.conv = nn.Conv3d(
            student_channels, teacher_channels, kernel_size=1, bias=False
        )
        self.bn = nn.BatchNorm3d(teacher_channels)

        # Target spatial dimensions for interpolation
        self.teacher_spatial = teacher_spatial

        # Initialize with Kaiming
        nn.init.kaiming_normal_(
            self.conv.weight, mode="fan_out", nonlinearity="relu"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project Student features into Teacher feature space.

        Args:
            x: Student features of shape ``(B, C_s, T_s, H_s, W_s)``.

        Returns:
            Adapted features of shape ``(B, C_t, T_t, H_t, W_t)``
            matching the Teacher's dimensions.
        """
        # Channel projection
        out = self.bn(self.conv(x))

        # Spatial interpolation (if dimensions differ)
        if self.teacher_spatial is not None:
            current_spatial = out.shape[2:]
            if current_spatial != self.teacher_spatial:
                out = F.interpolate(
                    out,
                    size=self.teacher_spatial,
                    mode="trilinear",
                    align_corners=False,
                )

        return out


class HintLoss(nn.Module):
    """Hint-based KD loss using adapter-projected features.

    Computes MSE between the adapter-projected Student features and the
    Teacher features across all configured feature pairs.

    This loss DOES add trainable parameters (the adapter convolutions),
    which must be included in the optimizer.

    Args:
        adapters: ``nn.ModuleDict`` of ``HintAdapter`` modules, keyed by
            ``"{teacher_key}_{student_key}"``.
        feature_pairs: List of ``(teacher_key, student_key)`` tuples.
        beta: Scalar weight for the total hint loss.
    """

    def __init__(
        self,
        adapters: nn.ModuleDict,
        feature_pairs: Optional[List[Tuple[str, str]]] = None,
        beta: float = 1.0,
    ) -> None:
        super().__init__()
        self.adapters = adapters
        self.feature_pairs = feature_pairs or DEFAULT_FEATURE_PAIRS
        self.beta = beta

    def forward(
        self,
        student_features: Dict[str, torch.Tensor],
        teacher_features: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute hint-based MSE loss.

        Args:
            student_features: Dict from ``student.get_features()``.
            teacher_features: Dict from ``teacher.get_features()``.

        Returns:
            Tuple of (weighted loss, loss_dict).
        """
        total_loss = torch.tensor(0.0, device=next(iter(
            student_features.values()
        )).device)
        loss_dict: Dict[str, float] = {}

        for t_key, s_key in self.feature_pairs:
            adapter_key = f"{t_key}_{s_key}"
            if adapter_key not in self.adapters:
                continue
            if t_key not in teacher_features or s_key not in student_features:
                continue

            adapter = self.adapters[adapter_key]
            s_feat = student_features[s_key]
            t_feat = teacher_features[t_key].detach()  # No grad through teacher

            # Project student features to teacher space
            adapted = adapter(s_feat)

            # MSE loss
            pair_loss = F.mse_loss(adapted, t_feat)
            total_loss = total_loss + pair_loss

            loss_dict[f"hint_loss_{adapter_key}"] = pair_loss.item()

        weighted_loss = self.beta * total_loss
        loss_dict["hint_loss_total"] = total_loss.item()
        loss_dict["hint_loss_weighted"] = weighted_loss.item()

        return weighted_loss, loss_dict


# ======================================================================
# FACTORY FUNCTION
# ======================================================================

def build_hint_adapters(
    teacher: nn.Module,
    student: nn.Module,
    feature_pairs: Optional[List[Tuple[str, str]]] = None,
    device: Optional[torch.device] = None,
) -> nn.ModuleDict:
    """Auto-detect channel/spatial dimensions and create adapter modules.

    Performs a single dummy forward pass through both models to determine
    the exact shapes of intermediate feature maps, then creates matching
    ``HintAdapter`` modules.

    Args:
        teacher: Teacher model (used to detect feature map shapes).
        student: Student model (used to detect feature map shapes).
        feature_pairs: List of ``(teacher_key, student_key)`` pairs.
            Defaults to ``DEFAULT_FEATURE_PAIRS``.
        device: Device for the dummy forward pass.

    Returns:
        ``nn.ModuleDict`` of ``HintAdapter`` modules, keyed by
        ``"{teacher_key}_{student_key}"``.

    Example::

        adapters = build_hint_adapters(teacher, student, device=device)
        # adapters["layer2_stage_2"] adapts (B,32,8,14,14) → (B,512,8,28,28)
        # adapters["layer3_stage_4"] adapts (B,96,4,7,7) → (B,1024,4,14,14)
    """
    if feature_pairs is None:
        feature_pairs = DEFAULT_FEATURE_PAIRS

    if device is None:
        device = torch.device("cpu")

    # Dummy forward pass to detect feature shapes
    teacher.eval()
    student.eval()
    dummy = torch.randn(1, 3, 16, 112, 112, device=device)

    with torch.no_grad():
        _ = teacher(dummy)
        _ = student(dummy)

    t_feats = teacher.get_features()
    s_feats = student.get_features()

    adapters = nn.ModuleDict()

    for t_key, s_key in feature_pairs:
        if t_key not in t_feats:
            logger.warning(
                f"Teacher key '{t_key}' not in features. "
                f"Available: {list(t_feats.keys())}. Skipping."
            )
            continue
        if s_key not in s_feats:
            logger.warning(
                f"Student key '{s_key}' not in features. "
                f"Available: {list(s_feats.keys())}. Skipping."
            )
            continue

        t_shape = t_feats[t_key].shape   # (1, C_t, T_t, H_t, W_t)
        s_shape = s_feats[s_key].shape   # (1, C_s, T_s, H_s, W_s)

        adapter_key = f"{t_key}_{s_key}"
        adapters[adapter_key] = HintAdapter(
            student_channels=s_shape[1],
            teacher_channels=t_shape[1],
            teacher_spatial=tuple(t_shape[2:]),
        )

        logger.info(
            f"HintAdapter [{adapter_key}]: "
            f"Student {tuple(s_shape[1:])} → Teacher {tuple(t_shape[1:])}"
        )

    adapters = adapters.to(device)
    total_adapter_params = sum(p.numel() for p in adapters.parameters())
    logger.info(f"Total adapter parameters: {total_adapter_params:,}")

    return adapters
