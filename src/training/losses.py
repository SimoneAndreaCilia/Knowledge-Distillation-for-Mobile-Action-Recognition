# -*- coding: utf-8 -*-
"""
Loss Functions for Knowledge Distillation
==========================================

Complete loss function library for the KD pipeline:

  • **CrossEntropyLoss** — wrapped CE for baseline training (Phase 1).
  • **KnowledgeDistillationLoss** — Hinton et al. (2015) KD loss combining
    hard-label CE with temperature-scaled KL divergence (Phase 2).
  • **AttentionTransferLoss** — Zagoruyko & Komodakis (2017) attention map
    matching between teacher and student intermediate layers (Phase 2 adv.).

Design rationale:
  All loss functions share a consistent interface and return a dict of
  component losses alongside the combined scalar loss.  This enables
  structured TensorBoard logging of each loss term.

Usage (Baseline)::

    criterion = CrossEntropyLoss(label_smoothing=0.0)
    loss, loss_dict = criterion(logits, targets)

Usage (KD)::

    criterion = KnowledgeDistillationLoss(temperature=5.0, alpha=0.3)
    loss, loss_dict = criterion(student_logits, teacher_logits, targets)

Usage (Attention Transfer)::

    at_criterion = AttentionTransferLoss(feature_pairs=[("layer2", "stage_2")])
    at_loss, at_dict = at_criterion(student_features, teacher_features)
"""

import logging
from typing import Tuple, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class CrossEntropyLoss(nn.Module):
    """Wrapped Cross-Entropy loss with consistent interface.

    Thin wrapper around ``nn.CrossEntropyLoss`` that returns both the
    scalar loss and a dictionary of named components for logging.

    Args:
        label_smoothing: Label smoothing factor in [0, 1). Default 0.0
            (no smoothing). Use 0.1 for mild regularization.
    """

    def __init__(self, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.label_smoothing = label_smoothing

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute cross-entropy loss.

        Args:
            logits: Model output logits of shape ``(B, C)``.
            targets: Ground-truth class indices of shape ``(B,)``.

        Returns:
            Tuple of:
              - ``loss``: Scalar loss tensor (for backprop).
              - ``loss_dict``: Dict with key ``"ce_loss"`` for logging.
        """
        loss = self.criterion(logits, targets)

        loss_dict = {
            "ce_loss": loss.item(),
        }

        return loss, loss_dict


class KnowledgeDistillationLoss(nn.Module):
    """Knowledge Distillation loss combining hard and soft targets.

    Implements the Hinton et al. (2015) KD loss:

    .. math::

        L_{KD} = \\alpha \\cdot L_{CE}(y, \\hat{y})
               + (1 - \\alpha) \\cdot T^2 \\cdot D_{KL}(\\sigma(z_t/T) \\| \\sigma(z_s/T))

    Where:
      - :math:`L_{CE}` is standard cross-entropy with ground truth (hard labels).
      - :math:`D_{KL}` is KL divergence between teacher and student softened
        distributions.
      - :math:`T` is the temperature (higher → softer distributions → more
        inter-class knowledge transferred).
      - :math:`\\alpha` balances hard vs. soft targets.
      - :math:`T^2` scaling compensates for the reduced gradient magnitude at
        higher temperatures (see Hinton et al., Section 2).

    Args:
        temperature: Softmax temperature for logit smoothing. Higher values
            produce softer probability distributions, revealing more about
            inter-class similarities. Typical range: [1, 20].
        alpha: Balance factor between CE loss (hard labels) and KD loss
            (soft labels). ``alpha=1.0`` means pure CE; ``alpha=0.0``
            means pure KD.
        label_smoothing: Optional label smoothing for the CE component.
    """

    def __init__(
        self,
        temperature: float = 5.0,
        alpha: float = 0.3,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        # KLDivLoss expects log-probabilities as input, probabilities as target
        self.kl_criterion = nn.KLDivLoss(reduction="batchmean")

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the combined KD loss.

        Args:
            student_logits: Raw logits from the student, shape ``(B, C)``.
            teacher_logits: Raw logits from the teacher, shape ``(B, C)``.
                The teacher should be in ``eval()`` mode and gradients
                should NOT flow through it.
            targets: Ground-truth class indices, shape ``(B,)``.

        Returns:
            Tuple of:
              - ``loss``: Combined scalar loss tensor for backprop.
              - ``loss_dict``: Dict with keys ``"total_loss"``,
                ``"ce_loss"``, ``"kd_loss"`` for structured logging.
        """
        # ---- Component 1: Hard-label Cross-Entropy ----
        ce_loss = self.ce_criterion(student_logits, targets)

        # ---- Component 2: Soft-label KL Divergence ----
        # Temperature scaling: divide logits by T before softmax
        # Student: log_softmax (KLDivLoss input requirement)
        # Teacher: softmax (KLDivLoss target requirement)
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)

        # KL divergence with T² scaling
        # The T² factor ensures gradients from the soft targets are
        # comparable in magnitude to those from hard targets regardless
        # of the temperature value.
        kd_loss = self.kl_criterion(student_soft, teacher_soft) * (
            self.temperature ** 2
        )

        # ---- Combined loss ----
        total_loss = self.alpha * ce_loss + (1 - self.alpha) * kd_loss

        loss_dict = {
            "total_loss": total_loss.item(),
            "ce_loss": ce_loss.item(),
            "kd_loss": kd_loss.item(),
        }

        return total_loss, loss_dict


# ======================================================================
# ATTENTION TRANSFER LOSS (Zagoruyko & Komodakis, ICLR 2017)
# ======================================================================

class AttentionTransferLoss(nn.Module):
    """Attention Transfer loss for intermediate feature map matching.

    Transfers spatial attention patterns from the Teacher to the Student
    without requiring the feature maps to have the same number of channels.

    **Method**: For each intermediate activation tensor ``A`` of shape
    ``(B, C, T, H, W)``, the attention map is computed as the L2 norm
    across the channel dimension:

    .. math::

        Q(A) = \\sum_{c=1}^{C} |A_c|^2

    yielding a tensor of shape ``(B, T, H, W)`` that captures *where*
    the network is attending, irrespective of *what* features it uses.
    The maps are L2-normalized and compared via mean squared error.

    When spatial or temporal dimensions differ between Teacher and Student
    feature maps, the Student's attention map is interpolated to match
    the Teacher's dimensions.

    This loss adds **zero trainable parameters** — it operates purely
    on the already-computed intermediate activations.

    Args:
        feature_pairs: List of ``(teacher_key, student_key)`` tuples
            specifying which intermediate feature maps to match.
            Keys must correspond to those returned by the model's
            ``get_features()`` method.
        normalize: If ``True``, L2-normalize attention maps before
            computing the loss. Recommended for stable training.
        p: Exponent for computing the attention map. Default 2
            (sum of squared activations).

    Example::

        at_loss_fn = AttentionTransferLoss(
            feature_pairs=[("layer2", "stage_2"), ("layer3", "stage_4")]
        )
        student_feats = student.get_features()
        teacher_feats = teacher.get_features()
        at_loss, at_dict = at_loss_fn(student_feats, teacher_feats)
    """

    def __init__(
        self,
        feature_pairs: Optional[List[Tuple[str, str]]] = None,
        normalize: bool = True,
        p: int = 2,
    ) -> None:
        super().__init__()

        # Default pairs: Teacher layer2/layer3 ↔ Student stage_2/stage_4
        if feature_pairs is None:
            feature_pairs = [
                ("layer2", "stage_2"),
                ("layer3", "stage_4"),
            ]

        self.feature_pairs = feature_pairs
        self.normalize = normalize
        self.p = p

    @staticmethod
    def _compute_attention_map(
        features: torch.Tensor, p: int = 2
    ) -> torch.Tensor:
        """Compute a spatial attention map from an activation tensor.

        Args:
            features: Activation tensor of shape ``(B, C, T, H, W)``.
            p: Exponent for aggregation across channels.

        Returns:
            Attention map of shape ``(B, T*H*W)`` — flattened and
            optionally normalized.
        """
        # Sum of |activations|^p across channels → (B, T, H, W)
        attn = torch.pow(features.abs(), p).sum(dim=1)
        # Flatten spatial + temporal → (B, T*H*W)
        attn = attn.view(attn.size(0), -1)
        return attn

    def forward(
        self,
        student_features: Dict[str, torch.Tensor],
        teacher_features: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the attention transfer loss across all feature pairs.

        Args:
            student_features: Dict from ``student.get_features()``.
            teacher_features: Dict from ``teacher.get_features()``.

        Returns:
            Tuple of:
              - ``loss``: Scalar AT loss (sum across all pairs).
              - ``loss_dict``: Dict with per-pair and total AT losses.
        """
        total_loss = torch.tensor(0.0, device=next(iter(
            student_features.values()
        )).device)

        loss_dict: Dict[str, float] = {}

        for teacher_key, student_key in self.feature_pairs:
            if teacher_key not in teacher_features:
                logger.warning(
                    f"AT: Teacher key '{teacher_key}' not found in features. "
                    f"Available: {list(teacher_features.keys())}"
                )
                continue
            if student_key not in student_features:
                logger.warning(
                    f"AT: Student key '{student_key}' not found in features. "
                    f"Available: {list(student_features.keys())}"
                )
                continue

            t_feat = teacher_features[teacher_key]
            s_feat = student_features[student_key]

            # Compute attention maps
            t_attn = self._compute_attention_map(t_feat.detach(), self.p)
            s_attn = self._compute_attention_map(s_feat, self.p)

            # If spatial dimensions differ, interpolate student's attention
            # map to match teacher's. We reshape back to spatial form,
            # interpolate, then flatten again.
            if s_attn.shape != t_attn.shape:
                # Recover spatial dims from teacher
                t_spatial = t_feat.shape[2:]  # (T, H, W)
                s_spatial = s_feat.shape[2:]  # (T', H', W')

                # Reshape student attn to (B, 1, T', H', W') for interpolation
                s_attn_spatial = s_attn.view(
                    s_feat.size(0), 1, *s_spatial
                )
                s_attn_spatial = F.interpolate(
                    s_attn_spatial,
                    size=t_spatial,
                    mode="trilinear",
                    align_corners=False,
                )
                s_attn = s_attn_spatial.view(s_feat.size(0), -1)

            # L2-normalize attention maps (stabilizes training)
            if self.normalize:
                t_attn = F.normalize(t_attn, p=2, dim=1)
                s_attn = F.normalize(s_attn, p=2, dim=1)

            # MSE between attention maps
            pair_loss = (s_attn - t_attn).pow(2).sum(dim=1).mean()
            total_loss = total_loss + pair_loss

            loss_dict[f"at_loss_{teacher_key}_{student_key}"] = pair_loss.item()

        loss_dict["at_loss_total"] = total_loss.item()
        return total_loss, loss_dict

