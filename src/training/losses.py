# -*- coding: utf-8 -*-
"""
Loss Functions for Knowledge Distillation
==========================================

Phase 1: Standard Cross-Entropy wrapper for baseline training.
Phase 2 (placeholder): Knowledge Distillation loss combining Cross-Entropy
         with KL-Divergence on temperature-scaled soft targets.

Design rationale:
  All loss functions share a consistent interface and return a dict of
  component losses alongside the combined scalar loss.  This enables
  structured TensorBoard logging of each loss term.

Usage (Phase 1 — Baseline)::

    criterion = CrossEntropyLoss(label_smoothing=0.0)
    loss, loss_dict = criterion(logits, targets)
    # loss_dict = {"ce_loss": <value>}

Usage (Phase 2 — KD, placeholder)::

    criterion = KnowledgeDistillationLoss(temperature=5.0, alpha=0.7)
    loss, loss_dict = criterion(student_logits, teacher_logits, targets)
    # loss_dict = {"total_loss": ..., "ce_loss": ..., "kd_loss": ...}
"""

import logging
from typing import Tuple, Dict

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

    Note:
        This class is defined here for Phase 2 readiness. It will NOT be
        used during Phase 1 baseline training.
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
