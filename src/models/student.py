# -*- coding: utf-8 -*-
"""
3D MobileNet — Student Model
==============================

An ultra-lightweight 3D video architecture built from scratch, designed
as the **Student** for knowledge distillation. Based on MobileNetV2's
inverted residual design (Sandler et al., 2018) inflated to the
spatiotemporal domain.

Architecture highlights:
  • **3D Inverted Residual blocks**: expand (1×1×1) → depthwise (3×3×3) →
    project (1×1×1), with ReLU6 activations and residual connections.
  • **Configurable width multiplier** (α): scales channel counts across all
    layers.  α=1.0 yields ~2.2M params; α=0.5 yields ~0.7M.
  • **Progressive downsampling**:
      - Spatial:  112 → 56 → 28 → 14 → 7 → 4  (5 reductions via stem + stages)
      - Temporal: 16 → 8 → 4                    (2 reductions in stages 2 & 4)
  • **Intermediate hooks**: exposes feature maps after stages 3 and 5 for
    attention-transfer knowledge distillation (Phase 2).
  • **Linear bottleneck**: the projection (1×1×1) conv uses *no* activation,
    preserving information in the low-dimensional bottleneck.

Expected input: ``(B, 3, 16, 112, 112)``
Parameter count: ~2.2M (α=1.0), target 5–10× smaller than Teacher.

Stage table (α=1.0)::

    Stage  | Expansion | Out Ch | Blocks | Stride (T,H,W) | Output Shape
    -------+-----------+--------+--------+-----------------+------------------
    Stem   |     —     |   32   |   —    |  (1, 2, 2)      | (B,  32, 16, 56, 56)
      1    |     1     |   16   |   1    |  (1, 1, 1)      | (B,  16, 16, 56, 56)
      2    |     6     |   24   |   2    |  (2, 2, 2)      | (B,  24,  8, 28, 28)
      3    |     6     |   32   |   3    |  (1, 2, 2)      | (B,  32,  8, 14, 14)
      4    |     6     |   64   |   4    |  (2, 2, 2)      | (B,  64,  4,  7,  7)
      5    |     6     |   96   |   3    |  (1, 1, 1)      | (B,  96,  4,  7,  7)
      6    |     6     |  160   |   3    |  (1, 2, 2)      | (B, 160,  4,  4,  4)
      7    |     6     |  320   |   1    |  (1, 1, 1)      | (B, 320,  4,  4,  4)
    Head   |     —     | 1280   |   —    |     —           | (B, 1280, 1, 1, 1)

References:
    [1] Sandler et al., "MobileNetV2", CVPR 2018.
    [2] Kopuklu et al., "Resource Efficient 3D CNNs", ICCV-W 2019.
"""

import logging
import math
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ======================================================================
# CONFIGURATION TABLE
# ======================================================================
# Each entry: (expansion_factor, output_channels, num_blocks,
#              temporal_stride, spatial_stride)
# Stride is applied only to the FIRST block in each stage.

MOBILENET3D_STAGE_CONFIG = [
    #  t,  c,  n, s_t, s_s
    (1,   16,  1,  1,   1),   # Stage 1
    (6,   24,  2,  2,   2),   # Stage 2 — temporal + spatial downsample
    (6,   32,  3,  1,   2),   # Stage 3 — spatial downsample only
    (6,   64,  4,  2,   2),   # Stage 4 — temporal + spatial downsample
    (6,   96,  3,  1,   1),   # Stage 5 — no downsample
    (6,  160,  3,  1,   2),   # Stage 6 — spatial downsample only
    (6,  320,  1,  1,   1),   # Stage 7 — no downsample
]

# Stages whose output is exposed for attention-transfer KD (0-indexed)
ATTENTION_HOOK_STAGES = [2, 4]  # After stages 3 and 5 (0-indexed as 2, 4)


# ======================================================================
# BUILDING BLOCKS
# ======================================================================

def _make_divisible(value: float, divisor: int = 8) -> int:
    """Round a channel count to the nearest multiple of ``divisor``.

    Ensures all channel counts are hardware-friendly (multiples of 8
    or 16 for efficient memory alignment on GPUs/NPUs).

    Args:
        value: The raw channel count (may be non-integer after scaling).
        divisor: The alignment granularity.

    Returns:
        Rounded channel count (at least ``divisor``).
    """
    new_value = max(divisor, int(value + divisor / 2) // divisor * divisor)
    # Ensure rounding doesn't reduce by more than 10%
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value


class InvertedResidual3D(nn.Module):
    """3D Inverted Residual Bottleneck block (MobileNetV2-style).

    Structure (with expansion > 1)::

        x ─→ [1×1×1 Conv → BN → ReLU6]     (expand channels)
          ─→ [3×3×3 DWConv → BN → ReLU6]   (depthwise spatiotemporal)
          ─→ [1×1×1 Conv → BN]              (project/compress — NO activation)
          ─→ + residual (if applicable)

    When ``expansion == 1``, the first 1×1×1 expansion conv is omitted
    (the input already has the right channel count).

    The **linear bottleneck** (no activation on the projection) is critical:
    applying ReLU to the low-dimensional projection destroys information
    (Sandler et al., 2018, Section 3.2).

    Residual connections are applied only when:
      1. Input and output channel counts match, AND
      2. Stride is (1, 1, 1) (no spatial/temporal downsampling).

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels (bottleneck width).
        expansion: Channel expansion factor for the hidden dimension.
        stride: Spatiotemporal stride as ``(t, h, w)``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion: int = 6,
        stride: Tuple[int, int, int] = (1, 1, 1),
    ) -> None:
        super().__init__()

        hidden_dim = in_channels * expansion
        self.use_residual = (stride == (1, 1, 1) and in_channels == out_channels)

        layers: List[nn.Module] = []

        # ---- Expansion phase (1×1×1 pointwise) ----
        if expansion != 1:
            layers.extend([
                nn.Conv3d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(inplace=True),
            ])

        # ---- Depthwise phase (3×3×3 grouped conv, groups=hidden_dim) ----
        layers.extend([
            nn.Conv3d(
                hidden_dim, hidden_dim,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=hidden_dim,   # Depthwise: each channel gets its own filter
                bias=False,
            ),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU6(inplace=True),
        ])

        # ---- Projection phase (1×1×1 pointwise, LINEAR — no activation) ----
        layers.extend([
            nn.Conv3d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels),
            # NO activation here — linear bottleneck preserves information
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional residual connection.

        Args:
            x: Input tensor of shape ``(B, C_in, T, H, W)``.

        Returns:
            Output tensor of shape ``(B, C_out, T', H', W')``.
        """
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)


# ======================================================================
# FULL NETWORK
# ======================================================================

class MobileNet3D(nn.Module):
    """Ultra-lightweight 3D MobileNet for video action recognition (Student).

    Constructed from ``InvertedResidual3D`` blocks following the MobileNetV2
    topology, inflated to handle spatiotemporal (video) inputs.

    The network exposes intermediate features for attention-transfer KD
    via the ``get_features()`` method.

    Args:
        num_classes: Number of output action classes (51 for HMDB-51).
        width_mult: Channel width multiplier (α). Scales all intermediate
            channel counts. Use 1.0 for the standard architecture (~2.2M
            params), 0.5 for a smaller variant (~0.7M).
        dropout: Dropout probability before the final classifier.
        stage_config: Stage configuration table. Defaults to the standard
            7-stage MobileNetV2 layout.
    """

    def __init__(
        self,
        num_classes: int = 51,
        width_mult: float = 1.0,
        dropout: float = 0.2,
        stage_config: Optional[List[Tuple[int, int, int, int, int]]] = None,
    ) -> None:
        super().__init__()
        self.width_mult = width_mult

        if stage_config is None:
            stage_config = MOBILENET3D_STAGE_CONFIG

        # ---- Stem: initial 3D convolution ----
        # (B, 3, 16, 112, 112) → (B, 32, 16, 56, 56)
        stem_channels = _make_divisible(32 * width_mult)
        self.stem = nn.Sequential(
            nn.Conv3d(
                3, stem_channels,
                kernel_size=3,
                stride=(1, 2, 2),
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(stem_channels),
            nn.ReLU6(inplace=True),
        )

        # ---- Inverted residual stages ----
        self.stages = nn.ModuleList()
        in_channels = stem_channels

        for stage_idx, (t, c, n, s_t, s_s) in enumerate(stage_config):
            out_channels = _make_divisible(c * width_mult)
            stage_blocks: List[nn.Module] = []

            for block_idx in range(n):
                # Only the first block in each stage applies the stride
                if block_idx == 0:
                    stride = (s_t, s_s, s_s)
                else:
                    stride = (1, 1, 1)

                stage_blocks.append(
                    InvertedResidual3D(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        expansion=t,
                        stride=stride,
                    )
                )
                in_channels = out_channels

            self.stages.append(nn.Sequential(*stage_blocks))

        # ---- Final convolution (1×1×1 to expand channels) ----
        final_channels = _make_divisible(1280 * width_mult) if width_mult > 1.0 else 1280
        self.final_conv = nn.Sequential(
            nn.Conv3d(in_channels, final_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(final_channels),
            nn.ReLU6(inplace=True),
        )

        # ---- Classification head ----
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(final_channels, num_classes)

        # ---- Weight initialization ----
        self._initialize_weights()

        # ---- Feature cache for attention transfer ----
        self._features: Dict[str, torch.Tensor] = {}

    def _initialize_weights(self) -> None:
        """Initialize weights using best practices for mobile networks.

        Conv layers: Kaiming normal (fan_out, ReLU mode).
        BN layers: weight=1, bias=0.
        FC layers: Normal(0, 0.01), bias=0.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def get_features(self) -> Dict[str, torch.Tensor]:
        """Retrieve cached intermediate feature maps.

        Call **after** a forward pass. Returns activations from the stages
        specified in ``ATTENTION_HOOK_STAGES`` (stages 3 and 5 by default).

        Returns:
            Dict with keys like ``"stage_2"``, ``"stage_4"`` mapping to
            their respective output tensors.
        """
        return self._features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the 3D MobileNet.

        Args:
            x: Input video clips of shape ``(B, 3, T, H, W)``.
                Expected: ``(B, 3, 16, 112, 112)``.

        Returns:
            Logits tensor of shape ``(B, num_classes)``.
        """
        # Stem
        x = self.stem(x)   # (B, 32, 16, 56, 56)

        # Inverted residual stages
        for stage_idx, stage in enumerate(self.stages):
            x = stage(x)
            # Cache features for attention-transfer KD
            if stage_idx in ATTENTION_HOOK_STAGES:
                self._features[f"stage_{stage_idx}"] = x

        # Final 1×1×1 expansion
        x = self.final_conv(x)    # (B, 1280, T', H', W')

        # Classification head
        x = self.avgpool(x)        # (B, 1280, 1, 1, 1)
        x = torch.flatten(x, 1)    # (B, 1280)
        x = self.dropout(x)
        x = self.classifier(x)     # (B, num_classes)
        return x


# ======================================================================
# CONVENIENCE CONSTRUCTOR
# ======================================================================

def mobilenet3d(
    num_classes: int = 51,
    width_mult: float = 1.0,
    dropout: float = 0.2,
) -> MobileNet3D:
    """Create a 3D MobileNet Student model.

    Args:
        num_classes: Number of output action classes (51 for HMDB-51).
        width_mult: Width multiplier α. Scales all internal channel counts.
            - ``1.0``: ~2.2M params (default, ~15× smaller than Teacher)
            - ``0.75``: ~1.4M params
            - ``0.5``:  ~0.7M params (~47× smaller)
        dropout: Dropout probability before the classifier.

    Returns:
        A ``MobileNet3D`` model instance.

    Example::

        student = mobilenet3d(num_classes=51, width_mult=1.0)
        logits = student(torch.randn(2, 3, 16, 112, 112))
        print(logits.shape)  # (2, 51)
    """
    model = MobileNet3D(
        num_classes=num_classes,
        width_mult=width_mult,
        dropout=dropout,
    )

    # Log parameter count and architecture summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = sum(
        p.numel() * p.element_size() for p in model.parameters()
    ) / (1024 ** 2)

    logger.info(
        f"MobileNet3D created (α={width_mult}): "
        f"{total_params:,} total params ({trainable_params:,} trainable), "
        f"{model_size_mb:.2f} MB"
    )

    return model
