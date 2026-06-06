# -*- coding: utf-8 -*-
"""
3D ResNet-50 — Teacher Model
=============================

A clean, from-scratch implementation of a 3D ResNet-50 for video action
recognition, designed to serve as the **Teacher** in our knowledge
distillation pipeline.

Architecture highlights:
  • **3D Bottleneck blocks** (1×1×1 → 3×3×3 → 1×1×1) with expansion=4.
  • **Layer configuration**: [3, 4, 6, 3] — standard ResNet-50 depth.
  • **Temporal downsampling**: stride (2,2,2) in layers 2–4 to reduce the
    temporal dimension progressively (16 → 8 → 4 → 2 frames).
  • **2D→3D weight inflation**: utility to convert pre-trained ImageNet
    ResNet-50 weights into 3D by replicating and averaging the temporal
    dimension (I3D strategy, Carreira & Zisserman 2017).
  • **Intermediate hooks**: exposes feature maps from ``layer2`` and
    ``layer3`` for attention-transfer KD (Phase 2 advanced objective).

Expected input tensor shape: ``(B, 3, T, H, W)`` where T=16, H=W=112.

Parameter count: ~33.4M (with 51-class head).

References:
    [1] He et al., "Deep Residual Learning", CVPR 2016.
    [2] Hara et al., "3D ResNets for Action Recognition", CVPR-W 2018.
    [3] Carreira & Zisserman, "Quo Vadis, Action Recognition?", CVPR 2017.
"""

import logging
from typing import Optional, List, Type, Tuple, Dict

import torch
import torch.nn as nn
import torchvision.models as tv_models

logger = logging.getLogger(__name__)


# ======================================================================
# BUILDING BLOCKS
# ======================================================================

class Bottleneck3D(nn.Module):
    """3D Bottleneck residual block (the building block of ResNet-50/101/152).

    Structure::

        x ─→ [1×1×1 conv → BN → ReLU]    (reduce channels)
          ─→ [3×3×3 conv → BN → ReLU]    (spatiotemporal processing)
          ─→ [1×1×1 conv → BN]            (restore channels)
          ─→ + residual ─→ ReLU

    The ``expansion`` factor of 4 means the output channel count is
    ``planes * 4``.  A ``downsample`` projection is applied to the
    shortcut connection when input and output dimensions differ.

    Args:
        in_planes: Number of input channels.
        planes: Number of intermediate (bottleneck) channels.
        stride: Convolution stride as ``(t, h, w)`` or a single int.
            Applied to the 3×3×3 conv to downsample spatially and/or
            temporally.
        downsample: Optional projection module for the residual path
            (used when ``in_planes != planes * expansion`` or stride > 1).
    """

    expansion: int = 4

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: Tuple[int, ...] = (1, 1, 1),
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()

        # ---- 1×1×1: channel reduction ----
        self.conv1 = nn.Conv3d(
            in_planes, planes, kernel_size=1, bias=False
        )
        self.bn1 = nn.BatchNorm3d(planes)

        # ---- 3×3×3: spatiotemporal processing ----
        self.conv2 = nn.Conv3d(
            planes, planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm3d(planes)

        # ---- 1×1×1: channel expansion ----
        self.conv3 = nn.Conv3d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection.

        Args:
            x: Input tensor of shape ``(B, C_in, T, H, W)``.

        Returns:
            Output tensor of shape ``(B, planes*4, T', H', W')``.
        """
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


# ======================================================================
# FULL NETWORK
# ======================================================================

class ResNet3D(nn.Module):
    """3D ResNet for video action recognition (Teacher model).

    Constructs a full 3D ResNet with configurable depth via the ``layers``
    argument.  The default ``[3, 4, 6, 3]`` yields ResNet-50.

    The network exposes intermediate activations via the ``get_features()``
    method for attention-transfer knowledge distillation.

    Args:
        block: Residual block class (``Bottleneck3D``).
        layers: Number of blocks in each of the 4 stages.
        num_classes: Number of output action classes.
        dropout: Dropout probability before the final FC layer.
        zero_init_residual: If ``True``, zero-initialize the last BN in
            each Bottleneck so the residual branch starts as identity.
    """

    def __init__(
        self,
        block: Type[Bottleneck3D],
        layers: List[int],
        num_classes: int = 51,
        dropout: float = 0.5,
        zero_init_residual: bool = True,
    ) -> None:
        super().__init__()
        self.in_planes = 64

        # ---- Stem: initial convolution + BN + ReLU ----
        # Kernel (3,7,7) with stride (1,2,2) preserves temporal dimension
        # and halves spatial resolution: 112→56
        self.stem = nn.Sequential(
            nn.Conv3d(
                3, 64,
                kernel_size=(3, 7, 7),
                stride=(1, 2, 2),
                padding=(1, 3, 3),
                bias=False,
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        # ---- Residual stages ----
        # Stage 1: No spatial/temporal downsampling
        # Stages 2-4: stride (2,2,2) → progressive downsampling
        #   Temporal: 16 → 16 → 8 → 4 → 2
        #   Spatial:  56 → 56 → 28 → 14 → 7
        self.layer1 = self._make_layer(block, 64,  layers[0], stride=(1, 1, 1))
        self.layer2 = self._make_layer(block, 128, layers[1], stride=(2, 2, 2))
        self.layer3 = self._make_layer(block, 256, layers[2], stride=(2, 2, 2))
        self.layer4 = self._make_layer(block, 512, layers[3], stride=(2, 2, 2))

        # ---- Head: global pooling + classifier ----
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # ---- Weight initialization ----
        self._initialize_weights(zero_init_residual)

        # ---- Storage for intermediate features (used by attention transfer) ----
        self._features: Dict[str, torch.Tensor] = {}

    def _make_layer(
        self,
        block: Type[Bottleneck3D],
        planes: int,
        num_blocks: int,
        stride: Tuple[int, ...] = (1, 1, 1),
    ) -> nn.Sequential:
        """Construct a residual stage.

        The first block uses the given ``stride`` for downsampling; all
        subsequent blocks use stride 1.

        Args:
            block: Block class (``Bottleneck3D``).
            planes: Number of bottleneck channels.
            num_blocks: Number of blocks in this stage.
            stride: Stride for the first block.

        Returns:
            An ``nn.Sequential`` containing all blocks for this stage.
        """
        # Downsample projection for the first block if dimensions change
        downsample = None
        if stride != (1, 1, 1) or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.in_planes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm3d(planes * block.expansion),
            )

        blocks = [block(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes * block.expansion

        for _ in range(1, num_blocks):
            blocks.append(block(self.in_planes, planes))

        return nn.Sequential(*blocks)

    def _initialize_weights(self, zero_init_residual: bool) -> None:
        """Kaiming initialization for conv layers; constant for BN."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-init the last BN in each residual branch so that each
        # block starts as an identity mapping (improves convergence).
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck3D):
                    nn.init.constant_(m.bn3.weight, 0)

    def get_features(self) -> Dict[str, torch.Tensor]:
        """Retrieve cached intermediate feature maps.

        Must be called **after** a forward pass. Returns activations from
        ``layer2`` and ``layer3``, which are used for attention transfer.

        Returns:
            Dict with keys ``"layer2"`` and ``"layer3"`` mapping to their
            respective output tensors.
        """
        return self._features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass through the 3D ResNet-50.

        Args:
            x: Input video clips of shape ``(B, 3, T, H, W)``.
                Expected: ``(B, 3, 16, 112, 112)``.

        Returns:
            Logits tensor of shape ``(B, num_classes)``.
        """
        # Stem
        x = self.stem(x)          # (B, 64, 16, 56, 56)

        # Residual stages
        x = self.layer1(x)        # (B, 256,  16, 56, 56)

        x = self.layer2(x)        # (B, 512,   8, 28, 28)
        self._features["layer2"] = x  # Cache for attention transfer

        x = self.layer3(x)        # (B, 1024,  4, 14, 14)
        self._features["layer3"] = x  # Cache for attention transfer

        x = self.layer4(x)        # (B, 2048,  2,  7,  7)

        # Classification head
        x = self.avgpool(x)       # (B, 2048, 1, 1, 1)
        x = torch.flatten(x, 1)   # (B, 2048)
        x = self.dropout(x)
        x = self.fc(x)            # (B, num_classes)
        return x


# ======================================================================
# 2D → 3D WEIGHT INFLATION (I3D STRATEGY)
# ======================================================================

def inflate_2d_to_3d(
    state_dict_2d: Dict[str, torch.Tensor],
    state_dict_3d: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Inflate 2D ImageNet weights into a 3D model's state dict.

    For each 2D convolution weight of shape ``(C_out, C_in, kH, kW)``,
    the 3D equivalent ``(C_out, C_in, kT, kH, kW)`` is created by
    repeating along the temporal axis and normalizing by ``kT`` so the
    output magnitude is preserved (I3D initialization strategy).

    BatchNorm and FC weights are copied directly when shapes match.

    Args:
        state_dict_2d: State dict from a 2D ResNet-50 (e.g., torchvision).
        state_dict_3d: State dict from the 3D ResNet model (target shapes).

    Returns:
        Updated 3D state dict with inflated weights.
    """
    inflated = {}
    skipped = []

    for key_3d, param_3d in state_dict_3d.items():
        # Map 3D key names to 2D equivalents
        # Our naming: stem.0.weight → conv1.weight, layer1.0.conv1 → same
        key_2d = key_3d
        key_2d = key_2d.replace("stem.0.", "conv1.")
        key_2d = key_2d.replace("stem.1.", "bn1.")

        if key_2d not in state_dict_2d:
            skipped.append(key_3d)
            inflated[key_3d] = param_3d  # Keep random init
            continue

        param_2d = state_dict_2d[key_2d]

        if param_2d.dim() == 4 and param_3d.dim() == 5:
            # Inflate 2D conv → 3D conv:
            # (C_out, C_in, kH, kW) → (C_out, C_in, kT, kH, kW)
            kT = param_3d.shape[2]
            # Repeat along temporal axis and normalize
            inflated_weight = param_2d.unsqueeze(2).repeat(1, 1, kT, 1, 1)
            inflated_weight = inflated_weight / kT
            if inflated_weight.shape == param_3d.shape:
                inflated[key_3d] = inflated_weight
            else:
                skipped.append(key_3d)
                inflated[key_3d] = param_3d
        elif param_2d.shape == param_3d.shape:
            # Direct copy (BN params, biases, etc.)
            inflated[key_3d] = param_2d
        else:
            # Shape mismatch (e.g., FC layer with different num_classes)
            skipped.append(key_3d)
            inflated[key_3d] = param_3d

    if skipped:
        logger.info(
            f"Weight inflation: {len(inflated) - len(skipped)} params inflated, "
            f"{len(skipped)} skipped (kept random init): "
            f"{skipped[:5]}{'...' if len(skipped) > 5 else ''}"
        )
    else:
        logger.info(f"Weight inflation: all {len(inflated)} params inflated.")

    return inflated


# ======================================================================
# CONVENIENCE CONSTRUCTOR
# ======================================================================

def resnet3d_50(
    num_classes: int = 51,
    pretrained: bool = False,
    dropout: float = 0.5,
) -> ResNet3D:
    """Create a 3D ResNet-50 Teacher model.

    Args:
        num_classes: Number of output action classes (51 for HMDB-51).
        pretrained: If ``True``, inflate ImageNet ResNet-50 weights from
            torchvision into the 3D architecture. The final FC layer is
            randomly initialized to match ``num_classes``.
        dropout: Dropout probability before the classifier.

    Returns:
        A ``ResNet3D`` model instance.

    Example::

        teacher = resnet3d_50(num_classes=51, pretrained=True)
        logits = teacher(torch.randn(2, 3, 16, 112, 112))
        print(logits.shape)  # (2, 51)
    """
    model = ResNet3D(
        block=Bottleneck3D,
        layers=[3, 4, 6, 3],
        num_classes=num_classes,
        dropout=dropout,
    )

    if pretrained:
        logger.info("Loading ImageNet ResNet-50 weights and inflating to 3D...")
        resnet2d = tv_models.resnet50(weights=tv_models.ResNet50_Weights.DEFAULT)
        state_2d = resnet2d.state_dict()
        state_3d = model.state_dict()

        inflated_state = inflate_2d_to_3d(state_2d, state_3d)
        model.load_state_dict(inflated_state, strict=False)
        logger.info("2D→3D weight inflation complete.")

    # Log parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"ResNet3D-50 created: {total_params:,} total params "
        f"({trainable_params:,} trainable)"
    )

    return model
