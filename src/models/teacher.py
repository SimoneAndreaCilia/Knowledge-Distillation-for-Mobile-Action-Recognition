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
    temporal dimension progressively.
  • **Kinetics-400 pre-training** (Hara et al., CVPR 2018): loads native
    3D weights trained on ~240K videos, with full spatiotemporal features.
  • **ImageNet inflation fallback**: utility to convert 2D ImageNet weights
    into 3D (I3D strategy, Carreira & Zisserman 2017).
  • **Intermediate hooks**: exposes feature maps from ``layer2`` and
    ``layer3`` for attention-transfer KD (Phase 2 advanced objective).

Expected input tensor shape: ``(B, 3, T, H, W)`` where T=16, H=W=112.

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

    The stem (first conv + maxpool) is configurable to match different
    pre-training sources:

      • **Kinetics** (Hara et al.): ``conv1_t_size=7``, isotropic maxpool
        ``(3,3,3)`` with stride ``(2,2,2)`` — halves temporal dimension.
      • **ImageNet inflation**: ``conv1_t_size=3``, spatial-only maxpool
        ``(1,3,3)`` with stride ``(1,2,2)`` — preserves temporal dimension.

    The network exposes intermediate activations via the ``get_features()``
    method for attention-transfer knowledge distillation.

    Args:
        block: Residual block class (``Bottleneck3D``).
        layers: Number of blocks in each of the 4 stages.
        num_classes: Number of output action classes.
        dropout: Dropout probability before the final FC layer.
        zero_init_residual: If ``True``, zero-initialize the last BN in
            each Bottleneck so the residual branch starts as identity.
        conv1_t_size: Temporal kernel size for the stem convolution.
            Use 7 for Kinetics-compatible architecture, 3 for ImageNet.
        conv1_t_stride: Temporal stride for the stem convolution.
        no_max_pool: If ``True``, skip the max pooling layer after stem.
    """

    def __init__(
        self,
        block: Type[Bottleneck3D],
        layers: List[int],
        num_classes: int = 51,
        dropout: float = 0.5,
        zero_init_residual: bool = True,
        conv1_t_size: int = 7,
        conv1_t_stride: int = 1,
        no_max_pool: bool = False,
    ) -> None:
        super().__init__()
        self.in_planes = 64

        # ---- Stem: initial convolution + BN + ReLU ----
        # The temporal kernel size is configurable:
        #   conv1_t_size=7 for Kinetics weights (full temporal receptive field)
        #   conv1_t_size=3 for ImageNet-inflated weights
        self.stem = nn.Sequential(
            nn.Conv3d(
                3, 64,
                kernel_size=(conv1_t_size, 7, 7),
                stride=(conv1_t_stride, 2, 2),
                padding=(conv1_t_size // 2, 3, 3),
                bias=False,
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        # ---- MaxPool: separate from stem for clean state_dict mapping ----
        # Kinetics (Hara): isotropic (3,3,3) stride (2,2,2) — halves T, H, W
        # ImageNet: spatial-only (1,3,3) stride (1,2,2) — preserves T
        if no_max_pool:
            self.maxpool = None
        else:
            # Default: Kinetics-compatible isotropic 3D max pooling
            self.maxpool = nn.MaxPool3d(
                kernel_size=3, stride=2, padding=1
            )

        # ---- Residual stages ----
        # Dimensions after stem+maxpool (Kinetics config, T=16, H=W=112):
        #   stem:    (B, 64, 16, 56, 56)
        #   maxpool: (B, 64,  8, 28, 28)  [halves T, H, W]
        #   layer1:  (B, 256,  8, 28, 28)
        #   layer2:  (B, 512,  4, 14, 14)
        #   layer3:  (B,1024,  2,  7,  7)
        #   layer4:  (B,2048,  1,  4,  4)
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
        # Stem (conv + BN + ReLU)
        x = self.stem(x)

        # MaxPool (if enabled)
        if self.maxpool is not None:
            x = self.maxpool(x)

        # Residual stages
        x = self.layer1(x)

        x = self.layer2(x)
        self._features["layer2"] = x  # Cache for attention transfer

        x = self.layer3(x)
        self._features["layer3"] = x  # Cache for attention transfer

        x = self.layer4(x)

        # Classification head
        x = self.avgpool(x)       # (B, 2048, 1, 1, 1)
        x = torch.flatten(x, 1)   # (B, 2048)
        x = self.dropout(x)
        x = self.fc(x)            # (B, num_classes)
        return x


# ======================================================================
# KINETICS-400 PRE-TRAINED WEIGHT LOADING (Hara et al., CVPR 2018)
# ======================================================================

def load_kinetics_weights(
    model: ResNet3D,
    pretrained_path: str,
) -> None:
    """Load Kinetics-400 pre-trained weights from Hara et al.'s checkpoint.

    Maps Hara's state dict key naming convention to our architecture:
      • ``conv1.*``  →  ``stem.0.*``
      • ``bn1.*``    →  ``stem.1.*``
      • ``layer*.*`` →  ``layer*.*``  (identical, no mapping needed)
      • ``fc.*``     →  skipped (different num_classes)

    Handles checkpoints that store state dict under ``'state_dict'`` key
    (common in Hara's releases) and strips ``'module.'`` prefix from
    DataParallel-wrapped checkpoints.

    Args:
        model: The ``ResNet3D`` instance to load weights into.
        pretrained_path: Path to the ``.pth`` checkpoint file
            (e.g., ``'./pretrained/r3d50_K_200ep.pth'``).

    Raises:
        FileNotFoundError: If ``pretrained_path`` does not exist.
    """
    import os
    if not os.path.isfile(pretrained_path):
        raise FileNotFoundError(
            f"Kinetics checkpoint not found: {pretrained_path}\n"
            f"Download 'r3d50_K_200ep.pth' from:\n"
            f"  https://github.com/kenshohara/3D-ResNets-PyTorch/releases\n"
            f"and place it in the ./pretrained/ directory."
        )

    logger.info(f"Loading Kinetics-400 weights from: {pretrained_path}")
    checkpoint = torch.load(pretrained_path, map_location="cpu", weights_only=False)

    # ---- Extract state dict from checkpoint ----
    # Hara's checkpoints often wrap state dict under 'state_dict' key
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        source_state = checkpoint["state_dict"]
        logger.info(
            f"Checkpoint contains keys: {list(checkpoint.keys())}. "
            f"Using 'state_dict' ({len(source_state)} params)."
        )
    elif isinstance(checkpoint, dict) and any(
        k.startswith(("conv1.", "layer1.", "bn1.")) for k in checkpoint.keys()
    ):
        # Raw state dict (no wrapper)
        source_state = checkpoint
    else:
        raise ValueError(
            f"Unexpected checkpoint format. Keys: {list(checkpoint.keys())[:10]}"
        )

    # ---- Strip 'module.' prefix from DataParallel wrapping ----
    cleaned_state = {}
    for key, value in source_state.items():
        clean_key = key.replace("module.", "", 1) if key.startswith("module.") else key
        cleaned_state[clean_key] = value

    # ---- Map Hara's keys to our architecture ----
    # Hara uses: conv1.weight, bn1.weight, layer1.0.conv1.weight, fc.weight
    # We use:    stem.0.weight, stem.1.weight, layer1.0.conv1.weight, fc.weight
    mapped_state = {}
    skipped = []

    model_state = model.state_dict()

    for hara_key, param in cleaned_state.items():
        # Map stem keys
        our_key = hara_key
        our_key = our_key.replace("conv1.", "stem.0.", 1) if hara_key.startswith("conv1.") else our_key
        our_key = our_key.replace("bn1.", "stem.1.", 1) if hara_key.startswith("bn1.") else our_key

        # Skip fc layer (400 classes → 51 classes)
        if our_key.startswith("fc."):
            skipped.append(f"{hara_key} → {our_key} (fc: class count mismatch)")
            continue

        # Check if key exists in our model
        if our_key not in model_state:
            skipped.append(f"{hara_key} → {our_key} (not in model)")
            continue

        # Check shape compatibility
        if param.shape != model_state[our_key].shape:
            skipped.append(
                f"{hara_key} → {our_key} "
                f"(shape: {tuple(param.shape)} vs {tuple(model_state[our_key].shape)})"
            )
            continue

        mapped_state[our_key] = param

    # ---- Load mapped weights ----
    load_result = model.load_state_dict(mapped_state, strict=False)

    # ---- Report ----
    logger.info(
        f"Kinetics weight loading complete:\n"
        f"  Loaded:  {len(mapped_state)} params\n"
        f"  Skipped: {len(skipped)} params\n"
        f"  Missing in checkpoint: {len(load_result.missing_keys)} "
        f"(randomly initialized)"
    )
    if skipped:
        for s in skipped[:10]:
            logger.info(f"  [SKIP] {s}")
        if len(skipped) > 10:
            logger.info(f"  ... and {len(skipped) - 10} more")

    if load_result.missing_keys:
        logger.info(
            f"  Missing keys (random init): "
            f"{load_result.missing_keys[:5]}"
            f"{'...' if len(load_result.missing_keys) > 5 else ''}"
        )


# ======================================================================
# 2D → 3D WEIGHT INFLATION (I3D STRATEGY — ImageNet fallback)
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
    pretrained_source: str = "kinetics",
    pretrained_path: Optional[str] = None,
    dropout: float = 0.5,
) -> ResNet3D:
    """Create a 3D ResNet-50 Teacher model.

    Supports two pre-training sources:

      • **``"kinetics"``** (recommended): Loads native 3D weights from
        Hara et al.'s Kinetics-400 checkpoint. The architecture uses
        ``conv1_t_size=7`` and isotropic 3D maxpool to match Hara's
        architecture exactly.

      • **``"imagenet"``**: Inflates 2D ImageNet ResNet-50 weights from
        torchvision into 3D. Uses ``conv1_t_size=3`` and spatial-only
        maxpool. Not recommended for action recognition (no temporal
        features).

    Args:
        num_classes: Number of output action classes (51 for HMDB-51).
        pretrained: If ``True``, load pre-trained weights.
        pretrained_source: ``"kinetics"`` or ``"imagenet"``.
        pretrained_path: Path to the Kinetics ``.pth`` file. Required
            when ``pretrained_source="kinetics"`` and ``pretrained=True``.
        dropout: Dropout probability before the classifier.

    Returns:
        A ``ResNet3D`` model instance.

    Example::

        # Kinetics pre-trained (recommended)
        teacher = resnet3d_50(
            num_classes=51, pretrained=True,
            pretrained_source="kinetics",
            pretrained_path="./pretrained/r3d50_K_200ep.pth",
        )

        # ImageNet-inflated (fallback)
        teacher = resnet3d_50(
            num_classes=51, pretrained=True,
            pretrained_source="imagenet",
        )
    """
    if pretrained_source == "kinetics":
        # ---- Kinetics-compatible architecture ----
        # Stem conv: (7,7,7) — full temporal receptive field
        # MaxPool: (3,3,3) stride (2,2,2) — isotropic downsampling
        model = ResNet3D(
            block=Bottleneck3D,
            layers=[3, 4, 6, 3],
            num_classes=num_classes,
            dropout=dropout,
            conv1_t_size=7,
            conv1_t_stride=1,
            no_max_pool=False,  # Use isotropic maxpool
        )

        if pretrained:
            if pretrained_path is None:
                raise ValueError(
                    "pretrained_path is required when pretrained_source='kinetics'. "
                    "Download 'r3d50_K_200ep.pth' from:\n"
                    "  https://github.com/kenshohara/3D-ResNets-PyTorch/releases\n"
                    "and set pretrained_path='./pretrained/r3d50_K_200ep.pth'"
                )
            load_kinetics_weights(model, pretrained_path)

    elif pretrained_source == "imagenet":
        # ---- ImageNet-inflation architecture ----
        # Stem conv: (3,7,7) — small temporal kernel for inflation
        # MaxPool: spatial-only (1,3,3) stride (1,2,2)
        model = ResNet3D(
            block=Bottleneck3D,
            layers=[3, 4, 6, 3],
            num_classes=num_classes,
            dropout=dropout,
            conv1_t_size=3,
            conv1_t_stride=1,
            no_max_pool=True,  # We'll add a spatial-only maxpool manually
        )
        # Add spatial-only maxpool for ImageNet variant
        model.maxpool = nn.MaxPool3d(
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
        )

        if pretrained:
            logger.info("Loading ImageNet ResNet-50 weights and inflating to 3D...")
            resnet2d = tv_models.resnet50(weights=tv_models.ResNet50_Weights.DEFAULT)
            state_2d = resnet2d.state_dict()
            state_3d = model.state_dict()

            inflated_state = inflate_2d_to_3d(state_2d, state_3d)
            model.load_state_dict(inflated_state, strict=False)
            logger.info("2D→3D weight inflation complete.")
    else:
        raise ValueError(
            f"Unknown pretrained_source='{pretrained_source}'. "
            f"Use 'kinetics' or 'imagenet'."
        )

    # Log parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"ResNet3D-50 created ({pretrained_source}): "
        f"{total_params:,} total params ({trainable_params:,} trainable)"
    )

    return model
