# -*- coding: utf-8 -*-
"""
Pre-extracted Feature Loader for HMDB-51
=========================================

Alternative data pipeline for when raw video processing is too expensive
or slow.  Works with features pre-extracted by a backbone (e.g., the
Teacher's penultimate layer) and stored as ``.npy`` files.

**Pre-extracted feature directory structure**::

    features/
    ├── train/
    │   ├── brush_hair/
    │   │   ├── video_001.npy     # Shape: (T, D) or (D,) for pooled
    │   │   ├── video_002.npy
    │   │   └── ...
    │   ├── cartwheel/
    │   │   └── ...
    │   └── ... (51 classes)
    └── test/
        ├── brush_hair/
        │   └── ...
        └── ...

**Feature extraction script**:
  See ``extract_features()`` at the bottom of this module for a utility
  function that pre-extracts features from videos using a trained model.

Usage::

    dataset = HMDB51FeatureDataset(
        feature_dir="./features",
        split=1,
        subset="train",
    )
    features, label = dataset[0]
    print(features.shape)  # e.g., torch.Size([2048]) for pooled features
"""

import os
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class HMDB51FeatureDataset(Dataset):
    """PyTorch dataset for pre-extracted HMDB-51 features.

    Loads ``.npy`` files containing feature vectors extracted from videos
    by a backbone network.  This is much faster than loading raw videos
    and is suitable for distillation experiments where the teacher's
    features are pre-computed.

    Args:
        feature_dir: Root directory containing ``train/`` and ``test/``
            subdirectories with class-organized ``.npy`` files.
        split: HMDB-51 split number (1, 2, or 3). Used for logging only
            (the directory structure already encodes train/test).
        subset: ``"train"`` or ``"test"`` — selects the subdirectory.
    """

    def __init__(
        self,
        feature_dir: str,
        split: int = 1,
        subset: str = "train",
    ) -> None:
        super().__init__()

        self.feature_dir = Path(feature_dir) / subset
        self.split = split
        self.subset = subset

        if not self.feature_dir.exists():
            raise FileNotFoundError(
                f"Feature directory not found: {self.feature_dir}. "
                f"Run feature extraction first."
            )

        # ---- Build class-to-index mapping ----
        self.classes: List[str] = sorted([
            d.name for d in self.feature_dir.iterdir() if d.is_dir()
        ])
        self.class_to_idx: Dict[str, int] = {
            cls: idx for idx, cls in enumerate(self.classes)
        }

        # ---- Scan for .npy files ----
        self.samples: List[Tuple[str, int]] = []
        for cls_name in self.classes:
            cls_dir = self.feature_dir / cls_name
            for npy_file in sorted(cls_dir.glob("*.npy")):
                self.samples.append(
                    (str(npy_file), self.class_to_idx[cls_name])
                )

        logger.info(
            f"HMDB51FeatureDataset ({subset}, split {split}): "
            f"{len(self.samples)} samples, {len(self.classes)} classes"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """Load a pre-extracted feature vector.

        Args:
            index: Sample index.

        Returns:
            Tuple of:
              - ``features``: Float tensor (shape depends on extraction).
              - ``label``: Integer class index.
        """
        npy_path, label = self.samples[index]
        features = np.load(npy_path)
        return torch.from_numpy(features).float(), label


# ======================================================================
# FEATURE EXTRACTION UTILITY
# ======================================================================

@torch.no_grad()
def extract_features(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    output_dir: str,
    device: torch.device = torch.device("cpu"),
    layer_name: Optional[str] = None,
) -> None:
    """Pre-extract features from a dataset using a trained model.

    Runs the model in eval mode, hooks into the penultimate layer (before
    the final FC), and saves the pooled feature vectors as ``.npy`` files.

    This function creates the same directory structure expected by
    ``HMDB51FeatureDataset``.

    Args:
        model: Trained backbone model (Teacher or Student).
        dataloader: DataLoader wrapping an ``HMDB51VideoDataset``.
            The dataset must have a ``samples`` attribute with
            ``(video_path, class_idx)`` tuples and a ``classes`` list.
        output_dir: Root directory for the output features. Subdirectories
            are created per-class automatically.
        device: Device to run inference on.
        layer_name: Not used directly — we hook the avgpool layer.

    Example::

        from src.models import build_model
        from src.datasets import HMDB51VideoDataset
        from torch.utils.data import DataLoader

        teacher = build_model("teacher", num_classes=51, pretrained=True)
        teacher.load_state_dict(torch.load("teacher_best.pth")["model_state_dict"])

        dataset = HMDB51VideoDataset(
            root_dir="./data/hmdb51",
            annotation_dir="./data/hmdb51_splits",
            split=1, subset="train", num_frames=16, frame_size=112,
        )
        loader = DataLoader(dataset, batch_size=1, num_workers=4)

        extract_features(teacher, loader, "./features/train", device=torch.device("cuda"))
    """
    model = model.to(device)
    model.eval()

    dataset = dataloader.dataset
    classes = getattr(dataset, "classes", [])
    samples = getattr(dataset, "samples", [])

    output_path = Path(output_dir)

    # Create class subdirectories
    for cls_name in classes:
        (output_path / cls_name).mkdir(parents=True, exist_ok=True)

    # Hook to capture features before the FC layer
    features_cache = {}

    def hook_fn(module, input, output):
        # After avgpool, output is (B, C, 1, 1, 1) or (B, C)
        features_cache["features"] = output

    # Register hook on the avgpool layer
    if hasattr(model, "avgpool"):
        hook = model.avgpool.register_forward_hook(hook_fn)
    else:
        logger.warning("Model has no 'avgpool' attribute; saving logits instead.")
        hook = None

    for idx, (clip, label) in enumerate(dataloader):
        clip = clip.to(device)
        _ = model(clip)  # Forward pass to trigger hook

        if "features" in features_cache:
            feat = features_cache["features"].cpu().numpy()
            feat = feat.reshape(feat.shape[0], -1)  # Flatten to (B, D)
        else:
            continue

        # Save features for each sample in the batch
        for b in range(feat.shape[0]):
            sample_idx = idx * dataloader.batch_size + b
            if sample_idx >= len(samples):
                break

            video_path, class_idx = samples[sample_idx]
            class_name = classes[class_idx]
            video_name = Path(video_path).stem

            save_path = output_path / class_name / f"{video_name}.npy"
            np.save(save_path, feat[b])

        if (idx + 1) % 100 == 0:
            logger.info(f"Extracted features: {idx + 1}/{len(dataloader)} batches")

    if hook is not None:
        hook.remove()

    logger.info(
        f"Feature extraction complete → {output_path} "
        f"({len(samples)} samples)"
    )
