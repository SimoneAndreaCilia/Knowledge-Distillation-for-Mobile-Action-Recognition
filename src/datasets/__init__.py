# -*- coding: utf-8 -*-
"""Dataset factory for HMDB-51 video datasets and feature loaders.

Usage::

    from src.datasets import build_dataset

    train_ds, val_ds, test_ds = build_dataset(
        dataset_type="video",
        data_dir="./data/hmdb51",
        annotation_dir="./data/hmdb51_splits",
        split=1,
        num_frames=16,
        frame_size=112,
    )
"""

from typing import Tuple, Optional

from torch.utils.data import Dataset

from .hmdb51 import HMDB51VideoDataset
from .feature_loader import HMDB51FeatureDataset

__all__ = [
    "build_dataset",
    "HMDB51VideoDataset",
    "HMDB51FeatureDataset",
]


def build_dataset(
    dataset_type: str = "video",
    data_dir: str = "./data/hmdb51",
    annotation_dir: str = "./data/hmdb51_splits",
    split: int = 1,
    num_frames: int = 16,
    frame_size: int = 112,
    feature_dir: Optional[str] = None,
) -> Tuple[Dataset, Dataset, Dataset]:
    """Factory function to create train/val/test dataset splits.

    Args:
        dataset_type: ``"video"`` for raw video clips, ``"features"`` for
            pre-extracted feature tensors.
        data_dir: Path to the HMDB-51 video directory.
        annotation_dir: Path to the HMDB-51 split annotation files.
        split: Which HMDB-51 train/test split to use (1, 2, or 3).
        num_frames: Number of frames to sample per clip.
        frame_size: Spatial resolution (height = width).
        feature_dir: Path to pre-extracted features (required when
            ``dataset_type="features"``).

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).
    """
    if dataset_type == "video":
        train_ds = HMDB51VideoDataset(
            root_dir=data_dir,
            annotation_dir=annotation_dir,
            split=split,
            subset="train",
            num_frames=num_frames,
            frame_size=frame_size,
            augment=True,
        )
        # Use the test set as validation for HMDB-51 (standard practice)
        val_ds = HMDB51VideoDataset(
            root_dir=data_dir,
            annotation_dir=annotation_dir,
            split=split,
            subset="test",
            num_frames=num_frames,
            frame_size=frame_size,
            augment=False,
        )
        test_ds = val_ds  # HMDB-51 has no separate val; test = val

    elif dataset_type == "features":
        if feature_dir is None:
            raise ValueError(
                "feature_dir must be provided when dataset_type='features'"
            )
        train_ds = HMDB51FeatureDataset(
            feature_dir=feature_dir, split=split, subset="train"
        )
        val_ds = HMDB51FeatureDataset(
            feature_dir=feature_dir, split=split, subset="test"
        )
        test_ds = val_ds

    else:
        raise ValueError(
            f"Unknown dataset_type '{dataset_type}'. Use 'video' or 'features'."
        )

    return train_ds, val_ds, test_ds
