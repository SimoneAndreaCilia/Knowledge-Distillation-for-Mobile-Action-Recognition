# -*- coding: utf-8 -*-
"""
HMDB-51 Video Dataset
======================

A PyTorch ``Dataset`` for loading and sampling video clips from the HMDB-51
action recognition dataset.

**Dataset structure expected on disk**::

    data/hmdb51/
    ├── brush_hair/
    │   ├── April_09_brush_hair_u_nm_np1_ba_goo_0.avi
    │   ├── ...
    ├── cartwheel/
    │   ├── ...
    └── ... (51 action classes)

    data/hmdb51_splits/
    ├── brush_hair_test_split1.txt
    ├── brush_hair_test_split2.txt
    ├── brush_hair_test_split3.txt
    └── ... (split files for each class)

**Split file format** (HMDB-51 standard):
  Each line: ``<video_name> <label>`` where label is:
    - ``1`` = training set
    - ``2`` = test set
    - ``0`` = not used (unused videos)

**Temporal sampling strategy**:
  • Training: random start index → uniform sampling of ``num_frames`` frames.
  • Testing: center-crop temporal segment for deterministic evaluation.

**Spatial transforms**:
  • Training: random crop + horizontal flip + color jitter + normalization.
  • Testing: center crop + normalization.

Dependencies:
  Requires ``torchvision`` for video I/O. Falls back to OpenCV (``cv2``)
  if torchvision video backend is unavailable.

Usage::

    dataset = HMDB51VideoDataset(
        root_dir="./data/hmdb51",
        annotation_dir="./data/hmdb51_splits",
        split=1,
        subset="train",
        num_frames=16,
        frame_size=112,
        augment=True,
    )
    clip, label = dataset[0]
    print(clip.shape)   # torch.Size([3, 16, 112, 112])
    print(label)        # int in [0, 50]
"""

import os
import logging
from pathlib import Path
from typing import Tuple, Optional, List, Dict

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# ---- Normalization constants (Kinetics-400 statistics, standard for video models) ----
KINETICS_MEAN = [0.43216, 0.394666, 0.37645]
KINETICS_STD = [0.22803, 0.22145, 0.216989]


class HMDB51VideoDataset(Dataset):
    """PyTorch dataset for HMDB-51 video clips.

    Loads raw video files, samples a fixed number of frames using temporal
    uniform sampling, applies spatial transforms, and returns a
    ``(clip, label)`` pair.

    Args:
        root_dir: Path to the directory containing class subdirectories
            with video files.
        annotation_dir: Path to the directory containing HMDB-51 split
            annotation text files.
        split: Which official split to use (1, 2, or 3).
        subset: ``"train"`` or ``"test"``.
        num_frames: Number of frames to sample per clip.
        frame_size: Target spatial resolution (height = width).
        augment: If ``True``, apply training augmentations (random crop,
            horizontal flip, color jitter). If ``False``, use center crop.
    """

    def __init__(
        self,
        root_dir: str,
        annotation_dir: str,
        split: int = 1,
        subset: str = "train",
        num_frames: int = 16,
        frame_size: int = 112,
        augment: bool = True,
    ) -> None:
        super().__init__()

        self.root_dir = Path(root_dir)
        self.annotation_dir = Path(annotation_dir)
        self.split = split
        self.subset = subset
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.augment = augment

        # ---- Build class-to-index mapping ----
        self.classes: List[str] = sorted([
            d.name for d in self.root_dir.iterdir() if d.is_dir()
        ])
        self.class_to_idx: Dict[str, int] = {
            cls_name: idx for idx, cls_name in enumerate(self.classes)
        }

        # ---- Parse split annotations to build sample list ----
        self.samples: List[Tuple[str, int]] = []  # (video_path, class_idx)
        self._parse_split_files()

        logger.info(
            f"HMDB51VideoDataset ({subset}, split {split}): "
            f"{len(self.samples)} videos, {len(self.classes)} classes, "
            f"{num_frames} frames @ {frame_size}×{frame_size}"
        )

    def _parse_split_files(self) -> None:
        """Parse HMDB-51 split annotation files to build the sample list.

        Each split file is named ``<classname>_test_split<N>.txt`` and
        contains lines of the form ``<video_filename> <label>`` where
        label 1=train, 2=test, 0=unused.
        """
        # Determine which label value corresponds to our subset
        target_label = "1" if self.subset == "train" else "2"

        for class_name in self.classes:
            split_file = (
                self.annotation_dir
                / f"{class_name}_test_split{self.split}.txt"
            )

            if not split_file.exists():
                logger.warning(f"Split file not found: {split_file}")
                continue

            with open(split_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split()
                    if len(parts) < 2:
                        continue

                    video_name, label = parts[0], parts[1]

                    if label == target_label:
                        video_path = str(
                            self.root_dir / class_name / video_name
                        )
                        class_idx = self.class_to_idx[class_name]
                        self.samples.append((video_path, class_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """Load a video, sample frames, apply transforms.

        Args:
            index: Sample index.

        Returns:
            Tuple of:
              - ``clip``: Float tensor of shape ``(3, T, H, W)`` in [0, 1]
                range, normalized with Kinetics statistics.
              - ``label``: Integer class index.
        """
        video_path, label = self.samples[index]

        # Load frames from video file
        frames = self._load_video_frames(video_path)

        # Temporal sampling: select `num_frames` evenly spaced frames
        frames = self._temporal_sample(frames)

        # Spatial transforms + normalization
        clip = self._spatial_transform(frames)

        return clip, label

    def _load_video_frames(self, video_path: str) -> np.ndarray:
        """Load all frames from a video file.

        Tries torchvision's video reader first, falls back to OpenCV.

        Args:
            video_path: Path to the video file.

        Returns:
            NumPy array of shape ``(total_frames, H, W, 3)`` in uint8.
        """
        # ---- Strategy 1: torchvision.io (preferred) ----
        try:
            import torchvision.io as vio
            video_tensor, _, _ = vio.read_video(
                video_path, pts_unit="sec"
            )
            # video_tensor: (T, H, W, C) uint8
            return video_tensor.numpy()
        except Exception:
            pass

        # ---- Strategy 2: OpenCV fallback ----
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # OpenCV loads BGR; convert to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            cap.release()

            if len(frames) == 0:
                logger.warning(f"Empty video: {video_path}. Returning zeros.")
                return np.zeros(
                    (self.num_frames, self.frame_size, self.frame_size, 3),
                    dtype=np.uint8,
                )
            return np.stack(frames)
        except ImportError:
            raise RuntimeError(
                "Neither torchvision.io nor cv2 is available for video "
                "loading. Install torchvision with video support or opencv-python."
            )

    def _temporal_sample(self, frames: np.ndarray) -> np.ndarray:
        """Sample ``num_frames`` frames using uniform temporal sampling.

        Training: random offset within each temporal segment.
        Testing: deterministic center of each segment.

        Args:
            frames: All frames, shape ``(total_frames, H, W, 3)``.

        Returns:
            Sampled frames, shape ``(num_frames, H, W, 3)``.
        """
        total_frames = len(frames)

        if total_frames >= self.num_frames:
            # Uniform segment sampling (TSN-style)
            segment_length = total_frames / self.num_frames
            indices = []
            for i in range(self.num_frames):
                start = int(i * segment_length)
                end = int((i + 1) * segment_length)
                if self.augment:
                    # Random offset within segment (training)
                    idx = np.random.randint(start, max(start + 1, end))
                else:
                    # Center of segment (testing)
                    idx = (start + end) // 2
                indices.append(min(idx, total_frames - 1))
            return frames[indices]
        else:
            # Video is shorter than num_frames: loop/repeat frames
            indices = np.linspace(0, total_frames - 1, self.num_frames)
            indices = np.clip(indices.astype(int), 0, total_frames - 1)
            return frames[indices]

    def _spatial_transform(self, frames: np.ndarray) -> torch.Tensor:
        """Apply spatial augmentations and normalization.

        Training: random resize → random crop → horizontal flip → normalize.
        Testing: resize → center crop → normalize.

        Args:
            frames: Sampled frames, shape ``(T, H, W, 3)`` uint8.

        Returns:
            Normalized clip tensor of shape ``(3, T, H, W)`` float32.
        """
        T, H, W, C = frames.shape
        target_size = self.frame_size

        # Convert to float32 [0, 1]
        clip = frames.astype(np.float32) / 255.0

        if self.augment:
            # ---- Training augmentations ----

            # Random resize: scale between 1.0x and 1.25x of target
            scale = np.random.uniform(1.0, 1.25)
            resize_h = int(target_size * scale)
            resize_w = int(target_size * scale)
            clip = self._resize_clip(clip, resize_h, resize_w)

            # Random crop to target_size × target_size
            new_h, new_w = clip.shape[1], clip.shape[2]
            top = np.random.randint(0, max(1, new_h - target_size + 1))
            left = np.random.randint(0, max(1, new_w - target_size + 1))
            clip = clip[:, top:top + target_size, left:left + target_size, :]

            # Random horizontal flip (50% probability)
            if np.random.random() < 0.5:
                clip = clip[:, :, ::-1, :].copy()

        else:
            # ---- Testing: deterministic center crop ----
            resize_h = int(target_size * 1.14)  # ~128 for target=112
            resize_w = int(target_size * 1.14)
            clip = self._resize_clip(clip, resize_h, resize_w)

            new_h, new_w = clip.shape[1], clip.shape[2]
            top = (new_h - target_size) // 2
            left = (new_w - target_size) // 2
            clip = clip[:, top:top + target_size, left:left + target_size, :]

        # Convert to tensor: (T, H, W, C) → (C, T, H, W)
        clip_tensor = torch.from_numpy(clip).permute(3, 0, 1, 2).contiguous()

        # Normalize with Kinetics-400 statistics
        mean = torch.tensor(KINETICS_MEAN).view(3, 1, 1, 1)
        std = torch.tensor(KINETICS_STD).view(3, 1, 1, 1)
        clip_tensor = (clip_tensor - mean) / std

        return clip_tensor

    @staticmethod
    def _resize_clip(
        clip: np.ndarray, target_h: int, target_w: int
    ) -> np.ndarray:
        """Resize all frames in a clip using bilinear interpolation.

        Args:
            clip: Shape ``(T, H, W, C)`` float32.
            target_h: Target height.
            target_w: Target width.

        Returns:
            Resized clip of shape ``(T, target_h, target_w, C)``.
        """
        try:
            import cv2
            resized = np.stack([
                cv2.resize(frame, (target_w, target_h))
                for frame in clip
            ])
        except ImportError:
            # Pure NumPy fallback (nearest neighbor, less quality)
            from PIL import Image
            resized = np.stack([
                np.array(
                    Image.fromarray((frame * 255).astype(np.uint8))
                    .resize((target_w, target_h), Image.BILINEAR)
                ).astype(np.float32) / 255.0
                for frame in clip
            ])
        return resized
