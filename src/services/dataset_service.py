# -*- coding: utf-8 -*-
"""DatasetService — filesystem access layer for the HMDB-51 dataset.

Abstracts all ``Path`` manipulation and directory iteration away from the
UI and callback layers.

Usage::

    service = DatasetService(data_dir=Path("data/hmdb51"))
    classes = service.get_classes()
    videos  = service.get_videos("brush_hair")
    path    = service.resolve_path("brush_hair", "Amy_Adams_03.avi")
"""

import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

_VIDEO_EXTENSIONS = frozenset({".avi", ".mp4", ".mkv", ".mov"})


class DatasetService:
    """Provides read access to the HMDB-51 dataset directory structure.

    Responsibilities (SRP):
        - Enumerate action class directories.
        - List video files within a class directory.
        - Resolve the full path for a given class/video pair.

    Non-responsibilities:
        - Video decoding or preprocessing.
        - Any model or inference logic.
        - Caching (callers may cache results themselves).

    Args:
        data_dir: Root of the HMDB-51 dataset (contains one sub-dir per class).
    """

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def data_dir(self) -> Path:
        return self._data_dir

    def dataset_exists(self) -> bool:
        """Return True if the dataset directory is present."""
        return self._data_dir.is_dir()

    def get_classes(self) -> List[str]:
        """Return a sorted list of HMDB-51 action class names.

        Returns an empty list (rather than raising) if the dataset directory
        does not exist, so that the UI can start without data.
        """
        if not self.dataset_exists():
            logger.warning(
                "DatasetService: data_dir does not exist: %s", self._data_dir
            )
            return []
        return sorted(
            [d.name for d in self._data_dir.iterdir() if d.is_dir()]
        )

    def get_videos(self, class_name: str) -> List[str]:
        """Return sorted video filenames for *class_name*.

        Args:
            class_name: Name of the action class (must match a sub-directory).

        Returns:
            List of filenames (not full paths) for that class.
            Empty list if the class does not exist or has no videos.
        """
        if not class_name:
            return []
        class_dir = self._data_dir / class_name
        if not class_dir.is_dir():
            return []
        return sorted(
            f.name
            for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in _VIDEO_EXTENSIONS
        )

    def resolve_path(self, class_name: str, video_name: str) -> Optional[Path]:
        """Return the absolute Path for a dataset video, or None.

        Returns None rather than raising if either argument is empty or the
        file does not exist.

        Args:
            class_name:  HMDB-51 action class name.
            video_name:  Filename of the video within that class.

        Returns:
            ``Path`` to the video, or ``None`` if not resolvable.
        """
        if not class_name or not video_name:
            return None
        candidate = self._data_dir / class_name / video_name
        return candidate if candidate.is_file() else None

    def __repr__(self) -> str:
        return f"DatasetService(data_dir={self._data_dir})"
