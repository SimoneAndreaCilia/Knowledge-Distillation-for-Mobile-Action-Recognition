# -*- coding: utf-8 -*-
"""Dataset constants — HMDB-51 class names and class count.

Class names are derived lazily from the dataset directory so that the
module can be imported even when the data directory does not exist yet
(e.g. during unit testing or CI).

Usage::

    from src.config.constants import CLASS_NAMES, NUM_CLASSES
"""

from functools import lru_cache
from pathlib import Path
from typing import List

from .settings import get_settings


@lru_cache(maxsize=1)
def get_class_names() -> List[str]:
    """Return sorted HMDB-51 class names, derived from directory structure.

    Returns an empty list if the dataset directory does not exist rather
    than raising an exception, allowing the app to start without data.
    """
    data_dir: Path = get_settings().data_dir
    if not data_dir.exists():
        return []
    return sorted([d.name for d in data_dir.iterdir() if d.is_dir()])


def get_num_classes() -> int:
    """Return number of action classes (defaults to 51 if data is missing)."""
    names = get_class_names()
    return len(names) if names else 51


# Module-level aliases for backwards-compatibility and convenience imports
CLASS_NAMES: List[str] = get_class_names()
NUM_CLASSES: int = get_num_classes()
