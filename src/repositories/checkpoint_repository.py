# -*- coding: utf-8 -*-
"""CheckpointRepository — file-system access layer for model checkpoints.

Isolates all ``torch.load`` calls in one place, separating *where to find*
a checkpoint from *how to build* a model.  The service layer only calls
``load_state_dict()``; it never constructs paths or opens files itself.

Usage::

    from src.repositories.checkpoint_repository import CheckpointRepository

    repo = CheckpointRepository()
    state_dict = repo.load_state_dict(Path("checkpoints/teacher/best_model.pth"))
"""

import logging
from pathlib import Path
from typing import Any, Dict

import torch

logger = logging.getLogger(__name__)

# Type alias for a state dict
StateDictType = Dict[str, Any]


class CheckpointRepository:
    """Loads PyTorch checkpoint files from the filesystem.

    Responsibilities (SRP):
        - Verify that a checkpoint file exists.
        - Deserialise the checkpoint with ``torch.load``.
        - Extract the ``model_state_dict`` sub-key if present.

    Non-responsibilities:
        - Building model architectures.
        - Moving tensors to devices.
        - Any caching.
    """

    def exists(self, path: Path) -> bool:
        """Return True if the checkpoint file is present on disk."""
        return path.is_file()

    def load_state_dict(self, path: Path) -> StateDictType:
        """Load and return the model state dict from *path*.

        Handles both "flat" checkpoints (the state dict itself) and
        "nested" checkpoints that store the state under ``model_state_dict``.

        Args:
            path: Absolute path to the ``.pth`` checkpoint file.

        Returns:
            A ``state_dict`` mapping parameter names to tensors.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
        """
        if not path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {path}\n"
                "Make sure you have run the training script for this model."
            )

        logger.info("Loading checkpoint: %s", path)
        raw = torch.load(path, map_location="cpu", weights_only=False)

        if isinstance(raw, dict) and "model_state_dict" in raw:
            return raw["model_state_dict"]
        return raw
