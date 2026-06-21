# -*- coding: utf-8 -*-
"""ModelMetadata DTO — typed descriptor for a model checkpoint."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ModelMetadata:
    """Complete metadata for a single model checkpoint.

    Replaces raw ``Dict[str, Any]`` entries that scattered ``model_name``,
    ``checkpoint``, ``params_m``, etc. across the codebase without any
    type guarantee.

    Attributes:
        name:         Human-readable display label (used as registry key).
        checkpoint:   Absolute path to the ``.pth`` checkpoint file.
        model_name:   Architecture identifier (``"teacher"`` / ``"student"``).
        width_mult:   Channel width multiplier (1.0 for standard student).
        params_m:     Number of parameters in millions.
        accuracy:     Top-1 accuracy on HMDB-51 test split, or None if unknown.
        size_mb:      Checkpoint file size in MB.
        description:  One-sentence human description of this configuration.
    """

    name: str
    checkpoint: Path
    model_name: str
    width_mult: float
    params_m: float
    accuracy: Optional[float]
    size_mb: float
    description: str

    @property
    def checkpoint_exists(self) -> bool:
        """Return True if the checkpoint file is present on disk."""
        return self.checkpoint.exists()

    def format_info(self, elapsed: Optional[float] = None) -> str:
        """Build a human-readable info string for display in the UI."""
        lines = [f"📋 {self.description}"]
        lines.append(f"⚙️ Parametri: {self.params_m}M")
        lines.append(f"💾 Dimensione: {self.size_mb} MB")
        if self.accuracy is not None:
            lines.append(f"📊 Accuracy test set: {self.accuracy}%")
        if elapsed is not None:
            lines.append(f"⏱️ Tempo inferenza: {elapsed:.2f}s")
        return "\n".join(lines)
