# -*- coding: utf-8 -*-
"""ModelMetadata DTO — typed descriptor for a model checkpoint."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.i18n.keys import TranslationKey

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
    description_key: TranslationKey

    @property
    def checkpoint_exists(self) -> bool:
        """Return True if the checkpoint file is present on disk."""
        return self.checkpoint.exists()

    def format_info(self, translator: "Translator", lang: "Language", elapsed: Optional[float] = None) -> str:
        """Build a human-readable info string for display in the UI."""
        from src.i18n.keys import TranslationKey
        
        desc = translator.t(self.description_key, lang=lang)
        lines = [f"📋 {desc}"]
        
        params_str = translator.t(TranslationKey.INFO_PARAMS, lang=lang, params=self.params_m)
        lines.append(params_str)
        
        size_str = translator.t(TranslationKey.INFO_SIZE, lang=lang, size=self.size_mb)
        lines.append(size_str)
        
        if self.accuracy is not None:
            acc_str = translator.t(TranslationKey.INFO_ACCURACY, lang=lang, accuracy=self.accuracy)
            lines.append(acc_str)
            
        if elapsed is not None:
            time_str = translator.t(TranslationKey.INFO_TIME, lang=lang, time=elapsed)
            lines.append(time_str)
            
        return "\n".join(lines)
