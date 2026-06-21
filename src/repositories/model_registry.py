# -*- coding: utf-8 -*-
"""ModelRegistry — typed lookup for model configurations.

Abstracts over the raw ``MODELS_MAIN`` / ``MODELS_ADVANCED`` dictionaries
and provides a clean interface that the service layer uses.  The registry
is constructed with the config dicts injected at build time (DIP), making
it trivial to substitute test fixtures.

Usage::

    from src.config.model_configs import MODELS_MAIN, MODELS_ADVANCED
    from src.repositories.model_registry import ModelRegistry

    registry = ModelRegistry(MODELS_MAIN, MODELS_ADVANCED)
    cfg = registry.get("Teacher (ResNet3D-50) — 62.94%")
"""

import logging
from typing import Dict, List, Optional

from src.config.model_configs import ModelConfig

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Read-only registry of ModelConfig descriptors.

    Responsibilities (SRP):
        - Resolve a display-name key to a ``ModelConfig``.
        - Enumerate available keys given the ``show_advanced`` flag.

    Non-responsibilities:
        - Loading models from disk.
        - Caching loaded models.
        - Any training/evaluation logic.
    """

    def __init__(
        self,
        main_models: Dict[str, ModelConfig],
        advanced_models: Dict[str, ModelConfig],
    ) -> None:
        self._main = dict(main_models)
        self._advanced = dict(advanced_models)
        self._all = {**self._main, **self._advanced}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: str) -> ModelConfig:
        """Return the ModelConfig for *key*.

        Raises:
            KeyError: If *key* is not present in main or advanced models.
        """
        if key not in self._all:
            available = list(self._all.keys())
            raise KeyError(
                f"Unknown model key: '{key}'.  "
                f"Available: {available}"
            )
        return self._all[key]

    def find(self, key: str) -> Optional[ModelConfig]:
        """Return the ModelConfig for *key* or None if not found."""
        return self._all.get(key)

    def keys(self, show_advanced: bool = False) -> List[str]:
        """Return ordered list of model keys for the UI dropdown."""
        keys = list(self._main.keys())
        if show_advanced:
            keys.extend(self._advanced.keys())
        return keys

    def all_configs(self, show_advanced: bool = False) -> Dict[str, ModelConfig]:
        """Return the active model config dict."""
        result = dict(self._main)
        if show_advanced:
            result.update(self._advanced)
        return result

    def __len__(self) -> int:
        return len(self._all)

    def __repr__(self) -> str:
        return (
            f"ModelRegistry(main={len(self._main)}, "
            f"advanced={len(self._advanced)})"
        )
