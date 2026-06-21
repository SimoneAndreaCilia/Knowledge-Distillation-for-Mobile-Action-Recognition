# -*- coding: utf-8 -*-
"""ModelCache — in-memory cache for loaded torch model instances.

Replaces the module-level ``_model_cache: Dict[str, torch.nn.Module] = {}``
mutable global.  The cache is a first-class object that can be injected,
reset, or swapped (e.g. for an LRU variant) without touching any other module.

Usage::

    cache = ModelCache()
    if not cache.has(key):
        cache.put(key, model)
    model = cache.get(key)
"""

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ModelCache:
    """Thread-unsafe in-process cache for torch model instances.

    For a multi-worker Gradio deployment, consider replacing the internal
    dict with an LRU cache or a shared-memory store.

    Responsibilities (SRP):
        - Store loaded ``nn.Module`` instances keyed by model_key string.
        - Report whether a key is present.
        - Evict all cached models (useful for memory management).

    Non-responsibilities:
        - Building or loading models.
        - Any business logic.
    """

    def __init__(self) -> None:
        self._store: Dict[str, nn.Module] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def has(self, key: str) -> bool:
        """Return True if the model for *key* is already cached."""
        return key in self._store

    def get(self, key: str) -> Optional[nn.Module]:
        """Return the cached model or None if not present."""
        return self._store.get(key)

    def put(self, key: str, model: nn.Module) -> None:
        """Store *model* under *key*."""
        logger.debug("ModelCache: storing model for key '%s'", key)
        self._store[key] = model

    def clear(self) -> None:
        """Remove all cached models and free their memory."""
        count = len(self._store)
        self._store.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("ModelCache: cleared %d cached model(s)", count)

    def __len__(self) -> int:
        return len(self._store)

    def __repr__(self) -> str:
        keys = list(self._store.keys())
        return f"ModelCache(cached={keys})"
