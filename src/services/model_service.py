# -*- coding: utf-8 -*-
"""ModelService — builds, loads, and caches torch model instances.

This is the single place in the codebase that calls ``build_model()``,
``load_state_dict()``, ``.to(device)``, and ``.eval()``.  All other
layers receive a ready-to-use ``nn.Module`` from this service.

Usage::

    service = ModelService(registry, checkpoint_repo, cache, num_classes, device)
    model = service.get_or_load("Teacher (ResNet3D-50) — 62.94%")
"""

import logging

import torch
import torch.nn as nn

from src.cache.model_cache import ModelCache
from src.config.model_configs import ModelConfig
from src.repositories.checkpoint_repository import CheckpointRepository
from src.repositories.model_registry import ModelRegistry

logger = logging.getLogger(__name__)


class ModelService:
    """Provides ready-to-use model instances with transparent caching.

    Responsibilities (SRP):
        - Build a model architecture via ``build_model()``.
        - Load weights from disk via ``CheckpointRepository``.
        - Move the model to the target device.
        - Set the model to ``eval()`` mode.
        - Delegate caching to ``ModelCache``.

    Non-responsibilities:
        - Running inference.
        - Managing checkpoint paths (that's ``CheckpointRepository``).
        - UI logic of any kind.

    Args:
        registry:        Source of ``ModelConfig`` descriptors.
        checkpoint_repo: Loads state dicts from disk.
        cache:           Stores loaded model instances.
        num_classes:     Number of output classes for the model head.
        device:          Torch device string (``"cuda"`` or ``"cpu"``).
    """

    def __init__(
        self,
        registry: ModelRegistry,
        checkpoint_repo: CheckpointRepository,
        cache: ModelCache,
        num_classes: int,
        device: str,
    ) -> None:
        self._registry = registry
        self._checkpoint_repo = checkpoint_repo
        self._cache = cache
        self._num_classes = num_classes
        self._device = torch.device(device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_or_load(self, model_key: str) -> nn.Module:
        """Return a cached model or load it from disk on first call.

        Args:
            model_key: Display-name key present in the model registry.

        Returns:
            The model in ``eval()`` mode on the target device.

        Raises:
            KeyError:           If ``model_key`` is not in the registry.
            FileNotFoundError:  If the checkpoint file does not exist.
        """
        if self._cache.has(model_key):
            logger.debug("ModelService: cache hit for '%s'", model_key)
            return self._cache.get(model_key)  # type: ignore[return-value]

        config: ModelConfig = self._registry.get(model_key)
        model = self._build_and_load(config)
        self._cache.put(model_key, model)
        return model

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_and_load(self, config: ModelConfig) -> nn.Module:
        """Build the architecture and restore weights from the checkpoint."""
        # Import deferred so that torch is not required for importing the
        # service module itself in environments without a GPU stack.
        from src.models import build_model  # noqa: PLC0415

        logger.info(
            "ModelService: building '%s' (%s, width_mult=%.1f)",
            config.name,
            config.model_name,
            config.width_mult,
        )

        model = build_model(
            model_name=config.model_name,
            num_classes=self._num_classes,
            pretrained=False,
            width_mult=config.width_mult,
        )

        state_dict = self._checkpoint_repo.load_state_dict(config.checkpoint)
        model.load_state_dict(state_dict)

        model.to(self._device)
        model.eval()

        logger.info("ModelService: '%s' ready on %s", config.name, self._device)
        return model
