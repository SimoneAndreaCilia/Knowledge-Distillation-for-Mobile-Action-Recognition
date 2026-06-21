# -*- coding: utf-8 -*-
"""Configuration layer — application settings, constants, and model descriptors."""

from .settings import AppSettings, get_settings
from .model_configs import ModelConfig, MODELS_MAIN, MODELS_ADVANCED

__all__ = [
    "AppSettings",
    "get_settings",
    "ModelConfig",
    "MODELS_MAIN",
    "MODELS_ADVANCED",
]
