# -*- coding: utf-8 -*-
"""Application settings — single source of truth for all path and device config.

All configurable values are centralised here.  No other module should
construct ``Path`` objects relative to ``__file__`` or call
``torch.cuda.is_available()`` directly.

Usage::

    from src.config.settings import get_settings

    cfg = get_settings()
    print(cfg.data_dir)
"""

import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_project_root() -> Path:
    """Locate the repository root by walking up from this file.

    Looks for the ``checkpoints/`` directory as a sentinel.  Falls back to
    three levels above this file if none is found.
    """
    candidate = Path(__file__).resolve().parent
    for _ in range(6):
        if (candidate / "checkpoints").exists():
            return candidate
        candidate = candidate.parent
    # Fallback: src/config/settings.py → src/config → src → project root
    return Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# AppSettings
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AppSettings:
    """Immutable application-wide configuration.

    Attributes:
        project_root:    Absolute path to the repository root.
        data_dir:        Path to the HMDB-51 dataset directory.
        checkpoints_dir: Path to the trained model checkpoints.
        device:          Torch device string (``"cuda"`` or ``"cpu"``).
        server_name:     Bind address for the Gradio server.
        server_port:     Port for the Gradio server.
        top_k:           Number of top predictions to return from inference.
    """

    project_root: Path
    data_dir: Path
    checkpoints_dir: Path
    device: str
    server_name: str = "0.0.0.0"
    server_port: int = 7860
    top_k: int = 5

    def __post_init__(self) -> None:
        # Ensure project root is on sys.path so src.* imports resolve
        root_str = str(self.project_root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)


def _build_settings() -> AppSettings:
    project_root = _find_project_root()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return AppSettings(
        project_root=project_root,
        data_dir=project_root / "data" / "hmdb51",
        checkpoints_dir=project_root / "checkpoints",
        device=device,
    )


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Return the singleton AppSettings instance (constructed once)."""
    return _build_settings()
