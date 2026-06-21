# -*- coding: utf-8 -*-
"""GUI Demo — Knowledge Distillation for Action Recognition
==========================================================

.. deprecated::
    This file now delegates entirely to :mod:`src.bootstrap.application`.
    All application logic lives in the modular Clean Architecture structure
    under ``src/``.  This file is kept only for backwards compatibility with
    the original ``python -m src.gui.app`` invocation.

Usage::

    python -m src.gui.app

    # Preferred new entry point:
    python src/main.py
"""

import sys
from pathlib import Path

# Ensure the project root is on sys.path when run as a script
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.bootstrap.application import ApplicationBuilder  # noqa: E402


def main() -> None:
    """Launch the Gradio demo server (legacy entry point)."""
    app = ApplicationBuilder().build()
    app.launch()


if __name__ == "__main__":
    main()
