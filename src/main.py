# -*- coding: utf-8 -*-
"""Main entry point for the Knowledge Distillation Action Recognition demo.

Usage::

    python src/main.py

    # Or as a module:
    python -m src.main
"""

from src.bootstrap.application import ApplicationBuilder


def main() -> None:
    """Build and launch the Gradio application."""
    app = ApplicationBuilder().build()
    app.launch()


if __name__ == "__main__":
    main()
