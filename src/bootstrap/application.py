# -*- coding: utf-8 -*-
"""ApplicationBuilder — the composition root of the Knowledge Distillation demo.

This module is the **only** place in the entire codebase that wires
dependencies together.  It follows the Dependency Inversion Principle:
high-level modules (UI, callbacks) declare what they need (interfaces /
protocols); this builder constructs the concrete implementations and
injects them.

No other module should instantiate services or repositories directly.

Usage::

    from src.bootstrap.application import ApplicationBuilder

    app = ApplicationBuilder().build()
    app.launch()
"""

import logging
import sys
import torch

from src.cache.model_cache import ModelCache
from src.config.constants import get_class_names, get_num_classes
from src.config.model_configs import MODELS_MAIN, MODELS_ADVANCED
from src.config.settings import get_settings
from src.repositories.checkpoint_repository import CheckpointRepository
from src.repositories.model_registry import ModelRegistry
from src.services.comparison_service import ComparisonService
from src.services.dataset_service import DatasetService
from src.services.inference_service import InferenceService
from src.services.model_service import ModelService
from src.visualization.comparison_chart import ComparisonChartBuilder
from src.gui.callbacks.dataset_callbacks import DatasetCallbackHandler
from src.gui.callbacks.inference_callbacks import InferenceCallbackHandler
from src.gui.callbacks.comparison_callbacks import ComparisonCallbackHandler
from src.gui.ui.theme import build_theme
from src.gui.ui.styles import CUSTOM_CSS
from src.gui.ui.single_inference_tab import build_single_inference_tab
from src.gui.ui.comparison_tab import build_comparison_tab

logger = logging.getLogger(__name__)


class GradioApp:
    """Thin wrapper around a built ``gr.Blocks`` interface.

    Encapsulates Gradio launch parameters so that ``main.py`` stays minimal.

    Args:
        demo:   The fully constructed ``gr.Blocks`` instance.
        theme:  The Gradio theme to apply at launch.
        css:    Custom CSS string to inject.
        settings: Application settings (for server host/port).
    """

    def __init__(self, demo, theme, css: str, settings) -> None:
        self._demo = demo
        self._theme = theme
        self._css = css
        self._settings = settings

    def launch(self, **kwargs) -> None:
        """Launch the Gradio server with the configured settings.

        Any keyword arguments override the defaults from ``AppSettings``.
        """
        params = {
            "server_name": self._settings.server_name,
            "server_port": self._settings.server_port,
            "share": False,
            "show_error": True,
            "favicon_path": None,
            "theme": self._theme,
            "css": self._css,
        }
        params.update(kwargs)
        logger.info(
            "Launching Gradio on %s:%d",
            params["server_name"],
            params["server_port"],
        )
        self._demo.launch(**params)


class ApplicationBuilder:
    """Constructs and wires the complete application object graph.

    Call ``build()`` once to get a ``GradioApp`` ready to ``launch()``.

    The build sequence is strictly ordered bottom-up:
      1. Settings (no deps)
      2. Repositories (depend on settings/config)
      3. Cache (no deps)
      4. Services (depend on repositories + cache)
      5. Visualization (depends on domain objects only)
      6. Callbacks (depend on services + visualization)
      7. UI / Gradio layout (depends on callbacks)
    """

    def build(self) -> GradioApp:
        """Build and return the fully wired ``GradioApp``."""
        self._configure_logging()

        # ---- 1. Settings -------------------------------------------------
        settings = get_settings()
        class_names = get_class_names()
        num_classes = get_num_classes()

        logger.info("Project root : %s", settings.project_root)
        logger.info("Device       : %s", settings.device)
        logger.info("Classes found: %d", len(class_names))

        # ---- 2. Repositories ---------------------------------------------
        registry = ModelRegistry(MODELS_MAIN, MODELS_ADVANCED)
        checkpoint_repo = CheckpointRepository()

        logger.info("Registry     : %r", registry)

        # ---- 3. Cache ----------------------------------------------------
        cache = ModelCache()

        # ---- 4. Services -------------------------------------------------
        model_service = ModelService(
            registry=registry,
            checkpoint_repo=checkpoint_repo,
            cache=cache,
            num_classes=num_classes,
            device=settings.device,
        )
        inference_service = InferenceService(
            model_service=model_service,
            class_names=class_names,
            device=settings.device,
            top_k=settings.top_k,
        )
        dataset_service = DatasetService(data_dir=settings.data_dir)
        comparison_service = ComparisonService(
            inference_service=inference_service,
            registry=registry,
        )

        # ---- 5. Visualization --------------------------------------------
        chart_builder = ComparisonChartBuilder()

        # ---- 6. Callback Handlers ----------------------------------------
        inference_handler = InferenceCallbackHandler(
            inference_service=inference_service,
            dataset_service=dataset_service,
            registry=registry,
        )
        comparison_handler = ComparisonCallbackHandler(
            comparison_service=comparison_service,
            dataset_service=dataset_service,
            chart_builder=chart_builder,
        )
        dataset_handler = DatasetCallbackHandler(
            dataset_service=dataset_service,
            registry=registry,
        )

        # ---- 7. Gradio UI ------------------------------------------------
        theme = build_theme()
        classes = dataset_service.get_classes()
        default_class = classes[0] if classes else None

        import gradio as gr  # noqa: PLC0415

        with gr.Blocks(title="KD Action Recognition Demo") as demo:

            # Header
            with gr.Column(elem_classes="header-section"):
                gr.Markdown(
                    "# 🎬 Knowledge Distillation — Action Recognition\n"
                    "Confronta Teacher, Student e modelli Distillati "
                    "su video HMDB-51 in tempo reale"
                )

            # Tabs
            with gr.Tabs(elem_classes="tabs"):
                build_single_inference_tab(
                    demo=demo,
                    inference_handler=inference_handler,
                    dataset_handler=dataset_handler,
                    registry=registry,
                    classes=classes,
                    default_class=default_class,
                )
                build_comparison_tab(
                    demo=demo,
                    comparison_handler=comparison_handler,
                    dataset_handler=dataset_handler,
                    classes=classes,
                    default_class=default_class,
                )

            # Footer
            device_label = "CUDA" if torch.cuda.is_available() else "CPU"
            gr.Markdown(
                "<div style='text-align:center; color:#666; padding:20px 0; "
                "font-size:0.85rem;'>"
                "Knowledge Distillation for Mobile Action Recognition  ·  "
                f"Device: <code>{device_label}</code>  ·  "
                f"Dataset: HMDB-51 ({len(class_names)} classi)"
                "</div>"
            )

        return GradioApp(
            demo=demo,
            theme=theme,
            css=CUSTOM_CSS,
            settings=settings,
        )

    @staticmethod
    def _configure_logging() -> None:
        """Configure root logger if not already set up."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            stream=sys.stdout,
        )
