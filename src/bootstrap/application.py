# -*- coding: utf-8 -*-
"""ApplicationBuilder — the composition root of the Knowledge Distillation demo."""

import logging
import sys
from pathlib import Path

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
from src.gui.ui.header import Header
from src.gui.ui.footer import Footer
from src.gui.ui.single_inference_tab import SingleInferenceTab
from src.gui.ui.comparison_tab import ComparisonTab
from src.i18n.translator import Translator
from src.i18n.languages import Language

logger = logging.getLogger(__name__)


class GradioApp:
    """Thin wrapper around a built ``gr.Blocks`` interface."""

    def __init__(self, demo, theme, css: str, settings) -> None:
        self._demo = demo
        self._theme = theme
        self._css = css
        self._settings = settings

    def launch(self, **kwargs) -> None:
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
    """Constructs and wires the complete application object graph."""

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
        
        # Load I18n Translator
        i18n_dir = settings.project_root / "src" / "i18n" / "translations"
        translator = Translator(translations_dir=i18n_dir)

        # ---- 5. Visualization --------------------------------------------
        chart_builder = ComparisonChartBuilder()

        # ---- 6. Callback Handlers ----------------------------------------
        inference_handler = InferenceCallbackHandler(
            inference_service=inference_service,
            dataset_service=dataset_service,
            registry=registry,
            translator=translator,
        )
        comparison_handler = ComparisonCallbackHandler(
            comparison_service=comparison_service,
            dataset_service=dataset_service,
            chart_builder=chart_builder,
            translator=translator,
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
            lang_state = gr.State(Language.IT.value)
            
            # Header
            header = Header()
            header.build()

            # Tabs
            with gr.Tabs(elem_classes="tabs"):
                single_tab = SingleInferenceTab(
                    demo=demo,
                    inference_handler=inference_handler,
                    dataset_handler=dataset_handler,
                    registry=registry,
                    classes=classes,
                    default_class=default_class,
                )
                single_tab.build(lang_state=lang_state)

                comp_tab = ComparisonTab(
                    demo=demo,
                    comparison_handler=comparison_handler,
                    dataset_handler=dataset_handler,
                    classes=classes,
                    default_class=default_class,
                )
                comp_tab.build(lang_state=lang_state)

            # Footer
            footer = Footer(num_classes=len(class_names))
            footer.build()
            
            # Apply translations
            def update_ui(lang: str):
                header_updates = header.get_language_updates(translator)
                single_updates = single_tab.get_language_updates(translator)
                comp_updates = comp_tab.get_language_updates(translator)
                footer_updates = footer.get_language_updates(translator)
                
                res = []
                for updates in [header_updates, single_updates, comp_updates, footer_updates]:
                    for comp, update_func in updates.items():
                        res.append(update_func(Language(lang)))
                return tuple(res)
                
            ui_components = []
            for updates in [header.get_language_updates(translator), 
                            single_tab.get_language_updates(translator), 
                            comp_tab.get_language_updates(translator), 
                            footer.get_language_updates(translator)]:
                ui_components.extend(list(updates.keys()))
                
            demo.load(
                fn=update_ui,
                inputs=[lang_state],
                outputs=ui_components,
            )
            
            header.language_selector.change(
                fn=lambda lang: lang,
                inputs=[header.language_selector],
                outputs=[lang_state],
            ).then(
                fn=update_ui,
                inputs=[lang_state],
                outputs=ui_components,
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
