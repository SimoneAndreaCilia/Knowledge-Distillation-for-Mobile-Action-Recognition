# -*- coding: utf-8 -*-
"""Single inference tab — Tab 1 layout and event wiring.

Responsible only for defining the Gradio component layout and connecting
the components to the appropriate callback handler methods.  Contains zero
business logic.

Usage::

    components = build_single_inference_tab(
        demo, inference_handler, dataset_handler, registry, classes
    )
"""

from typing import Dict, List, Optional

import gradio as gr

from src.gui.callbacks.dataset_callbacks import DatasetCallbackHandler
from src.gui.callbacks.inference_callbacks import InferenceCallbackHandler
from src.gui.ui.components import build_video_input_section
from src.repositories.model_registry import ModelRegistry


def build_single_inference_tab(
    demo: gr.Blocks,
    inference_handler: InferenceCallbackHandler,
    dataset_handler: DatasetCallbackHandler,
    registry: ModelRegistry,
    classes: List[str],
    default_class: Optional[str],
) -> None:
    """Build Tab 1: Single Model Inference.

    Constructs the layout and wires all event listeners.  All business
    logic is delegated to the injected handler objects.

    Args:
        demo:              The parent ``gr.Blocks`` instance (for ``demo.load``).
        inference_handler: Handles the classify button click.
        dataset_handler:   Handles dataset browser and source toggle events.
        registry:          Provides initial model dropdown choices.
        classes:           HMDB-51 class list for the class dropdown.
        default_class:     Pre-selected class name (or None).
    """
    with gr.Tab("🔍 Inferenza Singola", id="single"):
        with gr.Row():
            # ---- Left: Video Input ----------------------------------------
            with gr.Column(scale=1):
                gr.Markdown("### 🎥 Video Input")

                (
                    video_source,
                    upload_section,
                    uploaded_video,
                    dataset_section,
                    dataset_class,
                    dataset_video,
                    video_preview,
                ) = build_video_input_section(classes, default_class)

            # ---- Right: Model + Results ------------------------------------
            with gr.Column(scale=1):
                gr.Markdown("### 🧠 Modello")

                show_advanced = gr.Checkbox(
                    label="📦 Mostra tutte le varianti (ablation)",
                    value=False,
                    interactive=True,
                )

                model_dropdown = gr.Dropdown(
                    label="Seleziona Modello",
                    choices=registry.keys(show_advanced=False),
                    value=registry.keys(show_advanced=False)[0],
                    interactive=True,
                    allow_custom_value=False,
                )

                classify_btn = gr.Button(
                    "🔍 Classifica",
                    variant="primary",
                    elem_classes="primary-btn",
                )

                gr.Markdown("### 📊 Risultati")

                status_output = gr.Markdown(
                    value="*In attesa di classificazione...*",
                    elem_classes="status-msg",
                )
                results_output = gr.Label(
                    label="Top-5 Predizioni",
                    num_top_classes=5,
                )
                model_info = gr.Textbox(
                    label="ℹ️ Informazioni Modello",
                    interactive=False,
                    lines=5,
                    elem_classes="model-info-box",
                )

        # ---- Event Wiring -------------------------------------------------

        video_source.change(
            fn=dataset_handler.toggle_video_source,
            inputs=[video_source],
            outputs=[upload_section, dataset_section],
        )

        dataset_class.change(
            fn=dataset_handler.update_videos,
            inputs=[dataset_class],
            outputs=[dataset_video],
        )

        dataset_video.change(
            fn=dataset_handler.get_preview_path,
            inputs=[dataset_class, dataset_video],
            outputs=[video_preview],
        )

        show_advanced.change(
            fn=dataset_handler.update_model_dropdown,
            inputs=[show_advanced],
            outputs=[model_dropdown],
        )

        classify_btn.click(
            fn=inference_handler.classify,
            inputs=[
                model_dropdown,
                uploaded_video,
                dataset_class,
                dataset_video,
                video_source,
            ],
            outputs=[results_output, model_info, status_output],
        )

        demo.load(
            fn=dataset_handler.update_videos,
            inputs=[dataset_class],
            outputs=[dataset_video],
        )
