# -*- coding: utf-8 -*-
"""Comparison tab — Tab 2 layout and event wiring.

Responsible only for defining the Gradio component layout and connecting
components to the ComparisonCallbackHandler.  Contains zero business logic.

Usage::

    build_comparison_tab(demo, comparison_handler, dataset_handler, classes, default_class)
"""

from typing import List, Optional

import gradio as gr

from src.gui.callbacks.comparison_callbacks import ComparisonCallbackHandler
from src.gui.callbacks.dataset_callbacks import DatasetCallbackHandler
from src.gui.ui.components import build_video_input_section


def build_comparison_tab(
    demo: gr.Blocks,
    comparison_handler: ComparisonCallbackHandler,
    dataset_handler: DatasetCallbackHandler,
    classes: List[str],
    default_class: Optional[str],
) -> None:
    """Build Tab 2: Multi-Model Comparison.

    Constructs the layout and wires all event listeners for the model
    comparison tab.  All business logic is delegated to the handler objects.

    Args:
        demo:               Parent ``gr.Blocks`` instance (for ``demo.load``).
        comparison_handler: Handles the compare button click.
        dataset_handler:    Handles dataset browser events.
        classes:            HMDB-51 class list for the class dropdown.
        default_class:      Pre-selected class name (or None).
    """
    with gr.Tab("⚔️ Confronto Modelli", id="compare"):
        gr.Markdown(
            "### Confronta tutti i modelli sullo stesso video\n"
            "Esegue l'inferenza con tutti i modelli selezionati e mostra "
            "i risultati affiancati."
        )

        with gr.Row():
            # ---- Left: Video Input + Controls -----------------------------
            with gr.Column(scale=1):
                gr.Markdown("### 🎥 Video Input")

                (
                    cmp_video_source,
                    cmp_upload_section,
                    cmp_uploaded_video,
                    cmp_dataset_section,
                    cmp_dataset_class,
                    cmp_dataset_video,
                    cmp_video_preview,
                ) = build_video_input_section(classes, default_class)

                cmp_show_advanced = gr.Checkbox(
                    label="📦 Includi varianti avanzate",
                    value=False,
                    interactive=True,
                )

                compare_btn = gr.Button(
                    "⚔️ Confronta Tutti i Modelli",
                    variant="primary",
                    elem_classes="primary-btn",
                )

            # ---- Right: Results -------------------------------------------
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Risultati Confronto")

                comparison_plot = gr.Plot(
                    label="Confronto Top-1 Confidence",
                )
                comparison_summary = gr.Markdown(
                    value="*In attesa del confronto...*",
                    elem_classes="comparison-summary",
                )

        # ---- Event Wiring -------------------------------------------------

        cmp_video_source.change(
            fn=dataset_handler.toggle_video_source,
            inputs=[cmp_video_source],
            outputs=[cmp_upload_section, cmp_dataset_section],
        )

        cmp_dataset_class.change(
            fn=dataset_handler.update_videos,
            inputs=[cmp_dataset_class],
            outputs=[cmp_dataset_video],
        )

        cmp_dataset_video.change(
            fn=dataset_handler.get_preview_path,
            inputs=[cmp_dataset_class, cmp_dataset_video],
            outputs=[cmp_video_preview],
        )

        compare_btn.click(
            fn=comparison_handler.compare,
            inputs=[
                cmp_uploaded_video,
                cmp_dataset_class,
                cmp_dataset_video,
                cmp_video_source,
                cmp_show_advanced,
            ],
            outputs=[comparison_plot, comparison_summary],
        )

        demo.load(
            fn=dataset_handler.update_videos,
            inputs=[cmp_dataset_class],
            outputs=[cmp_dataset_video],
        )
