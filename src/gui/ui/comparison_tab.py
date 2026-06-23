# -*- coding: utf-8 -*-
"""Comparison tab — Tab 2 layout and event wiring."""

from typing import Callable, Dict, List, Optional

import gradio as gr

from src.gui.callbacks.comparison_callbacks import ComparisonCallbackHandler
from src.gui.callbacks.dataset_callbacks import DatasetCallbackHandler
from src.gui.ui.components import VideoInputSection
from src.i18n.keys import TranslationKey
from src.i18n.translator import Translator


class ComparisonTab:
    """Builds Tab 2: Multi-Model Comparison."""

    def __init__(
        self,
        demo: gr.Blocks,
        comparison_handler: ComparisonCallbackHandler,
        dataset_handler: DatasetCallbackHandler,
        classes: List[str],
        default_class: Optional[str],
    ) -> None:
        self.demo = demo
        self.comparison_handler = comparison_handler
        self.dataset_handler = dataset_handler
        self.classes = classes
        self.default_class = default_class

        # Components
        self.tab: Optional[gr.Tab] = None
        self.desc_md: Optional[gr.Markdown] = None
        self.video_md: Optional[gr.Markdown] = None
        self.video_section: Optional[VideoInputSection] = None
        self.show_advanced: Optional[gr.Checkbox] = None
        self.compare_btn: Optional[gr.Button] = None
        self.results_md: Optional[gr.Markdown] = None
        self.comparison_plot: Optional[gr.Plot] = None
        self.comparison_summary: Optional[gr.Markdown] = None

    def build(self, lang_state: gr.State) -> None:
        """Constructs the layout and wires events."""
        with gr.Tab(id="compare") as self.tab:
            self.desc_md = gr.Markdown()

            with gr.Column(elem_classes="comparison-container"):
                # ---- Top: Video Input + Controls (Full Width) ------------------
                self.video_md = gr.Markdown()

                self.video_section = VideoInputSection(self.classes, self.default_class)
                self.video_section.build()

                with gr.Row():
                    self.show_advanced = gr.Checkbox(
                        value=False,
                        interactive=True,
                    )
                    self.compare_btn = gr.Button(
                        variant="primary",
                        elem_classes="primary-btn",
                    )

                # ---- KPI Row: Stripe-inspired Summary Cards -------------------
                self.kpi_cards = gr.HTML(elem_classes="kpi-row")

                # ---- Center: Results & Chart -----------------------------------
                self.results_md = gr.Markdown()
                self.comparison_plot = gr.Plot()
                
                # ---- Bottom: Best Model Highlight ------------------------------
                self.best_model_html = gr.HTML()

            # ---- Event Wiring -------------------------------------------------
            self.video_section.video_source.change(
                fn=self.dataset_handler.toggle_video_source,
                inputs=[self.video_section.video_source],
                outputs=[self.video_section.upload_section, self.video_section.dataset_section],
            )

            self.video_section.dataset_class.change(
                fn=self.dataset_handler.update_videos_and_preview,
                inputs=[self.video_section.dataset_class, self.video_section.dataset_split],
                outputs=[self.video_section.dataset_video, self.video_section.video_preview],
            )
            
            self.video_section.dataset_split.change(
                fn=self.dataset_handler.update_videos_and_preview,
                inputs=[self.video_section.dataset_class, self.video_section.dataset_split],
                outputs=[self.video_section.dataset_video, self.video_section.video_preview],
            )

            self.video_section.dataset_video.change(
                fn=self.dataset_handler.get_preview_path,
                inputs=[self.video_section.dataset_class, self.video_section.dataset_video],
                outputs=[self.video_section.video_preview],
            )

            self.compare_btn.click(
                fn=self.comparison_handler.compare,
                inputs=[
                    self.video_section.uploaded_video,
                    self.video_section.dataset_class,
                    self.video_section.dataset_video,
                    self.video_section.video_source,
                    self.show_advanced,
                    lang_state,
                ],
                outputs=[self.comparison_plot, self.kpi_cards, self.best_model_html],
            )

            self.demo.load(
                fn=self.dataset_handler.update_videos_and_preview,
                inputs=[self.video_section.dataset_class, self.video_section.dataset_split],
                outputs=[self.video_section.dataset_video, self.video_section.video_preview],
            )

    def get_language_updates(self, translator: Translator) -> Dict[gr.components.Component, Callable]:
        """Returns updater functions for this section."""
        updates = self.video_section.get_language_updates(translator)

        def update_tab(lang):
            return gr.update(label=translator.t(TranslationKey.TAB_COMPARISON, lang=lang))

        def update_desc_md(lang):
            return gr.update(value=translator.t(TranslationKey.COMP_DESC, lang=lang))

        def update_video_md(lang):
            return gr.update(value=translator.t(TranslationKey.VIDEO_INPUT_TITLE, lang=lang))

        def update_show_advanced(lang):
            return gr.update(label=translator.t(TranslationKey.MODEL_SHOW_ADVANCED, lang=lang))

        def update_compare_btn(lang):
            return gr.update(value=translator.t(TranslationKey.MODEL_COMPARE_BTN, lang=lang))

        def update_results_md(lang):
            return gr.update(value=translator.t(TranslationKey.RESULTS_TITLE, lang=lang))

        def update_comparison_plot(lang):
            return gr.update(label=translator.t(TranslationKey.COMP_PLOT, lang=lang))

        def update_kpi_cards(lang):
            return gr.update(value=f"<div style='text-align: center; color: #718096;'>{translator.t(TranslationKey.COMP_STATUS_WAITING, lang=lang)}</div>")

        updates.update({
            self.tab: update_tab,
            self.desc_md: update_desc_md,
            self.video_md: update_video_md,
            self.show_advanced: update_show_advanced,
            self.compare_btn: update_compare_btn,
            self.results_md: update_results_md,
            self.comparison_plot: update_comparison_plot,
            self.kpi_cards: update_kpi_cards,
        })
        return updates
