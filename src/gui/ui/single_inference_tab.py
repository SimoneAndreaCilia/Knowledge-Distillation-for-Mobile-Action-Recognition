# -*- coding: utf-8 -*-
"""Single inference tab — Tab 1 layout and event wiring."""

from typing import Callable, Dict, List, Optional

import gradio as gr

from src.gui.callbacks.dataset_callbacks import DatasetCallbackHandler
from src.gui.callbacks.inference_callbacks import InferenceCallbackHandler
from src.gui.ui.components import VideoInputSection
from src.i18n.keys import TranslationKey
from src.i18n.translator import Translator
from src.repositories.model_registry import ModelRegistry


class SingleInferenceTab:
    """Builds Tab 1: Single Model Inference."""

    def __init__(
        self,
        demo: gr.Blocks,
        inference_handler: InferenceCallbackHandler,
        dataset_handler: DatasetCallbackHandler,
        registry: ModelRegistry,
        classes: List[str],
        default_class: Optional[str],
    ) -> None:
        self.demo = demo
        self.inference_handler = inference_handler
        self.dataset_handler = dataset_handler
        self.registry = registry
        self.classes = classes
        self.default_class = default_class

        # Components
        self.tab: Optional[gr.Tab] = None
        self.video_md: Optional[gr.Markdown] = None
        self.video_section: Optional[VideoInputSection] = None
        self.model_md: Optional[gr.Markdown] = None
        self.show_advanced: Optional[gr.Checkbox] = None
        self.model_dropdown: Optional[gr.Dropdown] = None
        self.classify_btn: Optional[gr.Button] = None
        self.results_md: Optional[gr.Markdown] = None
        self.status_output: Optional[gr.Markdown] = None
        self.results_output: Optional[gr.Label] = None
        self.model_info: Optional[gr.Textbox] = None

    def build(self, lang_state: gr.State) -> None:
        """Constructs the layout and wires events."""
        with gr.Tab(id="single") as self.tab:
            with gr.Row():
                # ---- Left: Video Input ----------------------------------------
                with gr.Column(scale=1):
                    self.video_md = gr.Markdown()
                    self.video_section = VideoInputSection(self.classes, self.default_class)
                    self.video_section.build()

                # ---- Right: Model + Results ------------------------------------
                with gr.Column(scale=1):
                    self.model_md = gr.Markdown()

                    self.show_advanced = gr.Checkbox(
                        value=False,
                        interactive=True,
                    )

                    self.model_dropdown = gr.Dropdown(
                        choices=self.registry.keys(show_advanced=False),
                        value=self.registry.keys(show_advanced=False)[0],
                        interactive=True,
                        allow_custom_value=False,
                    )

                    self.classify_btn = gr.Button(
                        variant="primary",
                        elem_classes="primary-btn",
                    )

                    self.results_md = gr.Markdown()

                    self.status_output = gr.Markdown(
                        elem_classes="status-msg",
                    )
                    
                    self.results_output = gr.Label(
                        num_top_classes=5,
                    )
                    
                    self.model_info = gr.Textbox(
                        interactive=False,
                        lines=5,
                        elem_classes="model-info-box",
                    )

            # ---- Event Wiring -------------------------------------------------
            self.video_section.video_source.change(
                fn=self.dataset_handler.toggle_video_source,
                inputs=[self.video_section.video_source],
                outputs=[self.video_section.upload_section, self.video_section.dataset_section],
            )

            self.video_section.dataset_class.change(
                fn=self.dataset_handler.update_videos_and_preview,
                inputs=[self.video_section.dataset_class],
                outputs=[self.video_section.dataset_video, self.video_section.video_preview],
            )

            self.video_section.dataset_video.change(
                fn=self.dataset_handler.get_preview_path,
                inputs=[self.video_section.dataset_class, self.video_section.dataset_video],
                outputs=[self.video_section.video_preview],
            )

            self.show_advanced.change(
                fn=self.dataset_handler.update_model_dropdown,
                inputs=[self.show_advanced],
                outputs=[self.model_dropdown],
            )

            self.classify_btn.click(
                fn=self.inference_handler.classify,
                inputs=[
                    self.model_dropdown,
                    self.video_section.uploaded_video,
                    self.video_section.dataset_class,
                    self.video_section.dataset_video,
                    self.video_section.video_source,
                    lang_state,
                ],
                outputs=[self.results_output, self.model_info, self.status_output],
            )

            self.demo.load(
                fn=self.dataset_handler.update_videos_and_preview,
                inputs=[self.video_section.dataset_class],
                outputs=[self.video_section.dataset_video, self.video_section.video_preview],
            )

    def get_language_updates(self, translator: Translator) -> Dict[gr.components.Component, Callable]:
        """Returns updater functions for this section."""
        updates = self.video_section.get_language_updates(translator)

        def update_tab(lang):
            return gr.update(label=translator.t(TranslationKey.TAB_SINGLE_INFERENCE, lang=lang))

        def update_video_md(lang):
            return gr.update(value=translator.t(TranslationKey.VIDEO_INPUT_TITLE, lang=lang))

        def update_model_md(lang):
            return gr.update(value=translator.t(TranslationKey.MODEL_TITLE, lang=lang))

        def update_show_advanced(lang):
            return gr.update(label=translator.t(TranslationKey.MODEL_SHOW_ADVANCED, lang=lang))

        def update_model_dropdown(lang):
            return gr.update(label=translator.t(TranslationKey.MODEL_SELECT, lang=lang))

        def update_classify_btn(lang):
            return gr.update(value=translator.t(TranslationKey.MODEL_CLASSIFY_BTN, lang=lang))

        def update_results_md(lang):
            return gr.update(value=translator.t(TranslationKey.RESULTS_TITLE, lang=lang))

        def update_status_output(lang):
            # Do not overwrite dynamic text if it's already running, but we can set placeholder
            # Actually, we should just update the placeholder if we can, or just update value if it is the waiting msg.
            # But Gradio status might be overwritten. We will just return the wait message for now.
            return gr.update(value=translator.t(TranslationKey.RESULTS_STATUS_WAITING, lang=lang))

        def update_results_output(lang):
            return gr.update(label=translator.t(TranslationKey.RESULTS_TOP5, lang=lang))

        def update_model_info(lang):
            return gr.update(label=translator.t(TranslationKey.RESULTS_MODEL_INFO, lang=lang))

        updates.update({
            self.tab: update_tab,
            self.video_md: update_video_md,
            self.model_md: update_model_md,
            self.show_advanced: update_show_advanced,
            self.model_dropdown: update_model_dropdown,
            self.classify_btn: update_classify_btn,
            self.results_md: update_results_md,
            self.status_output: update_status_output,
            self.results_output: update_results_output,
            self.model_info: update_model_info,
        })
        return updates
