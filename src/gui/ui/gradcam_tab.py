# -*- coding: utf-8 -*-
"""Grad-CAM visualisation tab — Tab 3 layout and event wiring."""

from typing import Callable, Dict, List, Optional

import gradio as gr

from src.gui.callbacks.gradcam_callbacks import GradCamCallbackHandler
from src.gui.callbacks.dataset_callbacks import DatasetCallbackHandler
from src.gui.ui.components import VideoInputSection
from src.i18n.keys import TranslationKey
from src.i18n.translator import Translator
from src.repositories.model_registry import ModelRegistry


class GradCamTab:
    """Builds Tab 3: Attention Visualization (Grad-CAM)."""

    def __init__(
        self,
        demo: gr.Blocks,
        gradcam_handler: GradCamCallbackHandler,
        dataset_handler: DatasetCallbackHandler,
        registry: ModelRegistry,
        classes: List[str],
        default_class: Optional[str],
    ) -> None:
        self.demo = demo
        self.gradcam_handler = gradcam_handler
        self.dataset_handler = dataset_handler
        self.registry = registry
        self.classes = classes
        self.default_class = default_class

        self.models_main = registry.keys(show_advanced=False)
        self.models_all = registry.keys(show_advanced=True)

        # Components
        self.tab: Optional[gr.Tab] = None
        self.video_section: Optional[VideoInputSection] = None
        self.settings_md: Optional[gr.Markdown] = None
        self.model1_dropdown: Optional[gr.Dropdown] = None
        self.layer1_dropdown: Optional[gr.Dropdown] = None
        self.model2_dropdown: Optional[gr.Dropdown] = None
        self.layer2_dropdown: Optional[gr.Dropdown] = None

        self.class_mode_dropdown: Optional[gr.Radio] = None
        self.manual_class_dropdown: Optional[gr.Dropdown] = None

        self.generate_btn: Optional[gr.Button] = None
        self.video1_out: Optional[gr.Video] = None
        self.video2_out: Optional[gr.Video] = None

    def build(self, lang_state: gr.State, translator: Optional[Translator] = None) -> None:
        """Constructs the layout and wires events."""
        # Retrieve the initial language value from the State object
        # gr.State stores its value in .value
        init_lang = getattr(lang_state, "value", "en")

        def t(key: TranslationKey) -> str:
            if translator is not None:
                return translator.t(key, lang=init_lang)
            return ""  # will be filled by demo.load → get_language_updates

        with gr.Tab(
            id="gradcam",
            label=t(TranslationKey.TAB_GRADCAM),
        ) as self.tab:

            with gr.Column():
                # ---- Top: Video Input ----
                self.video_section = VideoInputSection(self.classes, self.default_class)
                self.video_section.build()

                # ---- Middle: Settings ----
                self.settings_md = gr.Markdown(t(TranslationKey.GRADCAM_SETTINGS_TITLE))

                with gr.Row():
                    with gr.Column(scale=1):
                        self.model1_dropdown = gr.Dropdown(
                            label=t(TranslationKey.GRADCAM_MODEL1_LABEL),
                            choices=self.models_all,
                            value=self.models_main[0] if self.models_main else None,
                            interactive=True,
                        )
                        self.layer1_dropdown = gr.Dropdown(
                            label=t(TranslationKey.GRADCAM_LAYER1_LABEL),
                            info=t(TranslationKey.GRADCAM_LAYER_INFO),
                            choices=["layer4", "layer3", "stages[-1]", "stages[4]"],
                            value="layer4",
                            interactive=True,
                            allow_custom_value=True,
                        )

                    with gr.Column(scale=1):
                        self.model2_dropdown = gr.Dropdown(
                            label=t(TranslationKey.GRADCAM_MODEL2_LABEL),
                            choices=self.models_all,
                            value=self.models_main[1] if len(self.models_main) > 1 else None,
                            interactive=True,
                        )
                        self.layer2_dropdown = gr.Dropdown(
                            label=t(TranslationKey.GRADCAM_LAYER2_LABEL),
                            info=t(TranslationKey.GRADCAM_LAYER_INFO),
                            choices=["layer4", "layer3", "stages[-1]", "stages[4]"],
                            value="stages[-1]",
                            interactive=True,
                            allow_custom_value=True,
                        )

                with gr.Row():
                    self.class_mode_dropdown = gr.Radio(
                        label=t(TranslationKey.GRADCAM_CLASS_MODE_LABEL),
                        info=t(TranslationKey.GRADCAM_CLASS_MODE_INFO),
                        choices=[
                            t(TranslationKey.GRADCAM_CLASS_MODE_GROUND_TRUTH),
                            t(TranslationKey.GRADCAM_CLASS_MODE_PREDICTED),
                            t(TranslationKey.GRADCAM_CLASS_MODE_MANUAL),
                        ],
                        value=t(TranslationKey.GRADCAM_CLASS_MODE_GROUND_TRUTH),
                        interactive=True,
                    )
                    self.manual_class_dropdown = gr.Dropdown(
                        label=t(TranslationKey.GRADCAM_MANUAL_CLASS_LABEL),
                        info=t(TranslationKey.GRADCAM_MANUAL_CLASS_INFO),
                        choices=self.classes,
                        value=self.default_class,
                        interactive=True,
                        visible=False,
                    )

                self.generate_btn = gr.Button(
                    t(TranslationKey.GRADCAM_GENERATE_BTN),
                    variant="primary",
                )

                # ---- Bottom: Output ----
                with gr.Row():
                    with gr.Column(scale=1):
                        self.video1_out = gr.Video(
                            label=t(TranslationKey.GRADCAM_OUTPUT1_LABEL),
                        )
                    with gr.Column(scale=1):
                        self.video2_out = gr.Video(
                            label=t(TranslationKey.GRADCAM_OUTPUT2_LABEL),
                        )

            # ---- Event Wiring ----
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

            def toggle_manual_class(mode):
                return gr.update(visible=(mode == "manual"))

            self.class_mode_dropdown.change(
                fn=toggle_manual_class,
                inputs=[self.class_mode_dropdown],
                outputs=[self.manual_class_dropdown],
            )

            self.generate_btn.click(
                fn=self.gradcam_handler.generate,
                inputs=[
                    self.video_section.uploaded_video,
                    self.video_section.dataset_class,
                    self.video_section.dataset_video,
                    self.video_section.video_source,
                    self.model1_dropdown,
                    self.layer1_dropdown,
                    self.model2_dropdown,
                    self.layer2_dropdown,
                    self.class_mode_dropdown,
                    self.manual_class_dropdown,
                    lang_state,
                ],
                outputs=[self.video1_out, self.video2_out],
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
            return gr.update(label=translator.t(TranslationKey.TAB_GRADCAM, lang=lang))

        def update_settings_md(lang):
            return gr.update(value=translator.t(TranslationKey.GRADCAM_SETTINGS_TITLE, lang=lang))

        def update_model1_dropdown(lang):
            return gr.update(label=translator.t(TranslationKey.GRADCAM_MODEL1_LABEL, lang=lang))

        def update_layer1_dropdown(lang):
            return gr.update(
                label=translator.t(TranslationKey.GRADCAM_LAYER1_LABEL, lang=lang),
                info=translator.t(TranslationKey.GRADCAM_LAYER_INFO, lang=lang),
            )

        def update_model2_dropdown(lang):
            return gr.update(label=translator.t(TranslationKey.GRADCAM_MODEL2_LABEL, lang=lang))

        def update_layer2_dropdown(lang):
            return gr.update(
                label=translator.t(TranslationKey.GRADCAM_LAYER2_LABEL, lang=lang),
                info=translator.t(TranslationKey.GRADCAM_LAYER_INFO, lang=lang),
            )

        def update_class_mode_dropdown(lang):
            return gr.update(
                label=translator.t(TranslationKey.GRADCAM_CLASS_MODE_LABEL, lang=lang),
                info=translator.t(TranslationKey.GRADCAM_CLASS_MODE_INFO, lang=lang),
                choices=[
                    translator.t(TranslationKey.GRADCAM_CLASS_MODE_GROUND_TRUTH, lang=lang),
                    translator.t(TranslationKey.GRADCAM_CLASS_MODE_PREDICTED, lang=lang),
                    translator.t(TranslationKey.GRADCAM_CLASS_MODE_MANUAL, lang=lang),
                ],
            )

        def update_manual_class_dropdown(lang):
            return gr.update(
                label=translator.t(TranslationKey.GRADCAM_MANUAL_CLASS_LABEL, lang=lang),
                info=translator.t(TranslationKey.GRADCAM_MANUAL_CLASS_INFO, lang=lang),
            )

        def update_generate_btn(lang):
            return gr.update(value=translator.t(TranslationKey.GRADCAM_GENERATE_BTN, lang=lang))

        def update_video1_out(lang):
            return gr.update(label=translator.t(TranslationKey.GRADCAM_OUTPUT1_LABEL, lang=lang))

        def update_video2_out(lang):
            return gr.update(label=translator.t(TranslationKey.GRADCAM_OUTPUT2_LABEL, lang=lang))

        updates.update({
            self.tab: update_tab,
            self.settings_md: update_settings_md,
            self.model1_dropdown: update_model1_dropdown,
            self.layer1_dropdown: update_layer1_dropdown,
            self.model2_dropdown: update_model2_dropdown,
            self.layer2_dropdown: update_layer2_dropdown,
            self.class_mode_dropdown: update_class_mode_dropdown,
            self.manual_class_dropdown: update_manual_class_dropdown,
            self.generate_btn: update_generate_btn,
            self.video1_out: update_video1_out,
            self.video2_out: update_video2_out,
        })
        return updates
