# -*- coding: utf-8 -*-
"""Reusable widget factory helpers for common UI patterns.

These helpers eliminate duplication between the single-inference tab and the
comparison tab, both of which need a video-source selector + dataset browser.
"""

from typing import Callable, Dict, List, Optional

import gradio as gr

from src.i18n.keys import TranslationKey
from src.i18n.translator import Translator


class VideoInputSection:
    """Builds the reusable video input section (source selector + panels)."""

    def __init__(self, classes: List[str], default_class: Optional[str]) -> None:
        self.classes = classes
        self.default_class = default_class

        # Components
        self.video_source: Optional[gr.Radio] = None
        self.upload_section: Optional[gr.Column] = None
        self.uploaded_video: Optional[gr.Video] = None
        self.dataset_section: Optional[gr.Column] = None
        self.dataset_class: Optional[gr.Dropdown] = None
        self.dataset_split: Optional[gr.Dropdown] = None
        self.dataset_video: Optional[gr.Dropdown] = None
        self.video_preview: Optional[gr.Video] = None

    def build(self) -> None:
        """Constructs the layout."""
        self.video_source = gr.Radio(
            choices=[], # Set dynamically
            value="dataset",
            interactive=True,
        )

        with gr.Column(visible=False) as self.upload_section:
            self.uploaded_video = gr.Video(
                sources=["upload"],
                elem_classes="video-preview",
            )

        with gr.Column(visible=True) as self.dataset_section:
            with gr.Row():
                self.dataset_class = gr.Dropdown(
                    choices=self.classes,
                    value=self.default_class,
                    interactive=True,
                    allow_custom_value=False,
                )
                self.dataset_split = gr.Dropdown(
                    choices=[
                        ("All Videos", "all"),
                        ("Test Split 1", "split1"),
                        ("Test Split 2", "split2"),
                        ("Test Split 3", "split3")
                    ],
                    value="all",
                    interactive=True,
                    allow_custom_value=False,
                )
                self.dataset_video = gr.Dropdown(
                    choices=[],
                    interactive=True,
                    allow_custom_value=False,
                )
            self.video_preview = gr.Video(
                interactive=False,
                elem_classes="video-preview",
            )

    def get_language_updates(self, translator: Translator) -> Dict[gr.components.Component, Callable]:
        """Returns updater functions for this section."""
        def update_video_source(lang):
            return gr.update(
                label=translator.t(TranslationKey.VIDEO_INPUT_SOURCE, lang=lang),
                choices=[
                    (translator.t(TranslationKey.VIDEO_INPUT_UPLOAD_CHOICE, lang=lang), "upload"),
                    (translator.t(TranslationKey.VIDEO_INPUT_DATASET_CHOICE, lang=lang), "dataset")
                ]
            )

        def update_uploaded_video(lang):
            return gr.update(label=translator.t(TranslationKey.VIDEO_INPUT_UPLOAD, lang=lang))

        def update_dataset_class(lang):
            return gr.update(label=translator.t(TranslationKey.VIDEO_INPUT_CLASS, lang=lang))

        def update_dataset_split(lang):
            return gr.update(
                label=translator.t(TranslationKey.VIDEO_INPUT_SPLIT, lang=lang),
                choices=[
                    (translator.t(TranslationKey.DATASET_SPLIT_ALL, lang=lang), "all"),
                    (translator.t(TranslationKey.DATASET_SPLIT_1, lang=lang), "split1"),
                    (translator.t(TranslationKey.DATASET_SPLIT_2, lang=lang), "split2"),
                    (translator.t(TranslationKey.DATASET_SPLIT_3, lang=lang), "split3")
                ]
            )

        def update_dataset_video(lang):
            return gr.update(label=translator.t(TranslationKey.VIDEO_INPUT_VIDEO, lang=lang))

        def update_video_preview(lang):
            return gr.update(label=translator.t(TranslationKey.VIDEO_INPUT_PREVIEW, lang=lang))

        return {
            self.video_source: update_video_source,
            self.uploaded_video: update_uploaded_video,
            self.dataset_class: update_dataset_class,
            self.dataset_split: update_dataset_split,
            self.dataset_video: update_dataset_video,
            self.video_preview: update_video_preview,
        }
