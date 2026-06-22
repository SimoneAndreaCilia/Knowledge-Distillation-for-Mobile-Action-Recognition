# -*- coding: utf-8 -*-
"""Footer section of the application."""

from typing import Callable, Dict

import gradio as gr
import torch

from src.i18n.keys import TranslationKey
from src.i18n.translator import Translator


class Footer:
    """Builds the dynamic footer."""

    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.device_label = "CUDA" if torch.cuda.is_available() else "CPU"
        self.footer_md: gr.Markdown = None

    def build(self) -> None:
        """Constructs the layout."""
        self.footer_md = gr.Markdown()

    def get_language_updates(self, translator: Translator) -> Dict[gr.components.Component, Callable]:
        """Returns updater functions for this section."""
        def update_footer(lang):
            return gr.update(
                value=translator.t(
                    TranslationKey.FOOTER_TEXT, 
                    lang=lang, 
                    device=self.device_label, 
                    num_classes=self.num_classes
                )
            )

        return {
            self.footer_md: update_footer,
        }
