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
        self.footer_md = gr.Markdown(elem_classes="academic-footer")

    def get_language_updates(self, translator: Translator) -> Dict[gr.components.Component, Callable]:
        """Returns updater functions for this section."""
        def update_footer(lang):
            html = (
                "**Knowledge Distillation for Mobile Action Recognition**<br>"
                "University Project · Deep Learning Course<br>"
                "Department of Mathematics and Computer Science · **University of Catania**<br>"
                f"<span style='color: #A0AEC0; font-size: 0.8rem;'>Running on {self.device_label} · HMDB-51 Dataset ({self.num_classes} classes)</span>"
            )
            return gr.update(value=html)

        return {
            self.footer_md: update_footer,
        }
