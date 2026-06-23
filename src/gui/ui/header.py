# -*- coding: utf-8 -*-
"""Header section of the application."""

from typing import Callable, Dict

import gradio as gr

from src.i18n.keys import TranslationKey
from src.i18n.languages import Language
from src.i18n.translator import Translator


class Header:
    """Builds the main header and language selector."""

    def __init__(self) -> None:
        self.branding_md: gr.Markdown = None
        self.title_md: gr.Markdown = None
        self.language_selector: gr.Radio = None

    def build(self) -> None:
        """Constructs the layout."""
        with gr.Row(elem_classes="header-section"):
            # Left: Branding
            with gr.Column(scale=1, min_width=250):
                self.branding_md = gr.Markdown()
            
            # Center: Title & Subtitle
            with gr.Column(scale=3, min_width=300):
                self.title_md = gr.Markdown(elem_classes="text-center")
                
            # Right: Language Selector
            with gr.Column(scale=1, min_width=150):
                self.language_selector = gr.Radio(
                    choices=[], # Set dynamically
                    value=Language.IT.value,
                    interactive=True,
                    elem_classes="language-selector"
                )

    def get_language_updates(self, translator: Translator) -> Dict[gr.components.Component, Callable]:
        """Returns updater functions for this section."""
        def update_branding(lang):
            return gr.update(value=translator.t(TranslationKey.HEADER_BRANDING, lang=lang))

        def update_title(lang):
            title = translator.t(TranslationKey.HEADER_TITLE, lang=lang)
            subtitle = translator.t(TranslationKey.HEADER_SUBTITLE, lang=lang)
            return gr.update(value=f"# {title}\n{subtitle}")

        def update_language_selector(lang):
            return gr.update(
                label=translator.t(TranslationKey.HEADER_LANGUAGE, lang=lang),
                choices=[
                    ("🇮🇹 IT", Language.IT.value),
                    ("🇬🇧 EN", Language.EN.value)
                ]
            )

        return {
            self.branding_md: update_branding,
            self.title_md: update_title,
            self.language_selector: update_language_selector,
        }
