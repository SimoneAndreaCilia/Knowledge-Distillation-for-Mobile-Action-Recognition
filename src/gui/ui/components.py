# -*- coding: utf-8 -*-
"""Reusable widget factory helpers for common UI patterns.

These helpers eliminate duplication between the single-inference tab and the
comparison tab, both of which need a video-source selector + dataset browser.

Usage::

    from src.gui.ui.components import build_video_input_section
    components = build_video_input_section(classes, default_class)
"""

from typing import List, Optional, Tuple

import gradio as gr


def build_video_input_section(
    classes: List[str],
    default_class: Optional[str],
) -> Tuple[gr.Radio, gr.Column, gr.Video, gr.Column, gr.Dropdown, gr.Dropdown, gr.Video]:
    """Build the reusable video input section (source selector + panels).

    Creates:
      - A radio button to switch between ``📤 Upload`` and ``📂 Dataset HMDB-51``.
      - An upload panel (initially hidden).
      - A dataset browser panel (initially visible) with class/video dropdowns
        and a preview player.

    Args:
        classes:       List of HMDB-51 class names for the class dropdown.
        default_class: Pre-selected class (or None if the dataset is missing).

    Returns:
        A 7-tuple of Gradio components:
        ``(video_source, upload_section, uploaded_video,
           dataset_section, dataset_class, dataset_video, video_preview)``
    """
    video_source = gr.Radio(
        label="Sorgente Video",
        choices=["📤 Upload", "📂 Dataset HMDB-51"],
        value="📂 Dataset HMDB-51",
        interactive=True,
    )

    with gr.Column(visible=False) as upload_section:
        uploaded_video = gr.Video(
            label="Carica un video",
            sources=["upload"],
            elem_classes="video-preview",
        )

    with gr.Column(visible=True) as dataset_section:
        dataset_class = gr.Dropdown(
            label="Classe Azione",
            choices=classes,
            value=default_class,
            interactive=True,
            allow_custom_value=False,
        )
        dataset_video = gr.Dropdown(
            label="Video",
            choices=[],
            interactive=True,
            allow_custom_value=False,
        )
        video_preview = gr.Video(
            label="Anteprima",
            interactive=False,
            elem_classes="video-preview",
        )

    return (
        video_source,
        upload_section,
        uploaded_video,
        dataset_section,
        dataset_class,
        dataset_video,
        video_preview,
    )
