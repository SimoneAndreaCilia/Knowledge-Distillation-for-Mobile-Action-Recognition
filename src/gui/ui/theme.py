# -*- coding: utf-8 -*-
"""Gradio theme definition for the Knowledge Distillation demo.

Isolates theme configuration so that ``build_ui()`` stays short and the
theme can be changed or tested independently.

Usage::

    from src.gui.ui.theme import build_theme
    theme = build_theme()
"""


def build_theme():
    """Build and return the custom Gradio light theme.

    Returns:
        A configured ``gr.themes.Base`` instance.
    """
    import gradio as gr  # noqa: PLC0415

    return gr.themes.Base(
        primary_hue=gr.themes.colors.orange,
        secondary_hue=gr.themes.colors.blue,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
    ).set(
        # Backgrounds
        body_background_fill="#F7F8FA",
        body_background_fill_dark="#F7F8FA",
        block_background_fill="#FFFFFF",
        block_background_fill_dark="#FFFFFF",
        background_fill_primary="#FFFFFF",
        background_fill_primary_dark="#FFFFFF",
        background_fill_secondary="#F7F8FA",
        background_fill_secondary_dark="#F7F8FA",
        # Borders (minimal)
        block_border_color="transparent",
        block_border_color_dark="transparent",
        block_border_width="0px",
        block_border_width_dark="0px",
        block_radius="8px",
        # Typography
        block_label_text_color="#4A5568",
        block_label_text_color_dark="#4A5568",
        block_title_text_color="#1A202C",
        block_title_text_color_dark="#1A202C",
        body_text_color="#1A202C",
        body_text_color_dark="#1A202C",
        body_text_color_subdued="#718096",
        body_text_color_subdued_dark="#718096",
        # Buttons
        button_primary_background_fill="#F05A28",
        button_primary_background_fill_dark="#F05A28",
        button_primary_text_color="white",
        button_primary_text_color_dark="white",
        button_secondary_background_fill="#FFFFFF",
        button_secondary_background_fill_dark="#FFFFFF",
        button_secondary_border_color="#E2E8F0",
        button_secondary_border_color_dark="#E2E8F0",
        # Inputs
        border_color_primary="#F05A28",
        border_color_primary_dark="#F05A28",
        input_background_fill="#FFFFFF",
        input_background_fill_dark="#FFFFFF",
        input_border_color="#E2E8F0",
        input_border_color_dark="#E2E8F0",
        # Loader
        loader_color="#F05A28",
        loader_color_dark="#F05A28",
        # Shadows
        block_shadow="none",
        block_shadow_dark="none",
        # Checkboxes & Radios
        checkbox_background_color="#FFFFFF",
        checkbox_background_color_dark="#FFFFFF",
        checkbox_border_color="#E2E8F0",
        checkbox_border_color_dark="#E2E8F0",
        checkbox_label_background_fill="#FFFFFF",
        checkbox_label_background_fill_dark="#FFFFFF",
        checkbox_label_background_fill_hover="#EDF2F7",
        checkbox_label_background_fill_hover_dark="#EDF2F7",
        checkbox_label_background_fill_selected="#FFFFFF",
        checkbox_label_background_fill_selected_dark="#FFFFFF",
        checkbox_label_border_color_hover="#CBD5E0",
        checkbox_label_border_color_hover_dark="#CBD5E0",
        # Panel & Labels
        panel_background_fill="#FFFFFF",
        panel_background_fill_dark="#FFFFFF",
    )

