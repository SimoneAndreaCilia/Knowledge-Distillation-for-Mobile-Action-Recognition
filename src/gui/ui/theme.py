# -*- coding: utf-8 -*-
"""Gradio theme definition for the Knowledge Distillation demo.

Isolates theme configuration so that ``build_ui()`` stays short and the
theme can be changed or tested independently.

Usage::

    from src.gui.ui.theme import build_theme
    theme = build_theme()
"""


def build_theme():
    """Build and return the custom Gradio dark theme.

    Returns:
        A configured ``gr.themes.Base`` instance.
    """
    import gradio as gr  # noqa: PLC0415

    return gr.themes.Base(
        primary_hue=gr.themes.colors.indigo,
        secondary_hue=gr.themes.colors.emerald,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
    ).set(
        # Background
        body_background_fill="#0F0F1A",
        body_background_fill_dark="#0F0F1A",
        block_background_fill="#16213E",
        block_background_fill_dark="#16213E",
        # Borders
        block_border_color="rgba(255, 255, 255, 0.08)",
        block_border_color_dark="rgba(255, 255, 255, 0.08)",
        block_border_width="1px",
        block_border_width_dark="1px",
        block_radius="12px",
        # Typography
        block_label_text_color="rgba(255, 255, 255, 0.7)",
        block_label_text_color_dark="rgba(255, 255, 255, 0.7)",
        block_title_text_color="white",
        block_title_text_color_dark="white",
        body_text_color="rgba(255, 255, 255, 0.85)",
        body_text_color_dark="rgba(255, 255, 255, 0.85)",
        body_text_color_subdued="rgba(255, 255, 255, 0.5)",
        body_text_color_subdued_dark="rgba(255, 255, 255, 0.5)",
        # Buttons
        button_primary_background_fill="linear-gradient(135deg, #6C63FF, #00D68F)",
        button_primary_background_fill_dark="linear-gradient(135deg, #6C63FF, #00D68F)",
        button_primary_text_color="white",
        button_primary_text_color_dark="white",
        # Inputs
        border_color_primary="rgba(108, 99, 255, 0.4)",
        border_color_primary_dark="rgba(108, 99, 255, 0.4)",
        input_background_fill="#1a1a2e",
        input_background_fill_dark="#1a1a2e",
        input_border_color="rgba(255, 255, 255, 0.12)",
        input_border_color_dark="rgba(255, 255, 255, 0.12)",
        # Shadows
        shadow_drop="0 4px 20px rgba(0, 0, 0, 0.3)",
        shadow_drop_lg="0 8px 30px rgba(0, 0, 0, 0.4)",
        block_shadow="0 2px 15px rgba(0, 0, 0, 0.25)",
        block_shadow_dark="0 2px 15px rgba(0, 0, 0, 0.25)",
        # Checkboxes
        checkbox_background_color="#1a1a2e",
        checkbox_background_color_dark="#1a1a2e",
        checkbox_border_color="rgba(255, 255, 255, 0.2)",
        checkbox_border_color_dark="rgba(255, 255, 255, 0.2)",
    )
