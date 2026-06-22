# -*- coding: utf-8 -*-
"""DatasetCallbackHandler — handles UI events for the dataset browser section.

Covers:
  - Toggling between upload and dataset browser panels.
  - Updating the video dropdown when a class is selected.
  - Loading a video preview when a video is selected.
  - Updating the model dropdown when the advanced toggle changes.

Usage::

    handler = DatasetCallbackHandler(dataset_service, registry)
    class_dd.change(fn=handler.update_videos, inputs=[class_dd], outputs=[video_dd])
"""

import logging
from typing import Optional

from src.repositories.model_registry import ModelRegistry
from src.services.dataset_service import DatasetService

logger = logging.getLogger(__name__)

_SOURCE_DATASET = "📂 Dataset HMDB-51"


class DatasetCallbackHandler:
    """Handles dataset browser and video source toggle events.

    All methods in this class are designed to be passed directly to
    Gradio event listeners (``fn=handler.method``).

    Args:
        dataset_service: Provides class and video enumeration.
        registry:        Used to rebuild the model dropdown options.
    """

    def __init__(
        self,
        dataset_service: DatasetService,
        registry: ModelRegistry,
    ) -> None:
        self._dataset = dataset_service
        self._registry = registry

    # ------------------------------------------------------------------
    # Gradio-facing methods
    # ------------------------------------------------------------------

    def update_videos(self, class_name: Optional[str]):
        """Return an updated video dropdown for the selected *class_name*.

        Returns:
            A ``gr.update()`` dict for the video ``gr.Dropdown``.
        """
        import gradio as gr  # noqa: PLC0415

        videos = self._dataset.get_videos(class_name or "")
        return gr.update(choices=videos, value=videos[0] if videos else None)

    def get_preview_path(
        self, class_name: Optional[str], video_name: Optional[str]
    ) -> Optional[str]:
        """Return the absolute path string for the video preview player.

        Returns:
            Path string, or None if resolution fails.
        """
        path = self._dataset.resolve_path(class_name or "", video_name or "")
        return str(path) if path else None

    def toggle_video_source(self, source: str):
        """Toggle visibility between upload section and dataset browser.

        Returns:
            A 2-tuple of ``gr.update()`` dicts: ``(upload_visible, dataset_visible)``.
        """
        import gradio as gr  # noqa: PLC0415

        is_dataset = source == "dataset"
        return (
            gr.update(visible=not is_dataset),  # upload section
            gr.update(visible=is_dataset),       # dataset section
        )

    def update_model_dropdown(self, show_advanced: bool):
        """Rebuild the model dropdown choices based on the advanced toggle.

        Returns:
            A ``gr.update()`` dict for the model ``gr.Dropdown``.
        """
        import gradio as gr  # noqa: PLC0415

        choices = self._registry.keys(show_advanced)
        return gr.update(choices=choices, value=choices[0] if choices else None)
