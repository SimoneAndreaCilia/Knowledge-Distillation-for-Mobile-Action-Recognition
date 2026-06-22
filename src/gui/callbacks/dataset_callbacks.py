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
from src.services.video_converter import VideoConverter

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
        video_converter: VideoConverter,
    ) -> None:
        self._dataset = dataset_service
        self._registry = registry
        self._converter = video_converter

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
        if path is None:
            return None
        return self._converter.ensure_web_playable(str(path))

    def update_videos_and_preview(self, class_name: Optional[str]):
        """Return an updated video dropdown AND the preview path for the first video.

        Gradio does not propagate programmatic widget changes as new events, so
        updating the ``dataset_video`` dropdown from ``update_videos`` never
        triggers the ``dataset_video.change`` event that feeds ``video_preview``.
        This method solves that by resolving both in one shot.

        Returns:
            A 2-tuple:
            - ``gr.update`` dict for the video ``gr.Dropdown``.
            - Path string (or ``None``) for the ``gr.Video`` preview player.
        """
        import gradio as gr  # noqa: PLC0415

        videos = self._dataset.get_videos(class_name or "")
        first_video = videos[0] if videos else None
        preview_path = self._dataset.resolve_path(class_name or "", first_video or "")
        preview_str = (
            self._converter.ensure_web_playable(str(preview_path))
            if preview_path
            else None
        )
        return (
            gr.update(choices=videos, value=first_video),
            preview_str,
        )

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
