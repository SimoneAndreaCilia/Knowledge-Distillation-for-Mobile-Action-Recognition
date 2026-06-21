# -*- coding: utf-8 -*-
"""InferenceCallbackHandler — bridges Gradio events to InferenceService.

This handler contains **zero business logic**.  Its only job is to:
  1. Validate UI inputs and raise ``gr.Error`` for bad state.
  2. Delegate to the service layer.
  3. Format the service output for Gradio components.

The formatting code here (building markdown strings, icon selection) is
*presentation logic*, not *business logic* — it belongs in this layer.

Usage::

    handler = InferenceCallbackHandler(inference_service, dataset_service, registry)
    # Wire in build_ui():
    btn.click(fn=handler.classify, inputs=[...], outputs=[...])
"""

import logging
from typing import Optional, Tuple

from src.repositories.model_registry import ModelRegistry
from src.services.dataset_service import DatasetService
from src.services.inference_service import InferenceService

logger = logging.getLogger(__name__)

_SOURCE_DATASET = "📂 Dataset HMDB-51"


class InferenceCallbackHandler:
    """Handles the ``classify`` button click event for single-model inference.

    Responsibilities (SRP):
        - Resolve video path from UI state (upload vs. dataset selection).
        - Call ``InferenceService.run()``.
        - Format the ``InferenceResult`` into the 3-tuple expected by Gradio:
          ``(label_dict, model_info_str, status_markdown)``.

    Non-responsibilities:
        - Running inference (delegated to ``InferenceService``).
        - Loading models (delegated to ``ModelService`` via ``InferenceService``).

    Args:
        inference_service: Executes the ML inference pass.
        dataset_service:   Resolves dataset video paths.
        registry:          Provides ``ModelMetadata`` for info display.
    """

    def __init__(
        self,
        inference_service: InferenceService,
        dataset_service: DatasetService,
        registry: ModelRegistry,
    ) -> None:
        self._inference = inference_service
        self._dataset = dataset_service
        self._registry = registry

    # ------------------------------------------------------------------
    # Gradio-facing method
    # ------------------------------------------------------------------

    def classify(
        self,
        model_key: Optional[str],
        uploaded_video: Optional[str],
        dataset_class: Optional[str],
        dataset_video: Optional[str],
        video_source: str,
    ) -> Tuple[dict, str, str]:
        """Classify a video with the selected model.

        This method is passed directly to ``gr.Button.click(fn=...)``.

        Returns:
            A 3-tuple: ``(label_dict, model_info_text, status_markdown)``
              - ``label_dict``       → fed to ``gr.Label``
              - ``model_info_text``  → fed to ``gr.Textbox``
              - ``status_markdown``  → fed to ``gr.Markdown``

        Raises:
            gr.Error: For invalid user inputs (no model, no video).
        """
        import gradio as gr  # noqa: PLC0415 — lazy import keeps module testable

        # ---- Input validation ----
        if not model_key:
            raise gr.Error("Seleziona un modello prima di classificare.")

        video_path, ground_truth = self._resolve_video(
            video_source, uploaded_video, dataset_class, dataset_video
        )

        # ---- Delegate to service ----
        gr.Info(f"⏳ Caricamento modello: {model_key}…")
        try:
            result = self._inference.run(
                video_path=video_path,
                model_key=model_key,
                ground_truth=ground_truth,
            )
        except FileNotFoundError as exc:
            raise gr.Error(str(exc)) from exc

        # ---- Format output ----
        config = self._registry.find(model_key)
        model_info = (
            config.format_info(elapsed=result.elapsed_seconds)
            if config
            else f"⏱️ Tempo inferenza: {result.elapsed_seconds:.2f}s"
        )

        status = self._build_status(result)
        return result.as_label_dict(), model_info, status

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_video(
        self,
        source: str,
        uploaded: Optional[str],
        class_name: Optional[str],
        video_name: Optional[str],
    ):
        """Return (video_path_str, ground_truth) from the active video source."""
        import gradio as gr  # noqa: PLC0415

        if source == _SOURCE_DATASET:
            path = self._dataset.resolve_path(class_name or "", video_name or "")
            if path is None:
                raise gr.Error("Seleziona una classe e un video dal dataset.")
            return str(path), class_name
        else:
            if not uploaded:
                raise gr.Error("Carica un file video prima di classificare.")
            return uploaded, None

    @staticmethod
    def _build_status(result) -> str:
        """Build the status markdown string from an InferenceResult."""
        top1 = result.top1
        if top1 is None:
            return "❌ Nessuna predizione disponibile."

        status = f"✅ Predizione: **{top1.class_name}** ({top1.confidence_pct:.1f}%)"

        if result.ground_truth:
            if result.is_correct:
                status += f"  ·  🎯 Corretto! (ground truth: {result.ground_truth})"
            else:
                status += f"  ·  ❌ Errato (ground truth: {result.ground_truth})"

        return status
