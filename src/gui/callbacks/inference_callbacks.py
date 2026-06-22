# -*- coding: utf-8 -*-
"""InferenceCallbackHandler — bridges Gradio events to InferenceService."""

import logging
from typing import Optional, Tuple

from src.i18n.keys import TranslationKey
from src.i18n.languages import Language
from src.i18n.translator import Translator
from src.repositories.model_registry import ModelRegistry
from src.services.dataset_service import DatasetService
from src.services.inference_service import InferenceService

logger = logging.getLogger(__name__)


class InferenceCallbackHandler:
    """Handles the ``classify`` button click event for single-model inference."""

    def __init__(
        self,
        inference_service: InferenceService,
        dataset_service: DatasetService,
        registry: ModelRegistry,
        translator: Translator,
    ) -> None:
        self._inference = inference_service
        self._dataset = dataset_service
        self._registry = registry
        self._translator = translator

    def classify(
        self,
        model_key: Optional[str],
        uploaded_video: Optional[str],
        dataset_class: Optional[str],
        dataset_video: Optional[str],
        video_source: str,
        lang_state: str,
    ) -> Tuple[dict, str, str]:
        """Classify a video with the selected model."""
        import gradio as gr  # noqa: PLC0415
        
        lang = Language(lang_state)

        # ---- Input validation ----
        if not model_key:
            raise gr.Error(self._translator.t("model.select_model", lang=lang))

        video_path, ground_truth = self._resolve_video(
            video_source, uploaded_video, dataset_class, dataset_video, lang
        )

        # ---- Delegate to service ----
        gr.Info(self._translator.t(TranslationKey.STATUS_LOADING, lang=lang, model_key=model_key))
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
            config.format_info(translator=self._translator, lang=lang, elapsed=result.elapsed_seconds)
            if config
            else self._translator.t(TranslationKey.INFO_TIME, lang=lang, time=result.elapsed_seconds)
        )

        status = self._build_status(result, lang)
        return result.as_label_dict(), model_info, status

    def _resolve_video(
        self,
        source: str,
        uploaded: Optional[str],
        class_name: Optional[str],
        video_name: Optional[str],
        lang: Language,
    ):
        """Return (video_path_str, ground_truth) from the active video source."""
        import gradio as gr  # noqa: PLC0415

        if source == "dataset":
            path = self._dataset.resolve_path(class_name or "", video_name or "")
            if path is None:
                raise gr.Error(self._translator.t(TranslationKey.ERR_NO_PREVIEW, lang=lang))
            return str(path), class_name
        else:
            if not uploaded:
                raise gr.Error(self._translator.t(TranslationKey.ERR_UPLOAD_VIDEO, lang=lang))
            return uploaded, None

    def _build_status(self, result, lang: Language) -> str:
        """Build the status markdown string from an InferenceResult."""
        top1 = result.top1
        if top1 is None:
            return self._translator.t(TranslationKey.STATUS_NO_PRED, lang=lang)

        status = self._translator.t(
            TranslationKey.STATUS_PRED, 
            lang=lang, 
            class_name=top1.class_name, 
            confidence=top1.confidence_pct
        )

        if result.ground_truth:
            if result.is_correct:
                status += self._translator.t(TranslationKey.STATUS_CORRECT, lang=lang, truth=result.ground_truth)
            else:
                status += self._translator.t(TranslationKey.STATUS_INCORRECT, lang=lang, truth=result.ground_truth)

        return status
