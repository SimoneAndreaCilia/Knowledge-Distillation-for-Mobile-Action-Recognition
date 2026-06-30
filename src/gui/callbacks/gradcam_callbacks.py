# -*- coding: utf-8 -*-
import logging
from typing import Optional, Tuple

from src.i18n.keys import TranslationKey
from src.i18n.languages import Language
from src.i18n.translator import Translator
from src.services.inference_service import InferenceService
from src.services.dataset_service import DatasetService

logger = logging.getLogger(__name__)


class GradCamCallbackHandler:
    def __init__(
        self,
        inference_service: InferenceService,
        dataset_service: DatasetService,
        translator: Translator,
        classes: list,
    ) -> None:
        self._inference = inference_service
        self._dataset = dataset_service
        self._translator = translator
        self._classes = classes

    def generate(
        self,
        uploaded_video: Optional[str],
        dataset_class: Optional[str],
        dataset_video: Optional[str],
        video_source: str,
        model1_key: str,
        target_layer1: str,
        model2_key: str,
        target_layer2: str,
        class_mode: str,
        manual_class: str,
        lang_state: str,
    ) -> Tuple[str, str]:
        import gradio as gr
        lang = Language(lang_state)

        def t(key: TranslationKey, **kwargs) -> str:
            return self._translator.t(key, lang=lang, **kwargs)

        # 1. Resolve video
        if video_source == "dataset":
            video_path = self._dataset.resolve_path(dataset_class or "", dataset_video or "")
            if video_path is None:
                raise gr.Error(t(TranslationKey.GRADCAM_ERR_SELECT_VIDEO))
            video_path = str(video_path)
            gt_class = dataset_class
        else:
            if not uploaded_video:
                raise gr.Error(t(TranslationKey.GRADCAM_ERR_UPLOAD_VIDEO))
            video_path = uploaded_video
            gt_class = None

        # 2. Resolve Target Class
        target_class_idx = None
        if class_mode == "ground_truth":
            if not gt_class:
                raise gr.Error(t(TranslationKey.GRADCAM_ERR_NO_GROUND_TRUTH))
            try:
                target_class_idx = self._classes.index(gt_class)
            except ValueError:
                target_class_idx = None
        elif class_mode == "manual":
            if not manual_class:
                raise gr.Error(t(TranslationKey.GRADCAM_ERR_SELECT_MANUAL_CLASS))
            try:
                target_class_idx = self._classes.index(manual_class)
            except ValueError:
                target_class_idx = None
        elif class_mode == "predicted":
            target_class_idx = None  # Handled dynamically inside generation

        # 3. Generate Grad-CAM for Model 1
        logger.info("Generating Grad-CAM for Model 1: %s", model1_key)
        try:
            video1_path = self._inference.run_gradcam(
                video_path=video_path,
                model_key=model1_key,
                target_layer=target_layer1,
                target_class=target_class_idx,
            )
        except Exception as e:
            logger.exception("Grad-CAM error for Model 1: %s", e)
            raise gr.Error(t(TranslationKey.GRADCAM_ERR_MODEL1, error=e))

        # 4. Generate Grad-CAM for Model 2
        logger.info("Generating Grad-CAM for Model 2: %s", model2_key)
        try:
            video2_path = self._inference.run_gradcam(
                video_path=video_path,
                model_key=model2_key,
                target_layer=target_layer2,
                target_class=target_class_idx,
            )
        except Exception as e:
            logger.exception("Grad-CAM error for Model 2: %s", e)
            raise gr.Error(t(TranslationKey.GRADCAM_ERR_MODEL2, error=e))

        return video1_path, video2_path
