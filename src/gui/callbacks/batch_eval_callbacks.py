# -*- coding: utf-8 -*-
"""Callback handler for the folder-level batch evaluation tab."""

import logging
from typing import Any, List, Optional, Tuple

import gradio as gr
import pandas as pd

from src.domain.models.batch_result import BatchEvalResult
from src.i18n.keys import TranslationKey
from src.i18n.languages import Language
from src.i18n.translator import Translator
from src.repositories.model_registry import ModelRegistry
from src.services.batch_eval_service import BatchEvalService
from src.services.dataset_service import DatasetService
from src.visualization.batch_chart import BatchChartBuilder

logger = logging.getLogger(__name__)


class BatchEvalCallbackHandler:
    """Bridges Gradio events to BatchEvalService."""

    def __init__(self, batch_service: BatchEvalService, dataset_service: DatasetService, registry: ModelRegistry, chart_builder: BatchChartBuilder, translator: Translator) -> None:
        self._batch = batch_service
        self._dataset = dataset_service
        self._registry = registry
        self._charts = chart_builder
        self._translator = translator

    def update_model_choices(self, show_advanced: bool):
        choices = self._registry.keys(show_advanced=show_advanced)
        return gr.update(choices=choices, value=choices)

    def run_batch(self, class_name: Optional[str], split: str, model_keys: Optional[List[str]], lang_state: str, progress=gr.Progress()) -> Tuple[pd.DataFrame, Any, Any, Any, str, Any, List[BatchEvalResult]]:
        lang = Language(lang_state)
        if not class_name:
            raise gr.Error(self._translator.t(TranslationKey.BATCH_ERR_NO_CLASS, lang=lang))
        if not model_keys:
            raise gr.Error(self._translator.t(TranslationKey.BATCH_ERR_NO_MODELS, lang=lang))
        videos = self._dataset.get_videos(class_name, split=split)
        if not videos:
            raise gr.Error(self._translator.t(TranslationKey.ERR_NO_VIDEOS, lang=lang))

        def update_progress(current: int, total: int, model_key: str, video_name: str) -> None:
            if progress is None or total <= 0:
                return
            label = self._translator.t(TranslationKey.BATCH_PROGRESS_LABEL, lang=lang, current=min(current + 1, total), total=total, model=self._short_name(model_key), video=video_name)
            progress(min(current / total, 1.0), desc=label)

        results = self._batch.run_class_for_models(class_name=class_name, model_keys=model_keys, split=split, progress_callback=update_progress)
        dataframe = self._build_dataframe(results, lang)
        accuracy_fig = self._charts.build_accuracy_bar_chart(results)
        selected_result = results[0] if results else None
        confidence_fig = self._charts.build_confidence_distribution(selected_result)
        confusion_fig = self._charts.build_confusion_bar_chart(selected_result)
        summary = self._build_summary(results, class_name, split, lang)
        choices = [result.model_key for result in results]
        return (dataframe, accuracy_fig, confidence_fig, confusion_fig, summary, gr.update(choices=choices, value=choices[0] if choices else None), results)

    def render_selected_model(self, results: Optional[List[BatchEvalResult]], selected_model: Optional[str]):
        result = self._find_result(results or [], selected_model)
        return (self._charts.build_confidence_distribution(result), self._charts.build_confusion_bar_chart(result))

    def _build_dataframe(self, results: List[BatchEvalResult], lang: Language) -> pd.DataFrame:
        rows = []
        for result in results:
            for video in result.video_results:
                rows.append({
                    self._translator.t(TranslationKey.BATCH_COL_MODEL, lang=lang): self._short_name(result.model_key),
                    self._translator.t(TranslationKey.BATCH_COL_VIDEO, lang=lang): video.video_name,
                    self._translator.t(TranslationKey.BATCH_COL_PREDICTED, lang=lang): video.top1_class,
                    self._translator.t(TranslationKey.BATCH_COL_CONFIDENCE, lang=lang): video.top1_confidence_pct,
                    self._translator.t(TranslationKey.BATCH_COL_TRUE_CONFIDENCE, lang=lang): video.correct_class_confidence_pct,
                    self._translator.t(TranslationKey.BATCH_COL_TOP5, lang=lang): ", ".join(video.top5_classes),
                    self._translator.t(TranslationKey.BATCH_COL_CORRECT, lang=lang): "OK" if video.is_correct else "NO",
                })
        return pd.DataFrame(rows)

    def _build_summary(self, results: List[BatchEvalResult], class_name: str, split: str, lang: Language) -> str:
        if not results:
            return self._translator.t(TranslationKey.BATCH_STATUS_WAITING, lang=lang)
        lines = [self._translator.t(TranslationKey.BATCH_SUMMARY_TITLE, lang=lang, class_name=class_name, split=split)]
        for result in results:
            lines.append(self._translator.t(TranslationKey.BATCH_SUMMARY_MODEL, lang=lang, model=self._short_name(result.model_key), total=result.total_videos, top1=result.top1_accuracy * 100, top5=result.top5_accuracy * 100, skipped=len(result.skipped_videos)))
        return "\n".join(lines)

    @staticmethod
    def _find_result(results: List[BatchEvalResult], selected_model: Optional[str]) -> Optional[BatchEvalResult]:
        for result in results:
            if result.model_key == selected_model:
                return result
        return results[0] if results else None

    @staticmethod
    def _short_name(model_key: str) -> str:
        return model_key.split("—")[0].strip() if "—" in model_key else model_key
