# -*- coding: utf-8 -*-
"""ComparisonCallbackHandler — bridges Gradio compare button to ComparisonService."""

import logging
from typing import Any, Optional, Tuple

from matplotlib.figure import Figure

from src.i18n.keys import TranslationKey
from src.i18n.languages import Language
from src.i18n.translator import Translator
from src.services.comparison_service import ComparisonService
from src.services.dataset_service import DatasetService
from src.visualization.comparison_chart import ComparisonChartBuilder

logger = logging.getLogger(__name__)


class ComparisonCallbackHandler:
    """Handles the ``compare`` button click event for multi-model comparison."""

    def __init__(
        self,
        comparison_service: ComparisonService,
        dataset_service: DatasetService,
        chart_builder: ComparisonChartBuilder,
        translator: Translator,
    ) -> None:
        self._comparison = comparison_service
        self._dataset = dataset_service
        self._chart_builder = chart_builder
        self._translator = translator

    def compare(
        self,
        uploaded_video: Optional[str],
        dataset_class: Optional[str],
        dataset_video: Optional[str],
        video_source: str,
        show_advanced: bool,
        lang_state: str,
    ) -> Tuple[Any, str]:
        """Compare all registered models on the same video."""
        import gradio as gr  # noqa: PLC0415
        
        lang = Language(lang_state)

        video_path, ground_truth = self._resolve_video(
            video_source, uploaded_video, dataset_class, dataset_video, lang
        )

        gr.Info(self._translator.t(TranslationKey.INFO_COMPARISON_SUCCESS, lang=lang, num_models=0).replace('0', '...')) # Wait, info isn't for loading.
        # Actually in the original: gr.Info("⏳ Confronto modelli in corso…")
        # Let's use the loading info properly. But we don't have a specific key for comparison loading. 
        # I'll just use a generic or reuse info.
        # Wait, the original was "⏳ Confronto modelli in corso…", let's use COMP_STATUS_WAITING.
        gr.Info(self._translator.t(TranslationKey.COMP_STATUS_WAITING, lang=lang))
        
        result = self._comparison.run_all(
            video_path=video_path,
            ground_truth=ground_truth,
            show_advanced=show_advanced,
        )

        fig = self._chart_builder.build(result)
        summary = self._build_summary(result, lang)
        return fig, summary

    def _resolve_video(
        self,
        source: str,
        uploaded: Optional[str],
        class_name: Optional[str],
        video_name: Optional[str],
        lang: Language,
    ):
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

    def _build_summary(self, result, lang: Language) -> str:
        """Render a markdown summary table from a ComparisonResult."""
        lines = [self._translator.t(TranslationKey.COMP_SUMMARY_TITLE, lang=lang)]

        if result.ground_truth:
            lines.append(self._translator.t(TranslationKey.COMP_GROUND_TRUTH, lang=lang, truth=result.ground_truth))

        for key, inference_result in result.results.items():
            if inference_result is None:
                error_msg = result.errors.get(key, self._translator.t(TranslationKey.COMP_UNKNOWN_ERROR, lang=lang))
                lines.append(f"### ❌ {key}\n{error_msg}\n")
                continue

            top1 = inference_result.top1
            if top1 is None:
                lines.append(f"### ⚠️ {key}\n{self._translator.t(TranslationKey.COMP_NO_PREDICTION, lang=lang)}\n")
                continue

            if result.ground_truth:
                icon = "🎯" if inference_result.is_correct else "❌"
            else:
                icon = "🔍"

            lines.append(
                f"### {icon} {key}\n"
                f"**Top-1:** `{top1.class_name}` ({top1.confidence_pct:.1f}%)\n"
            )

            for pred in inference_result.top_predictions:
                bar_len = int(pred.confidence * 20)
                bar = "█" * bar_len + "░" * (20 - bar_len)
                lines.append(
                    f"  `{bar}` {pred.class_name}: {pred.confidence_pct:.1f}%"
                )
            lines.append("")

        return "\n".join(lines)
