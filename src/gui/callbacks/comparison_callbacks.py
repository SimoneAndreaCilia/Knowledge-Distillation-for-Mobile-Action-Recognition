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

        result = self._comparison.run_all(
            video_path=video_path,
            ground_truth=ground_truth,
            show_advanced=show_advanced,
        )

        fig = self._chart_builder.build(result)
        kpi_html, best_html = self._build_summary(result, lang)
        return fig, kpi_html, best_html

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

    def _build_summary(self, result, lang: Language) -> Tuple[str, str]:
        """Render HTML for KPI Summary Cards and the Best Model highlight."""
        if not result.successful_results:
            return "<div style='color: #718096;'>No results available</div>", ""

        # Build KPI Cards HTML
        kpi_html = "<div style='display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; justify-content: space-between;'>"
        best_model_name = None
        best_conf = -1.0
        best_class = None

        for key, inference_result in result.results.items():
            if inference_result is None or inference_result.top1 is None:
                continue
                
            top1 = inference_result.top1
            conf = top1.confidence_pct
            
            if conf > best_conf:
                best_conf = conf
                best_model_name = key
                best_class = top1.class_name

            # Small short name
            short_name = key.split("—")[0].strip() if "—" in key else key
            
            color = "#1A202C"
            if result.ground_truth:
                color = "#00D68F" if inference_result.is_correct else "#FF6B6B"

            kpi_html += f"""
            <div class='kpi-card' style='flex: 1; min-width: 150px;'>
                <div class='kpi-title'>{short_name}</div>
                <div class='kpi-value' style='color: {color};'>{conf:.1f}%</div>
                <div style='font-size: 0.8rem; color: #718096;'>{top1.class_name}</div>
            </div>
            """
        kpi_html += "</div>"

        # Build Best Model HTML
        if best_model_name:
            best_html = f"""
            <div class='best-model-card' style='margin-top: 24px;'>
                <h3 style='margin: 0; color: #F05A28; font-weight: 700;'>🏆 Best Prediction</h3>
                <p style='margin: 8px 0 0 0; color: #1A202C; font-size: 1.2rem; font-weight: 600;'>{best_model_name}</p>
                <p style='margin: 4px 0 0 0; color: #718096;'>Predicted <strong>{best_class}</strong> with <strong>{best_conf:.1f}%</strong> confidence</p>
            </div>
            """
        else:
            best_html = ""

        return kpi_html, best_html
