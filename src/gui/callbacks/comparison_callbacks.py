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
        
        best_correct_model_name = None
        best_correct_conf = -1.0
        best_correct_class = None

        for key, inference_result in result.results.items():
            if inference_result is None or inference_result.top1 is None:
                continue
                
            top1 = inference_result.top1
            conf = top1.confidence_pct
            
            # Track absolute highest confidence (fallback)
            if conf > best_conf:
                best_conf = conf
                best_model_name = key
                best_class = top1.class_name
                
            # Track highest confidence among correct predictions
            if result.ground_truth and inference_result.is_correct:
                if conf > best_correct_conf:
                    best_correct_conf = conf
                    best_correct_model_name = key
                    best_correct_class = top1.class_name

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

        # Resolve which model to show as "Best"
        is_best_correct = False
        if best_correct_model_name:
            best_model_name = best_correct_model_name
            best_conf = best_correct_conf
            best_class = best_correct_class
            is_best_correct = True
        elif not result.ground_truth:
            is_best_correct = True # Assume correct if no ground truth

        # Build Best Model HTML
        if best_model_name:
            if result.ground_truth and not is_best_correct:
                title = f"⚠️ {self._translator.t(TranslationKey.COMP_ALL_INCORRECT, lang=lang)} ⚠️"
                icon = ""
                color = "#FF6B6B"
            else:
                title = self._translator.t(TranslationKey.COMP_BEST_PREDICTION, lang=lang)
                icon = "🏆 "
                color = "#F05A28"

            best_html = f"""
            <div class='best-model-card' style='margin-top: 24px;'>
                <h3 style='margin: 0; color: {color}; font-weight: 700;'>{icon} {title}</h3>
                <p style='margin: 8px 0 0 0; color: #1A202C; font-size: 1.2rem; font-weight: 600;'>{best_model_name}</p>
                <p style='margin: 4px 0 0 0; color: #718096;'>Predicted <strong>{best_class}</strong> with <strong>{best_conf:.1f}%</strong> confidence</p>
            </div>
            """
        else:
            best_html = ""

        # Build Top-5 Predictions Grid HTML
        top5_title = self._translator.t(TranslationKey.RESULTS_TOP5, lang=lang)
        top5_html = f"<div style='margin-top: 32px; border-top: 1px solid #E2E8F0; padding-top: 24px;'>"
        top5_html += f"<h4 style='margin: 0 0 16px 0; color: #1A202C; font-size: 1.1rem; font-weight: 600;'>{top5_title}</h4>"
        top5_html += "<div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px;'>"

        for key, inference_result in result.results.items():
            if inference_result is None or not inference_result.top_predictions:
                continue
                
            short_name = key.split("—")[0].strip() if "—" in key else key
            
            top5_html += f"""
            <div style='background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.02);'>
                <h5 style='margin: 0 0 12px 0; color: #4A5568; font-size: 0.95rem; font-weight: 600; border-bottom: 1px solid #EDF2F7; padding-bottom: 8px;'>{short_name}</h5>
            """
            
            for pred in inference_result.top_predictions[:5]:
                is_correct = result.ground_truth and pred.class_name == result.ground_truth
                text_color = "#00D68F" if is_correct else "#1A202C"
                font_weight = "600" if is_correct else "400"
                
                top5_html += f"""
                <div style='display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 6px;'>
                    <span style='color: {text_color}; font-weight: {font_weight};'>{pred.class_name}</span>
                    <span style='color: #F05A28; font-weight: 600;'>{pred.confidence_pct:.1f}%</span>
                </div>
                """
            top5_html += "</div>"
            
        top5_html += "</div></div>"
        
        # Combine Best HTML and Top-5 HTML
        best_html += top5_html

        return kpi_html, best_html
