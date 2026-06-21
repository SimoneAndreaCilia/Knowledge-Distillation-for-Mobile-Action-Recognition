# -*- coding: utf-8 -*-
"""ComparisonCallbackHandler — bridges Gradio compare button to ComparisonService.

Contains only presentation logic: resolving inputs, delegating to the
comparison service, and formatting the result as markdown for display.

Usage::

    handler = ComparisonCallbackHandler(comparison_service, dataset_service, chart_builder)
    btn.click(fn=handler.compare, inputs=[...], outputs=[...])
"""

import logging
from typing import Any, Optional, Tuple

from matplotlib.figure import Figure

from src.services.comparison_service import ComparisonService
from src.services.dataset_service import DatasetService
from src.visualization.comparison_chart import ComparisonChartBuilder

logger = logging.getLogger(__name__)

_SOURCE_DATASET = "📂 Dataset HMDB-51"


class ComparisonCallbackHandler:
    """Handles the ``compare`` button click event for multi-model comparison.

    Responsibilities (SRP):
        - Resolve video path from UI state.
        - Call ``ComparisonService.run_all()``.
        - Build the matplotlib figure via ``ComparisonChartBuilder``.
        - Render the summary markdown table.

    Non-responsibilities:
        - Running inference (delegated to ``ComparisonService``).
        - Chart rendering details (delegated to ``ComparisonChartBuilder``).

    Args:
        comparison_service: Runs all models and aggregates results.
        dataset_service:    Resolves dataset video paths.
        chart_builder:      Converts ``ComparisonResult`` to a matplotlib figure.
    """

    def __init__(
        self,
        comparison_service: ComparisonService,
        dataset_service: DatasetService,
        chart_builder: ComparisonChartBuilder,
    ) -> None:
        self._comparison = comparison_service
        self._dataset = dataset_service
        self._chart_builder = chart_builder

    # ------------------------------------------------------------------
    # Gradio-facing method
    # ------------------------------------------------------------------

    def compare(
        self,
        uploaded_video: Optional[str],
        dataset_class: Optional[str],
        dataset_video: Optional[str],
        video_source: str,
        show_advanced: bool,
    ) -> Tuple[Any, str]:
        """Compare all registered models on the same video.

        Returns:
            A 2-tuple: ``(matplotlib_figure, summary_markdown)``

        Raises:
            gr.Error: For invalid user inputs.
        """
        import gradio as gr  # noqa: PLC0415

        video_path, ground_truth = self._resolve_video(
            video_source, uploaded_video, dataset_class, dataset_video
        )

        gr.Info("⏳ Confronto modelli in corso…")
        result = self._comparison.run_all(
            video_path=video_path,
            ground_truth=ground_truth,
            show_advanced=show_advanced,
        )

        fig = self._chart_builder.build(result)
        summary = self._build_summary(result)
        return fig, summary

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
        import gradio as gr  # noqa: PLC0415

        if source == _SOURCE_DATASET:
            path = self._dataset.resolve_path(class_name or "", video_name or "")
            if path is None:
                raise gr.Error("Seleziona una classe e un video dal dataset.")
            return str(path), class_name
        else:
            if not uploaded:
                raise gr.Error("Carica un file video prima di confrontare.")
            return uploaded, None

    @staticmethod
    def _build_summary(result) -> str:
        """Render a markdown summary table from a ComparisonResult."""
        lines = ["## 📊 Risultati Confronto\n"]

        if result.ground_truth:
            lines.append(f"**Ground Truth:** `{result.ground_truth}`\n")

        for key, inference_result in result.results.items():
            if inference_result is None:
                error_msg = result.errors.get(key, "Errore sconosciuto")
                lines.append(f"### ❌ {key}\n{error_msg}\n")
                continue

            top1 = inference_result.top1
            if top1 is None:
                lines.append(f"### ⚠️ {key}\nNessuna predizione.\n")
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
