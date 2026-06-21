# -*- coding: utf-8 -*-
"""ComparisonChartBuilder — builds matplotlib figures from ComparisonResult DTOs.

Completely decoupled from Gradio, services, or any ML framework.  The only
inputs are domain objects; the only output is a ``matplotlib.figure.Figure``.

This follows the Open/Closed Principle: to add a new chart type (e.g. radar
chart, top-5 grouped bars), subclass ``ComparisonChartBuilder`` and override
``build()`` — no existing code changes.

Usage::

    builder = ComparisonChartBuilder()
    fig = builder.build(comparison_result)
"""

import logging
from typing import List, Optional, Tuple

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from src.domain.models.comparison_result import ComparisonResult
from src.domain.models.inference_result import InferenceResult

matplotlib.use("Agg")  # Non-interactive backend — safe in server environments

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Design tokens — dark theme palette
# ---------------------------------------------------------------------------
_BG_OUTER = "#16213E"
_BG_INNER = "#1a1a2e"
_COLOR_NEUTRAL = "#6C63FF"
_COLOR_CORRECT = "#00D68F"
_COLOR_WRONG = "#FF6B6B"
_COLOR_SPINE = "#444"
_TEXT_COLOR = "white"


class ComparisonChartBuilder:
    """Builds a horizontal bar chart comparing Top-1 predictions across models.

    Responsibilities (SRP):
        - Accept a ``ComparisonResult`` (pure domain object).
        - Produce a ``matplotlib.figure.Figure`` with dark-theme styling.
        - Apply colour coding: green (correct), red (wrong), indigo (unknown).

    Non-responsibilities:
        - Running inference.
        - Gradio component creation.
        - Any I/O.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, result: ComparisonResult) -> Figure:
        """Build and return the comparison chart figure.

        Args:
            result: Aggregated comparison result from ``ComparisonService``.

        Returns:
            A ``Figure`` ready to be embedded in a ``gr.Plot`` component.
        """
        valid = result.successful_results

        if not valid:
            return self._empty_figure()

        model_names, top1_classes, top1_confs = self._extract_data(valid)
        colors = self._compute_colors(top1_classes, result.ground_truth)
        short_names = self._shorten_names(model_names)

        fig, ax = self._create_axes(len(model_names))
        self._draw_bars(ax, short_names, top1_confs, top1_classes, colors)
        self._style_axes(ax, top1_confs, result.ground_truth)

        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_data(
        self, valid: dict
    ) -> Tuple[List[str], List[str], List[float]]:
        model_names = list(valid.keys())
        top1_classes = [
            r.top1.class_name if r.top1 else "N/A"
            for r in valid.values()
        ]
        top1_confs = [
            r.top1.confidence_pct if r.top1 else 0.0
            for r in valid.values()
        ]
        return model_names, top1_classes, top1_confs

    def _compute_colors(
        self,
        top1_classes: List[str],
        ground_truth: Optional[str],
    ) -> List[str]:
        if ground_truth is None:
            return [_COLOR_NEUTRAL] * len(top1_classes)
        return [
            _COLOR_CORRECT if cls == ground_truth else _COLOR_WRONG
            for cls in top1_classes
        ]

    @staticmethod
    def _shorten_names(names: List[str]) -> List[str]:
        """Strip accuracy suffix from display names for axis readability."""
        return [
            name.split("—")[0].strip() if "—" in name else name
            for name in names
        ]

    def _create_axes(self, n_models: int):
        fig, ax = plt.subplots(figsize=(12, max(4, n_models * 0.9)))
        fig.patch.set_facecolor(_BG_OUTER)
        ax.set_facecolor(_BG_INNER)
        return fig, ax

    def _draw_bars(
        self,
        ax,
        short_names: List[str],
        top1_confs: List[float],
        top1_classes: List[str],
        colors: List[str],
    ) -> None:
        y_pos = np.arange(len(short_names))
        bars = ax.barh(
            y_pos,
            top1_confs,
            height=0.6,
            color=colors,
            edgecolor="white",
            linewidth=0.5,
            alpha=0.9,
        )

        for bar, conf, cls in zip(bars, top1_confs, top1_classes):
            ax.text(
                bar.get_width() + 1,
                bar.get_y() + bar.get_height() / 2,
                f"{cls}  ({conf:.1f}%)",
                va="center",
                ha="left",
                fontsize=10,
                fontweight="bold",
                color=_TEXT_COLOR,
            )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(
            short_names, fontsize=11, color=_TEXT_COLOR, fontweight="bold"
        )
        ax.invert_yaxis()

    def _style_axes(
        self,
        ax,
        top1_confs: List[float],
        ground_truth: Optional[str],
    ) -> None:
        ax.set_xlabel(
            "Confidenza Top-1 (%)",
            fontsize=12,
            color=_TEXT_COLOR,
            fontweight="bold",
        )
        max_conf = max(top1_confs) if top1_confs else 100
        ax.set_xlim(0, max_conf * 1.35)
        ax.tick_params(axis="x", colors=_TEXT_COLOR)

        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("bottom", "left"):
            ax.spines[spine].set_color(_COLOR_SPINE)

        ax.grid(axis="x", alpha=0.15, color=_TEXT_COLOR)

        title = "Confronto Top-1 Prediction"
        if ground_truth:
            title += f"  ·  Ground Truth: {ground_truth}"
        ax.set_title(
            title, fontsize=14, fontweight="bold", color=_TEXT_COLOR, pad=15
        )

        if ground_truth:
            handles = [
                mpatches.Patch(color=_COLOR_CORRECT, label="✅ Corretto"),
                mpatches.Patch(color=_COLOR_WRONG, label="❌ Errato"),
            ]
            ax.legend(
                handles=handles,
                loc="lower right",
                fontsize=10,
                facecolor=_BG_INNER,
                edgecolor=_COLOR_SPINE,
                labelcolor=_TEXT_COLOR,
            )

    @staticmethod
    def _empty_figure() -> Figure:
        """Return a placeholder figure when no results are available."""
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(
            0.5,
            0.5,
            "Nessun modello disponibile",
            ha="center",
            va="center",
            fontsize=16,
            color="#999",
        )
        ax.set_facecolor(_BG_INNER)
        fig.patch.set_facecolor(_BG_OUTER)
        ax.axis("off")
        return fig
