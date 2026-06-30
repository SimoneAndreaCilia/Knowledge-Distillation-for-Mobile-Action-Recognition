# -*- coding: utf-8 -*-
"""Matplotlib chart builders for folder-level batch evaluation."""

from typing import List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from src.domain.models.batch_result import BatchEvalResult

matplotlib.use("Agg")
_BG = "#FFFFFF"
_GRID = "#E2E8F0"
_TEXT = "#1A202C"
_TOP1 = "#F05A28"
_TOP5 = "#94A3B8"
_CONF = "#F59E0B"
_ERR = "#EA580C"


class BatchChartBuilder:
    """Builds charts for batch evaluation results."""

    def build_accuracy_bar_chart(self, results: List[BatchEvalResult]) -> Figure:
        if not results:
            return self._empty("No batch results available")
        labels = [self._short_name(result.model_key) for result in results]
        top1 = [result.top1_accuracy * 100 for result in results]
        top5 = [result.top5_accuracy * 100 for result in results]
        x = np.arange(len(labels))
        width = 0.36
        fig, ax = plt.subplots(figsize=(12, 4.8))
        fig.patch.set_facecolor(_BG)
        ax.set_facecolor(_BG)
        bars_top1 = ax.bar(x - width / 2, top1, width, label="Top-1", color=_TOP1)
        bars_top5 = ax.bar(x + width / 2, top5, width, label="Top-5", color=_TOP5)
        for bars in (bars_top1, bars_top5):
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9, color=_TEXT)
        ax.set_ylim(0, 105)
        ax.set_ylabel("Accuracy (%)", color=_TEXT, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=12, ha="right", color=_TEXT)
        ax.set_title("Top-1 / Top-5 Accuracy", color=_TEXT, fontweight="bold")
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.5, color=_GRID, linestyle="--")
        self._clean_spines(ax)
        plt.tight_layout()
        return fig

    def build_confidence_distribution(self, result: Optional[BatchEvalResult]) -> Figure:
        if result is None or not result.video_results:
            return self._empty("No confidence data available")
        confidences = [video.correct_class_confidence_pct for video in result.video_results if video.correct_class_confidence_pct is not None]
        if not confidences:
            return self._empty("No confidence data available")
        fig, ax = plt.subplots(figsize=(12, 4.5))
        fig.patch.set_facecolor(_BG)
        ax.set_facecolor(_BG)
        ax.hist(confidences, bins=10, range=(0, 100), color=_CONF, edgecolor="white")
        ax.set_title(f"Correct-Class Confidence - {self._short_name(result.model_key)}", color=_TEXT, fontweight="bold")
        ax.set_xlabel("Confidence for true class (%)", color=_TEXT)
        ax.set_ylabel("Videos", color=_TEXT)
        ax.grid(axis="y", alpha=0.5, color=_GRID, linestyle="--")
        self._clean_spines(ax)
        plt.tight_layout()
        return fig

    def build_confusion_bar_chart(self, result: Optional[BatchEvalResult]) -> Figure:
        if result is None or not result.confusion_counts:
            return self._empty("No wrong predictions")
        items = sorted(result.confusion_counts.items(), key=lambda item: item[1], reverse=True)[:12]
        labels = [item[0] for item in items]
        counts = [item[1] for item in items]
        y_pos = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(12, max(4, len(labels) * 0.45)))
        fig.patch.set_facecolor(_BG)
        ax.set_facecolor(_BG)
        bars = ax.barh(y_pos, counts, color=_ERR)
        for bar, count in zip(bars, counts):
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2, str(count), va="center", color=_TEXT)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, color=_TEXT)
        ax.invert_yaxis()
        ax.set_xlabel("Wrong predictions", color=_TEXT)
        ax.set_title(f"Confusion Breakdown - {self._short_name(result.model_key)}", color=_TEXT, fontweight="bold")
        ax.grid(axis="x", alpha=0.5, color=_GRID, linestyle="--")
        self._clean_spines(ax)
        plt.tight_layout()
        return fig

    @staticmethod
    def _short_name(model_key: str) -> str:
        return model_key.split("—")[0].strip() if "—" in model_key else model_key

    @staticmethod
    def _clean_spines(ax) -> None:
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(_GRID)
        ax.tick_params(colors=_TEXT)

    @staticmethod
    def _empty(message: str) -> Figure:
        fig, ax = plt.subplots(figsize=(10, 3.5))
        fig.patch.set_facecolor(_BG)
        ax.set_facecolor(_BG)
        ax.text(0.5, 0.5, message, ha="center", va="center", color="#718096")
        ax.axis("off")
        return fig
