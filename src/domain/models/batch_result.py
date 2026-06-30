# -*- coding: utf-8 -*-
"""Batch evaluation DTOs for class-level inference results."""

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class VideoResult:
    """Result for one video evaluated by one model."""

    video_name: str
    top1_class: str
    top1_confidence: float
    correct_class_confidence: Optional[float]
    is_correct: bool
    is_top5_correct: bool
    top5_classes: List[str]

    @property
    def top1_confidence_pct(self) -> float:
        return round(self.top1_confidence * 100, 2)

    @property
    def correct_class_confidence_pct(self) -> Optional[float]:
        if self.correct_class_confidence is None:
            return None
        return round(self.correct_class_confidence * 100, 2)


@dataclass
class BatchEvalResult:
    """Aggregated batch results for one class and one model."""

    class_name: str
    model_key: str
    video_results: List[VideoResult] = field(default_factory=list)
    skipped_videos: Dict[str, str] = field(default_factory=dict)

    @property
    def total_videos(self) -> int:
        return len(self.video_results)

    @property
    def top1_accuracy(self) -> float:
        if not self.video_results:
            return 0.0
        correct = sum(1 for result in self.video_results if result.is_correct)
        return correct / len(self.video_results)

    @property
    def top5_accuracy(self) -> float:
        if not self.video_results:
            return 0.0
        correct = sum(1 for result in self.video_results if result.is_top5_correct)
        return correct / len(self.video_results)

    @property
    def confusion_counts(self) -> Dict[str, int]:
        wrong_predictions = [
            result.top1_class
            for result in self.video_results
            if not result.is_correct
        ]
        return dict(Counter(wrong_predictions))
