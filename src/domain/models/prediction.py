# -*- coding: utf-8 -*-
"""Prediction DTO — represents a single class prediction with its confidence."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Prediction:
    """A single class prediction produced by a model.

    Attributes:
        class_name: Human-readable label of the predicted action class.
        confidence: Raw probability in [0, 1] as returned by softmax.
    """

    class_name: str
    confidence: float

    @property
    def confidence_pct(self) -> float:
        """Return confidence as a percentage in [0, 100]."""
        return round(self.confidence * 100, 2)

    def __str__(self) -> str:
        return f"{self.class_name}: {self.confidence_pct:.1f}%"
