# -*- coding: utf-8 -*-
"""InferenceResult DTO — the complete output of a single-model inference run."""

from dataclasses import dataclass, field
from typing import List, Optional

from .prediction import Prediction


@dataclass(frozen=True)
class InferenceResult:
    """Encapsulates the full output of one inference pass.

    The UI consumes this object and never touches torch tensors, logits,
    or softmax probabilities directly.

    Attributes:
        model_key:          Display-name key of the model that produced this result.
        top_predictions:    Ordered list of top-k Prediction objects (highest first).
        elapsed_seconds:    Wall-clock time taken for the inference call.
        ground_truth:       Optional ground-truth class name (from dataset browser).
    """

    model_key: str
    top_predictions: List[Prediction]
    elapsed_seconds: float
    ground_truth: Optional[str] = None
    correct_class_confidence: Optional[float] = None
    """Softmax probability for the ground-truth class (None if unknown)."""

    @property
    def top1(self) -> Optional[Prediction]:
        """Return the highest-confidence prediction, or None if empty."""
        return self.top_predictions[0] if self.top_predictions else None

    @property
    def is_correct(self) -> Optional[bool]:
        """Return True/False if ground truth is known, else None."""
        if self.ground_truth is None or self.top1 is None:
            return None
        return self.top1.class_name == self.ground_truth

    def as_label_dict(self) -> dict:
        """Return a {class_name: confidence} dict compatible with gr.Label."""
        return {p.class_name: p.confidence for p in self.top_predictions}
