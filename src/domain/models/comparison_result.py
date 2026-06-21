# -*- coding: utf-8 -*-
"""ComparisonResult DTO — aggregated output of a multi-model comparison run."""

from dataclasses import dataclass, field
from typing import Dict, Optional

from .inference_result import InferenceResult


@dataclass(frozen=True)
class ComparisonResult:
    """Encapsulates the results of running all models on the same video.

    Attributes:
        results:      Mapping from model_key to InferenceResult.
                      A None value indicates the checkpoint was missing or
                      an error occurred for that model.
        errors:       Mapping from model_key to error message string.
        ground_truth: Optional ground-truth class label.
    """

    results: Dict[str, Optional[InferenceResult]]
    errors: Dict[str, str]
    ground_truth: Optional[str] = None

    @property
    def successful_results(self) -> Dict[str, InferenceResult]:
        """Return only results that completed without error."""
        return {k: v for k, v in self.results.items() if v is not None}

    @property
    def has_any_result(self) -> bool:
        return len(self.successful_results) > 0
