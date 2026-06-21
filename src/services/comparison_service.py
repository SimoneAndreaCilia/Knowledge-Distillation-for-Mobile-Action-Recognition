# -*- coding: utf-8 -*-
"""ComparisonService — orchestrates multi-model inference on a single video.

This service is the only place that coordinates running ``InferenceService``
across multiple model keys and aggregating the results into a single
``ComparisonResult`` DTO.

Usage::

    service = ComparisonService(inference_service, registry)
    result  = service.run_all(
        video_path=Path("/data/hmdb51/brush_hair/Amy_Adams_03.avi"),
        ground_truth="brush_hair",
        show_advanced=False,
    )
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Union

from src.domain.models.comparison_result import ComparisonResult
from src.domain.models.inference_result import InferenceResult
from src.repositories.model_registry import ModelRegistry
from src.services.inference_service import InferenceService

logger = logging.getLogger(__name__)


class ComparisonService:
    """Runs all registered models on the same video and collects results.

    Responsibilities (SRP):
        - Iterate over registry keys (filtered by ``show_advanced``).
        - Skip models whose checkpoints are missing.
        - Catch per-model errors without aborting the entire comparison.
        - Assemble a ``ComparisonResult``.

    Non-responsibilities:
        - Running the actual forward pass (delegated to ``InferenceService``).
        - Visualising results (delegated to ``ComparisonChartBuilder``).
        - Any UI logic.

    Args:
        inference_service: Executes single-model inference.
        registry:          Provides the list of available model configs.
    """

    def __init__(
        self,
        inference_service: InferenceService,
        registry: ModelRegistry,
    ) -> None:
        self._inference = inference_service
        self._registry = registry

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_all(
        self,
        video_path: Union[str, Path],
        ground_truth: Optional[str] = None,
        show_advanced: bool = False,
    ) -> ComparisonResult:
        """Run inference with all eligible models and return aggregated results.

        Models whose checkpoint files are missing are silently skipped (error
        recorded in ``ComparisonResult.errors``).

        Args:
            video_path:    Path to the video file.
            ground_truth:  Optional ground-truth class name.
            show_advanced: Include advanced/ablation models if True.

        Returns:
            A ``ComparisonResult`` containing per-model results and errors.
        """
        configs = self._registry.all_configs(show_advanced)
        results: Dict[str, Optional[InferenceResult]] = {}
        errors: Dict[str, str] = {}

        logger.info(
            "ComparisonService: running %d model(s) on '%s'",
            len(configs),
            video_path,
        )

        for key, config in configs.items():
            if not config.checkpoint.is_file():
                msg = f"Checkpoint non trovato: {config.checkpoint}"
                logger.warning("ComparisonService: skip '%s' — %s", key, msg)
                results[key] = None
                errors[key] = msg
                continue

            try:
                result = self._inference.run(
                    video_path=video_path,
                    model_key=key,
                    ground_truth=ground_truth,
                )
                results[key] = result
            except Exception as exc:
                logger.error(
                    "ComparisonService: error for '%s': %s", key, exc, exc_info=True
                )
                results[key] = None
                errors[key] = str(exc)

        return ComparisonResult(
            results=results,
            errors=errors,
            ground_truth=ground_truth,
        )
