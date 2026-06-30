# -*- coding: utf-8 -*-
"""BatchEvalService - class-level evaluation over HMDB-51 folders."""

import logging
from typing import Callable, List, Optional

from src.domain.models.batch_result import BatchEvalResult, VideoResult
from src.services.dataset_service import DatasetService
from src.services.inference_service import InferenceService

logger = logging.getLogger(__name__)
ProgressCallback = Callable[[int, int, str, str], None]


class BatchEvalService:
    """Runs one or more models over every selected video in an action class."""

    def __init__(self, dataset_service: DatasetService, inference_service: InferenceService) -> None:
        self._dataset = dataset_service
        self._inference = inference_service

    def run_class(
        self,
        class_name: str,
        model_key: str,
        split: str = "all",
        progress_callback: Optional[ProgressCallback] = None,
        completed_offset: int = 0,
        total_steps: Optional[int] = None,
    ) -> BatchEvalResult:
        """Run inference on every video in a class folder."""
        videos = self._dataset.get_videos(class_name, split=split)
        total = len(videos)
        denominator = total_steps if total_steps is not None else total
        result = BatchEvalResult(class_name=class_name, model_key=model_key)

        for index, video_name in enumerate(videos, start=1):
            if progress_callback is not None:
                progress_callback(completed_offset + index - 1, denominator, model_key, video_name)

            path = self._dataset.resolve_path(class_name, video_name)
            if path is None:
                result.skipped_videos[video_name] = "Video path not found"
                continue

            try:
                inference = self._inference.run(path, model_key=model_key, ground_truth=class_name)
            except Exception as exc:
                logger.warning("BatchEvalService: skipped '%s' for '%s': %s", video_name, model_key, exc, exc_info=True)
                result.skipped_videos[video_name] = str(exc)
                continue

            top1 = inference.top1
            if top1 is None:
                result.skipped_videos[video_name] = "No prediction returned"
                continue

            top5_classes = [prediction.class_name for prediction in inference.top_predictions[:5]]
            result.video_results.append(
                VideoResult(
                    video_name=video_name,
                    top1_class=top1.class_name,
                    top1_confidence=top1.confidence,
                    correct_class_confidence=inference.correct_class_confidence,
                    is_correct=top1.class_name == class_name,
                    is_top5_correct=class_name in top5_classes,
                    top5_classes=top5_classes,
                )
            )

        if progress_callback is not None:
            progress_callback(completed_offset + total, denominator, model_key, "")
        return result

    def run_class_for_models(
        self,
        class_name: str,
        model_keys: List[str],
        split: str = "all",
        progress_callback: Optional[ProgressCallback] = None,
    ) -> List[BatchEvalResult]:
        """Run class evaluation for each selected model."""
        videos = self._dataset.get_videos(class_name, split=split)
        total_steps = len(videos) * len(model_keys)
        completed = 0
        results: List[BatchEvalResult] = []

        for model_key in model_keys:
            results.append(
                self.run_class(
                    class_name=class_name,
                    model_key=model_key,
                    split=split,
                    progress_callback=progress_callback,
                    completed_offset=completed,
                    total_steps=total_steps,
                )
            )
            completed += len(videos)

        if progress_callback is not None:
            progress_callback(total_steps, total_steps, "", "")
        return results
