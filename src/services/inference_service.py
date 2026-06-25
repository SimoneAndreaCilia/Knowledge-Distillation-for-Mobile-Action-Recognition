# -*- coding: utf-8 -*-
"""InferenceService — runs inference on a video and returns a typed result.

This is the only place in the codebase that touches torch tensors, logits,
softmax, and topk.  Everything above this layer (callbacks, UI) receives
only ``InferenceResult`` objects — no ML primitives leak through.

Usage::

    service = InferenceService(model_service, class_names, device)
    result = service.run(video_path="/path/to/clip.avi", model_key="Teacher...")
"""

import logging
import time
from pathlib import Path
from typing import List, Optional, Union

import torch
import torch.nn.functional as F

from src.domain.models.inference_result import InferenceResult
from src.domain.models.prediction import Prediction
from src.services.model_service import ModelService

logger = logging.getLogger(__name__)


class InferenceService:
    """Executes a single-model inference pass and returns an ``InferenceResult``.

    Responsibilities (SRP):
        - Pre-process the video tensor.
        - Run a forward pass on the loaded model.
        - Convert raw logits to top-k ``Prediction`` objects.
        - Measure wall-clock elapsed time.

    Non-responsibilities:
        - Loading or caching models.
        - Resolving video file paths (that's ``DatasetService``).
        - Any UI formatting.

    Args:
        model_service: Provides ready-to-use model instances.
        class_names:   Ordered list of action class names matching model output indices.
        device:        Torch device string (``"cuda"`` or ``"cpu"``).
        top_k:         Number of top predictions to return (default: 5).
    """

    def __init__(
        self,
        model_service: ModelService,
        class_names: List[str],
        device: str,
        top_k: int = 5,
    ) -> None:
        self._model_service = model_service
        self._class_names = class_names
        self._device = torch.device(device)
        self._top_k = top_k

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        video_path: Union[str, Path],
        model_key: str,
        ground_truth: Optional[str] = None,
    ) -> InferenceResult:
        """Run inference on *video_path* with the model identified by *model_key*.

        Args:
            video_path:   Path to a video file accepted by ``preprocess_video``.
            model_key:    Registry key of the model to use.
            ground_truth: Optional ground-truth class label for accuracy reporting.

        Returns:
            A fully populated ``InferenceResult`` DTO.

        Raises:
            KeyError:          If ``model_key`` is unknown.
            FileNotFoundError: If the checkpoint does not exist.
            RuntimeError:      If video preprocessing fails.
        """
        # Deferred import: keeps this module importable without a full GPU stack
        from src.evaluation.inference import preprocess_video  # noqa: PLC0415

        model = self._model_service.get_or_load(model_key)

        logger.debug("InferenceService: preprocessing '%s'", video_path)
        clip_tensor = preprocess_video(str(video_path)).to(self._device)

        t0 = time.perf_counter()
        predictions = self._forward(model, clip_tensor)
        elapsed = time.perf_counter() - t0

        logger.info(
            "InferenceService: '%s' → top1='%s' (%.3fs)",
            model_key,
            predictions[0].class_name if predictions else "N/A",
            elapsed,
        )

        return InferenceResult(
            model_key=model_key,
            top_predictions=predictions,
            elapsed_seconds=elapsed,
            ground_truth=ground_truth,
        )

    def run_gradcam(
        self,
        video_path: Union[str, Path],
        model_key: str,
        target_layer: str,
        target_class: Optional[int] = None,
        output_dir: Union[str, Path] = "results/gradcam"
    ) -> str:
        """Run Grad-CAM inference and save the video overlay.
        
        Args:
            video_path:   Path to a video file.
            model_key:    Registry key of the model to use.
            target_layer: The name of the layer to attach the hook.
            target_class: Optional specific class index to analyze. If None, uses top predicted class.
            output_dir:   Directory to save the generated video.
            
        Returns:
            Path to the generated video file.
        """
        import os
        import datetime
        from src.evaluation.inference import preprocess_video
        from src.visualization.grad_cam import generate_gradcam_video
        
        model = self._model_service.get_or_load(model_key)
        
        # Preprocess to get tensor AND raw frames
        clip_tensor, raw_frames = preprocess_video(str(video_path), return_frames=True)
        clip_tensor = clip_tensor.to(self._device)
        
        # Generate output filename
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        video_name = Path(video_path).stem
        safe_model_key = model_key.split(" ")[0].lower() # e.g. "teacher" or "student"
        output_filename = f"{safe_model_key}_{video_name}_{timestamp}.mp4"
        output_path = os.path.join(output_dir, output_filename)
        
        logger.info(
            "InferenceService: Generating Grad-CAM for '%s' using layer '%s'. Output: %s",
            model_key, target_layer, output_path
        )
        
        generate_gradcam_video(
            model=model,
            video_tensor=clip_tensor,
            raw_frames=raw_frames,
            target_layer_name=target_layer,
            output_path=output_path,
            target_class=target_class
        )
        
        return output_path

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _forward(
        self, model: torch.nn.Module, clip: torch.Tensor
    ) -> List[Prediction]:
        """Run the forward pass and convert logits to Prediction objects."""
        with torch.no_grad():
            logits = model(clip)
            probs = F.softmax(logits, dim=1)
            k = min(self._top_k, probs.shape[1])
            top_probs, top_indices = torch.topk(probs, k, dim=1)

        predictions: List[Prediction] = []
        for i in range(k):
            idx = int(top_indices[0][i].item())
            prob = float(top_probs[0][i].item())
            label = (
                self._class_names[idx]
                if idx < len(self._class_names)
                else f"Class {idx}"
            )
            predictions.append(Prediction(class_name=label, confidence=prob))

        return predictions
