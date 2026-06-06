# -*- coding: utf-8 -*-
"""
Model Profiling Utilities
==========================

Measures deployment-critical metrics for comparing Teacher vs. Student:
  • **Parameter count** (total and trainable)
  • **Model size** on disk (MB)
  • **Inference latency** (ms, with GPU warmup and averaging)
  • **FLOPs** estimation (optional, requires ``thop`` or ``fvcore``)

Usage::

    from src.evaluation.profile import profile_model
    from src.models import build_model

    model = build_model("student", num_classes=51)
    result = profile_model(model, device=torch.device("cuda"))
    print(result)
    # {'param_count': 2_235_123, 'trainable_params': 2_235_123,
    #  'model_size_mb': 8.53, 'latency_ms': 12.45, 'flops': ...}
"""

import logging
import tempfile
import time
from typing import Dict, Any, Tuple, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters.

    Args:
        model: The model to profile.

    Returns:
        Tuple of (total_params, trainable_params).
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def compute_model_size_mb(model: nn.Module) -> float:
    """Estimate model size in megabytes (parameter memory only).

    Accounts for the actual dtype of each parameter (float32 = 4 bytes,
    float16 = 2 bytes, etc.).

    Args:
        model: The model to measure.

    Returns:
        Model size in megabytes.
    """
    total_bytes = sum(
        p.numel() * p.element_size() for p in model.parameters()
    )
    return total_bytes / (1024 ** 2)


def compute_checkpoint_size_mb(model: nn.Module) -> float:
    """Measure the actual on-disk size of a saved checkpoint.

    Saves the model's state_dict to a temporary file and measures its
    file size. This gives a more accurate estimate than parameter-only
    size because it includes serialization overhead and buffers.

    Args:
        model: The model to measure.

    Returns:
        Checkpoint file size in megabytes.
    """
    import os
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
            tmp_path = tmp.name
        # Save outside the 'with' block so the file handle is released
        torch.save(model.state_dict(), tmp_path)
        size_bytes = os.path.getsize(tmp_path)
    finally:
        if tmp_path is not None and os.path.exists(tmp_path):
            os.remove(tmp_path)
    return size_bytes / (1024 ** 2)


@torch.no_grad()
def measure_latency(
    model: nn.Module,
    device: torch.device,
    input_shape: Tuple[int, ...] = (1, 3, 16, 112, 112),
    num_warmup: int = 10,
    num_runs: int = 50,
) -> float:
    """Measure average single-sample inference latency.

    Performs GPU warmup runs, then times ``num_runs`` forward passes
    and returns the average latency in milliseconds.

    Args:
        model: The model to benchmark (set to eval mode internally).
        device: Device for inference (should match model's device).
        input_shape: Input tensor shape ``(B, C, T, H, W)``.
        num_warmup: Number of warmup runs (not timed).
        num_runs: Number of timed runs for averaging.

    Returns:
        Average inference latency in milliseconds.
    """
    model.eval()
    model.to(device)
    dummy_input = torch.randn(input_shape, device=device)

    # Warmup: let GPU JIT-compile kernels and reach steady state
    for _ in range(num_warmup):
        _ = model(dummy_input)

    # Synchronize before timing (critical for CUDA)
    if device.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()
    for _ in range(num_runs):
        _ = model(dummy_input)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time

    avg_latency_ms = (elapsed / num_runs) * 1000.0
    return avg_latency_ms


def estimate_flops(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 3, 16, 112, 112),
) -> Optional[int]:
    """Estimate FLOPs using the ``thop`` library (if installed).

    Args:
        model: The model to profile.
        input_shape: Input tensor shape.

    Returns:
        Total FLOPs as an integer, or ``None`` if ``thop`` is not
        installed.
    """
    try:
        from thop import profile as thop_profile
        dummy = torch.randn(input_shape)
        model.eval()
        model.cpu()
        flops, _ = thop_profile(model, inputs=(dummy,), verbose=False)
        return int(flops)
    except ImportError:
        logger.debug(
            "thop not installed — FLOPs estimation skipped. "
            "Install with: pip install thop"
        )
        return None
    except Exception as e:
        logger.debug(f"FLOPs estimation failed: {e}")
        return None


def profile_model(
    model: nn.Module,
    device: Optional[torch.device] = None,
    input_shape: Tuple[int, ...] = (1, 3, 16, 112, 112),
    measure_latency_flag: bool = True,
) -> Dict[str, Any]:
    """Run full model profiling and return a summary dict.

    Combines all profiling utilities into a single call that produces
    a comprehensive deployment profile.

    Args:
        model: The model to profile.
        device: Device for latency measurement. If ``None``, uses CPU.
        input_shape: Input tensor shape for latency and FLOPs.
        measure_latency_flag: If ``True``, run latency benchmarking.

    Returns:
        Dict containing:
          - ``param_count``: Total parameter count.
          - ``trainable_params``: Trainable parameter count.
          - ``model_size_mb``: Parameter memory in MB.
          - ``checkpoint_size_mb``: On-disk checkpoint size in MB.
          - ``latency_ms``: Average inference latency (if measured).
          - ``flops``: Estimated FLOPs (if ``thop`` is available).
    """
    if device is None:
        device = torch.device("cpu")

    total_params, trainable_params = count_parameters(model)
    model_size = compute_model_size_mb(model)
    ckpt_size = compute_checkpoint_size_mb(model)

    result: Dict[str, Any] = {
        "param_count": total_params,
        "trainable_params": trainable_params,
        "model_size_mb": model_size,
        "checkpoint_size_mb": ckpt_size,
    }

    if measure_latency_flag:
        latency = measure_latency(model, device, input_shape)
        result["latency_ms"] = latency

    flops = estimate_flops(model, input_shape)
    if flops is not None:
        result["flops"] = flops
        result["gflops"] = flops / 1e9

    # Log summary
    logger.info(
        f"Model Profile: "
        f"{total_params:,} params | "
        f"{model_size:.2f} MB (memory) | "
        f"{ckpt_size:.2f} MB (checkpoint)"
        + (f" | {result.get('latency_ms', 0):.2f} ms latency"
           if 'latency_ms' in result else "")
        + (f" | {result.get('gflops', 0):.2f} GFLOPs"
           if 'gflops' in result else "")
    )

    return result
