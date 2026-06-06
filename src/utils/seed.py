# -*- coding: utf-8 -*-
"""
Reproducibility Utilities
=========================

Sets random seeds across all sources of non-determinism in a PyTorch pipeline:
  • Python's built-in `random` module
  • NumPy's random number generator
  • PyTorch CPU and all CUDA devices
  • CuDNN backend (deterministic vs. benchmark mode)
  • Python hash seed (for dict/set ordering)

Usage::

    from src.utils import set_seed
    set_seed(42, deterministic=True)
"""

import os
import random
import logging
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set random seed for full reproducibility across all libraries.

    This function should be called at the very beginning of any training or
    evaluation script, **before** creating any tensors or data loaders.

    Args:
        seed: The random seed value. Default is 42 (the answer to everything).
        deterministic: If ``True``, forces CuDNN to use deterministic algorithms.
            This may incur a ~10-15% performance penalty but guarantees
            bit-exact reproducibility across runs. Set to ``False`` during
            hyper-parameter search for faster iteration.

    Note:
        Even with ``deterministic=True``, certain PyTorch operations (e.g.,
        ``scatter_add``, ``index_put``) may still exhibit non-determinism on
        CUDA. Use ``torch.use_deterministic_algorithms(True)`` for strictest
        mode, but beware of unsupported-operation errors.
    """
    # ------------------------------------------------------------------
    # 1. Python built-in random
    # ------------------------------------------------------------------
    random.seed(seed)

    # ------------------------------------------------------------------
    # 2. Numpy random
    # ------------------------------------------------------------------
    np.random.seed(seed)

    # ------------------------------------------------------------------
    # 3. PyTorch CPU + CUDA
    # ------------------------------------------------------------------
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Multi-GPU safety

    # ------------------------------------------------------------------
    # 4. Python hash seed (affects dict/set iteration order)
    # ------------------------------------------------------------------
    os.environ["PYTHONHASHSEED"] = str(seed)

    # ------------------------------------------------------------------
    # 5. CuDNN backend configuration
    # ------------------------------------------------------------------
    if deterministic:
        # Deterministic mode: slower but reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info(
            f"Seed set to {seed} (deterministic mode ON — CuDNN benchmark OFF)"
        )
    else:
        # Benchmark mode: CuDNN auto-tuner selects fastest algorithms
        # for the given input shapes (non-deterministic across runs)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        logger.info(
            f"Seed set to {seed} (deterministic mode OFF — CuDNN benchmark ON)"
        )
