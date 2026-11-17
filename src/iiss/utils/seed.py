"""Global seeding for reproducible runs."""

from __future__ import annotations

import logging
import os
import random

import numpy as np

logger = logging.getLogger(__name__)


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed ``random``, NumPy and PyTorch (CPU and all CUDA devices).

    Args:
        seed: The seed value.
        deterministic: If True, also request deterministic cuDNN / PyTorch algorithms. This is
            slower and some ops raise errors, so it is off by default. Seeding alone does not
            make GPU training bit-exact.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Required by cuBLAS for deterministic matmuls (CUDA >= 10.2).
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)

    logger.info("Seeded everything with seed=%d (deterministic=%s)", seed, deterministic)
