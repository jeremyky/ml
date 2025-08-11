# src/nns/core/losses/reductions.py
import numpy as np
from typing import Literal

Reduction = Literal["none", "mean", "sum"]

def apply_reduction(x, reduction: Reduction = "mean", sample_weight=None):
    if sample_weight is not None:
        x = x * sample_weight
    if reduction == "none":
        return x
    if reduction == "sum":
        return np.sum(x, dtype=np.float64)
    return np.sum(x, dtype=np.float64) / max(1, x.size)
