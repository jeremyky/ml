# src/nns/core/losses/regression.py
import numpy as np
from .reductions import apply_reduction
from .__init__ import LossOut

def mse(yhat, y, reduction="mean", sample_weight=None, return_grad=False):
    diff = yhat - y
    loss = np.square(diff)
    if return_grad:
        grad = 2 * diff
        if sample_weight is not None:
            grad = grad * sample_weight
        return LossOut(
            value=apply_reduction(loss, reduction, sample_weight),
            grad=grad / np.maximum(1, y.size) if reduction == "mean" else grad
        )
    return LossOut(value=apply_reduction(loss, reduction, sample_weight))
