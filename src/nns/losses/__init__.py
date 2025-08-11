# src/nns/core/losses/base.py
from dataclasses import dataclass
from typing import Literal, Optional
import numpy as np

Reduction = Literal["none", "mean", "sum"]

@dataclass
class LossOut:
    value: np.ndarray               # raw (unreduced) loss per sample
    grad: Optional[np.ndarray] = None  # gradient w.r.t. predictions or logits


# common interface for loss classes
class Loss:
    def __call__(self, prediction, target):
        raise NotImplementedError
    def backward(self):
        raise NotImplementedError
