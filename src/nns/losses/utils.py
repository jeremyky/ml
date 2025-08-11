"""
Numerically stable utilites for loss computations
"""

from __future__ import annotations
# enable postponed evaluation of type annotations
# allows for forward references (using class name before definition)
# prevents types from being evaluated at function definition time
# makes annotations faster and lighter in runtime (treated as plain strings)
from typing import Optional, Tuple
# part of pythons static typing system, do nothing at runtime but help type checkers
# Optional[X] equivalent to Union[X, None]
# Means the value can be of type X or None
# Tuple[X, Y] means a tuple where the first item is type X, the second is type Y
# Tuple[float, float] can be (12.3, 45.6)
# if we want a Tuple of any length, use Tuple[int, ...] # any length tuple of ints
import numpy as np

Array = np.ndarray
# Array = np.ndarray, type alias, giving the name Array to np.ndarray for readability

# Core numerically stable ops

# typed Python function definition
# x: Array; first argument x must be of type Array
# optional argument named axis w/ default value -1
# boolean is another optional argument
# return type will be Array
def logsumexp(x: Array, axis: int = -1, keepdims: bool = False) --> Array:
    # Stable log-sum-exp
    # logsumexp(x) = log(sum(exp(x))) computed by factoring out max(x)

    x = np.asarray(x)
    m = np.max(x, axis=axis, keepdims=True)
    # subtract max for stability, handle -inf by zeroing exp where needed
    y = np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True)) + m
    return y if keepdims else np.squeeze(y, axis=axis)

def stable_log_softmax(logits: Array, axis: int = -1) -> Array:
    # Stable log-softmax(logits)
    # log_softmax(x) = x - logsumexp(x)
    logits = np.asarray(logits)
    lse = logsumexp(logits, axis=axis, keepdims=True)
    return logits - lse

def stable_softmax(logits: Array, axis: int = -1) -> Array:
    # Stable softmax using log-softmax, returns probabilities with rows (or along axis) summing to ~1
    return np.exp(stable_log_softmax(logits, axis=axis))

def stable_sigmoid(logits: Array) -> Array:
    # Stable sigmoid from logits, uses piecewise formulation to avoid overflow when x << 0 or x >> 0
    x = np.asarray(logits)
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[neg])
    out[neg] = ex / (1.0 + ex)
    return out.astype(x.dtype, copy=False)

def log_sigmoid(logits: Array) -> Array:
    # Stable log(sigmoid(x))
    x = np.asarray(logits)
    # log(sigmoid(x)) = -softplus(-x)
    # softplus(z) = log(1 + exp(z))
    return -np.log1p(np.exp(-x))