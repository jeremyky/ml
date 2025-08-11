import numpy as np
from typing import Tuple, Union

def ensure_grad(t) -> None:
    """Initializes gradient storage for the tensor if it doesn't exist."""
    if t.grad is None:
        t.grad = np.zeros_like(t.data)

def unbroadcast(grad: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    """
    Reverses broadcasting by summing along broadcasted dimensions.
    """
    while grad.ndim > len(target_shape):
        grad = grad.sum(axis=0)
    for i, (g, t) in enumerate(zip(grad.shape, target_shape)):
        if t == 1 and g != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad

def topo_sort(root) -> reversed:
    """
    Performs a topological sort of the computation graph rooted at `root`,
    returning a list of Tensors in the order they should be backpropagated.
    """
    visited = set()
    order = []

    def build(v):
        if v not in visited:
            visited.add(v)
            for p in v._prev:
                build(p)
            order.append(v)

    build(root)
    return reversed(order)
