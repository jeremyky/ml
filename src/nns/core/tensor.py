from __future__ import annotations
# labels to describe expected variables, function parameters, return values
import numpy as np
# fast computation

"""
Tiny Tensor
A class that wraps a number or array and adds three things
- data: the actual numeric values it holds
- grad: the gradient w.r.t. some loss
- creator: the operation that created this tensor (from an add, mul, etc.) so we can trace back for backward pass

Autograd: automatic differentiation
This is the system that automatically computes gradients to be used for backpropagation

i.e.
computing z = x * y + y
then calling z.backward(), this framework will walk backward through the computation graph, computing:
dz/dx
dz/dy

We need these to train a neural network because we want to compute gradients of a loss function w.r.t. each parameter then update via an algorithm like gradient descent

Creator Operation:
whenever we do something like z = x + y, we:
1. create a new tensor z
2. which was created by an operation +
3. so z.creator = AddOp(x, y)
This creator operation stores the parents of z (the inputs) and knows how to compute the gradients during backprop

This creates a computation graph of operations

Backward Closure:
A backward closure is just a function (or lambda) that we attach to a tensor or operation to define
How do I compute the gradients of the inputs, given the gradient of the output

For example
def backward(dz):
    dx = dz * 1
    dy = dz * 1
    return dx, dy

in the Tensor object, we would store something like
self._backward = backward
And during z.backward(), we would call this recursively


"""
class Tensor:
    def __init__(self, data, requires_grad=False, _op=None, _prev=()):
        self.data = np.asarray(data, dtype=float)
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self.requires_grad = requires_grad
        self._op = _op
        self._prev = set(_prev)

    def zero_grad(self):
        if self.grad is not None:
            self.grad[...] = 0
    
    def backward(self, grad=None):
        if not self.requires_grad:
            return
        if grad is None:
            np.ones_like(self.data)
        self.grad = self.grad + grad if self.grad is not None else grad
        for t in topo_sort(self):
            t._backward()

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)
    
        def _backward():
            if self.requires_grad:
                self.grad += out.grad
            if other.requires_grad:
                other.grad += out.grad
            out._backward = _backward
            out._prev = {self, other}
            return out

    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad)
        def _backward():
            if self.requires_grad:
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                other.grad += self.data.T @ out.grad
        out._backward = _backward
        out._prev = {self, other}
        return out

    def relu(self):
        out = Tensor(np.maximum(self.data, 0), requires_grad=self.requires_grad)
        def _backward():
            if self.requires_grad:
                self.grad += (self.data > 0) * out.grad
        out._backward = _backward
        out._prev = {self}
        return out
    
    def _backward(self):
        pass
        
    """
    topo_sort

    we need this because when we call .backward() on a Tensor, we want to:
    - start at the final output (i.e. loss)
    - propagate gradients backward to all the input tensors
    - in the correct order, respecting dependencies

    this requires a topological sort of our computation graph, so we can visit each node after all of its inputs (parents) have been processed

    DFS Topological Order
    A topological sort is an ordering of nodes in a directed acyclic graph (DAG) s.t. each node comes after all the nodes that point to it
    In our case:
    - Each Tensor is a node
    - _prev are edges from input tensors to the current tensor
    - we need to compute gradients starting from outputs and flowing back to inputs
    - do this using depth-first saerch (DFS) to build this order
    """
    def topo_sort(root: Tensor):
        # DFS topological order
        visited, order = set(), []
        # visted tracks al lthe nodes we have seen, order stores the sorted nodes (post-order DFS)
        def build(v): # if a node hasn't been visited, recursively visit its parents, after all parents handled, append v to order
            if v not in visited:
                visited.add(v)
                for p in v._prev: # recursively visit all parents
                    build(p)
                order.append(v) # add after all inputs visited
        build(root)
        return reversed(order) # reverse to get output to inputs to be used for backward()