from __future__ import annotations
import numpy as np
from .utils import ensure_grad, unbroadcast, topo_sort

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
        self._prev = tuple(_prev) # tuple over set to maintain order
        self._backward = lambda: None # default no-op _backward

    def zero_grad(self):
        if self.grad is not None:
            self.grad[...] = 0
    
    """
    backward

    triggers backpropagation starting from the current Tnesor

    calling some_tensor.backward() means
    this is the final output (like a loss), and I want to compute how every other tensor affects it (compute gradients)
    """
    def backward(self, grad=None):
        # if gradients are not needed, stop
        if not self.requires_grad:
            return
        # set the default gradient if none is provided
        # gradient of the final output w.r.t. itself is 1, dL/dL = 1

        # Accumulate or set the gradient
        # if self.grad exists from previous backward passes, then just accumulate it
        # else, set it to the incoming radient
        # this is important when the same tensor is being used in multiple places, i.e. x + x
        if grad is None:
            grad = np.ones_like(self.data)
        ensure_grad(self)
        self.grad += grad
        # perform reverse-mode autodiff: 
        # topologically sort all tensors that contributed to this one
        # run _backward() on each tensor in the correct order (outputs --> inputs)
        # each _backward() uses the gradient of the output to compute and store the gradient for its own inputs
        for t in topo_sort(self):
            t._backward()

    """
    add

    defines the + operator
    z = x + y
    enables adding two tensors while
    - storing the forward result
    - defining how to compute gradients for each input
    - hooking it into the computation graph via _prev and _backward

    Backward Closure
    z = x + y
    dz/dx = 1, dy/dz = 1
    dL/dx = dL/dz * dz/dx = out.grad * 1 = out.grad
    dL/dy = dL/dy * dz/dy = out.grad * 1 = out.grad
    """
    def __add__(self, other):
        # if other is not a Tensor (i.e. just a number), convert it to one. Allows support for Tensor + float or Tensor + np.array
        other = other if isinstance(other, Tensor) else Tensor(other)
        requires = self.requires_grad or other.requires_grad
        # Forward Pass: perform elementwise addition using NumPy, wrap in new Tnesor, requires_grad is True if either input needs gradients
        out = Tensor(self.data + other.data, requires_grad = requires, _op="add", _prev=(self,other))    
        def _backward():
            if self.requires_grad:
                ensure_grad(self)
                self.grad += Tensor._unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                ensure_grad(self)
                other.grad += Tensor._unbroadcast(out.grad, other.data.shape)
        # hook into computation graph
        out._backward = _backward
        return out
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __neg__(self):
        return self * -1.0
    
    def __sub__(self, other):
        return self + (- (other if isinstance(other, Tensor) else Tensor(other)))
    
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        requires = self.requires_grad or other.requires_grad
        out = Tensor(self.data * other.data, requires_grad=requires, _op="mul", _prev=(self,other))

        def _backward():
            if self.requires_grad:
                ensure_grad(self)
                self.grad += unbroadcast(out.grad * other.data, self.data.shape)
            if other.requires_grad:
                ensure_grad(other)
                other.grad += unbroadcast(out.grad * self.data, other.data.shape)
        out._backward = _backward
        return out
    
    def pow(self, p: "float"):
        out = Tensor(self.data ** p, requires_grad=self.requires_grad, _op=f"pow{p}", _prev=(self,))

        def _backward():
            if self.requires_grad:
                ensure_grad(self)
                self.grad += (p * (self.data ** (p-1))) * out.grad
        out._backward = _backward
        return out

    """
    matmul

    special method for operator overloading @ --> __matmul__()
    this allows us to do: z = a @ b
    when a and b are Tensor objects, it calls: a.__matmul__(b)

    Backward CLosure
    Suppose z = x @ y
    - x = shape (m x n)
    - y = shape (n x p)
    - z = shape (m x p)
    During backpropagation, we get ∂L/∂z = out.grad

    Using matrix calculus:
    Gradient w.r.t. x is ∂L/∂x = ∂L/∂z dot y transpose = out.grad @ other.data.T
    Gradient w.r.t. y is ∂L/∂y = x transpose dot ∂L/∂z = self.data.T @ out.grad
    Compute and accumulate these gradients via self.grad += ... and other.grad += ...
    """
    def __matmul__(self, other):
        # Forward Pass
        # Perform matrix multiplication using NumPy, wrap the result in a new Tensor called out, requires_grad = True if either input needs gradients
        requires = self.requires_grad or other.requires_grad
        out = Tensor(self.data @ other.data, requires_grad=requires, _op="matmul", _prev=(self, other))        # This function tells autograd how to compute the gradients of the inputs given the gradient of the output (out.grad)
        def _backward():
            if self.requires_grad:
                ensure_grad(self)
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                ensure_grad(self)
                other.grad += self.data.T @ out.grad
        # Save the backward() closure so it can be called via .backward()
        out._backward = _backward
        # Track dependencies for topo_sort() so autograd knows how out was computed
        # Return resulting Tensor with forward result and attached gradient logic
        return out
    

    def mean(self, axis=None, keepdims=False):
        out = Tensor(self.data.mean(axis=axis, keepdims=keepdims),
                    requires_grad=self.requires_grad, _op="mean", _prev=(self,))
        # number of elements reduced
        if axis is None:
            count = self.data.size
        else:
            axes = (axis,) if isinstance(axis, int) else tuple(axis)
            count = np.prod([self.data.shape[a] for a in axes])
        def _backward():
            if self.requires_grad:
                ensure_grad(self)
                g = out.grad / count
                if axis is not None and not keepdims:
                    # re-expand grad to original shape
                    axes = (axis,) if isinstance(axis, int) else tuple(axis)
                    for ax in sorted(axes):
                        g = np.expand_dims(g, ax)
                self.grad += np.ones_like(self.data) * g
        out._backward = _backward
        return out
    
    """
    relu

    Rectified Linear Unit (ReLU) is a simple activation function
    ReLU(x) = max(0, x)
    if x > 0 return x
    if x <= 0, return 0

    ReLU adds non-linearity to the model so it can learn complex patterns
    - very simple and fast
    - doesn't saturate for large positive values (unlike sigmoid/tanh)
    - helps reduce vanishing gradients
    """
    def relu(self):
        # compute np.maximum(x, 0) for the forward pass, wrap result into new Tensor, if input needs gradients the output should too
        out = Tensor(np.maximum(self.data, 0), requires_grad=self.requires_grad)

        # define how to compute gradients for the backward pass
        # ReLU has derivative 1 if x > 0, and 0 elsewise
        # during backprop, multiply the incoming gradient by x > 0
        def _backward():
            if self.requires_grad:
                # out.grad is the gradient flowing into ReLU
                # multiply by self.data > 0 to zero out gradients where input was <= 0
                # accumulate the result into self.grad
                self.grad += (self.data > 0) * out.grad
        # stores how to backprop through this operation
        out._backward = _backward
        # tells autograd that this tensor came from self
        out._prev = {self}
        return out
    
    
    
        
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
    
    @staticmethod
    def topo_sort(root: "Tensor"):
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
    """

    """
    some operations will crash if .grad is None
    safer and memory-efficient to lazy-init grads when needed
    utilize _ensure_grad helper function
    
    @staticmethod
    def _ensure_grad(t: "Tensor"):
        if t.grad is None:
            t.grad = np.zeros_like(t.data)
    """


    """
    relying on NumPy broadcasting in forward, we must unbroadcast the gradient in backward so shapes match inputs
    
    @staticmethod
    def _unbroadcast(grad, target_shape):
        # reduce grad to target_shape by summing along broadcasted axes
        while grad.ndim > len(target_shape):
            grad = grad.sum(axis=0)
        for i, (g, t) in enumerate(zip(grad.shape, target_shape)):
            if t == 1 and g != 1:
                grad = grad.sum(axis=i, keepdims=True)
            return grad

            """
        

# Sanity Test
x = Tensor([[1., -2., 3.]], requires_grad=True)
W = Tensor([[2.], [0.5], [-1.]], requires_grad=True)
y = (x @ W).relu() + 1.0
y.backward()  # seed = 1

# Expected:
# dy/d(x@W) = 1 where x@W > 0 else 0
# mask depends on value of x@W
print("y:", y.data)
print("x.grad:", x.grad)
print("W.grad:", W.grad)


