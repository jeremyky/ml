mini-PyTorch



src/nns/core/
- tensor.py: Tensorclass + autograd ops
- ops.py: extra ops: exp, log, sum, pow, etc.
- utils.py: detach(), item(), no_grad, etc.
- graph.py: topo_sort, backward utilities
- test_tensor.py: Tests for all core features

modularity:
tensor.py: core class w/ methods that act on a single tensor (self)

ops.py: mathematical ops between tensors (loss functions, elementwise ops, etc.)

utils.py: generic tools and helpers (graph traversal, no_grad context, etc.)



todo

""" 
TODO: more ops w/ correct backward
Neg / Sub / RAdd / RMul
Mul (elemntwise) w/ broadcasting-safe backward
Sum / Mean (axis, keepdims)
Reshape / Transpose (no-op grads w/ shape restore)
Exp / Log / Pow

TODO: utilities
detach(): break graph
item(): get Python scalar from 0-D tensor
Context manager no_grad() to stop building graph temporarily
Free graph after backward unless retain_graph = True to avoid memory leaks

TODO: Backward
add a retain_graph feature: after the topo loop, if not retained, set each nodes ._prev=() and ._backward=lambda:None to free the graph
check if the seed grad shape matches self.data.shape or broadcastable --> assert early

TODO: Minimal loss + optimizer for training
MSE loss: (pred-target).pow(2).mean()
SGD: Step over a list of tesnors w/ requires_grad = True

TODO: Tests
ReLU on/off
Add w/ broadcasting
Sum / Mean shapes
Matmul gradient check
Mul rule (x * y).sum().backward() → x.grad == y.data, y.grad == x.data.
"""