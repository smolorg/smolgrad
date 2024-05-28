![smolgrad logo](./images/logo.png)

A small auto-grad engine (inspired from Karpathy's micrograd and PyTorch) using Apple's MLX and Numpy. That's why the name: **smolgrad**

it will help y'all understand the core concepts behind automatic differentiation and backpropagation. I mean, AI is literally powered by these things.

### what is autograd?

auto-grad, short for automatic differentiation, is a technique used in machine learning to efficiently compute gradients of complex functions. In the context of neural networks, autograd enables the computation of gradients with respect to the model's parameters, which is crucial for training the network using optimization algorithms like gradient descent.

it works by constructing a computational graph that represents the flow of data through the network. Each node in the graph represents a mathematical operation, and the edges represent the flow of data between these operations. By tracking the operations and their dependencies, autograd can automatically compute the gradients using the chain rule of calculus.

an algorithm used in conjuction with autograd is **Backpropagation** to train neural networks. It is the process of propagating the gradients backward through the computational graph, from the output layer to the input layer, in order to update the model's parameters. This is the stuff which makes ML/DL models "learn".


### how does this work

so the most important file here is [smolgrad/core/engine.py](./smolgrad/core/engine.py) because it defines a `Tensor` class.

the `Tensor` class implements automatic differentiation, which allows the computation of gradients of the output tensor with respect to its input tensors. This is achieved by building a computational graph that keeps track of the operations performed on tensors.

when a tensor is created, it can be marked as requiring gradients by using the `set_requires_grad()` method. This indicates that the tensor's gradients should be computed during the backward pass.

the `backward()` method is responsible for performing backpropagation and computing the gradients. It first topologically sorts the computational graph to ensure that gradients are computed in the correct order i.e. gradients of children tensors should be calculated first and so on... then, it initializes the gradient of the output tensor to 1 and iteratively computes the gradients of each tensor in the graph using the chain rule.

**Backward Functions for Operations**
for each operation performed on tensors, a corresponding backward function is defined to compute the gradients during the backward pass. These backward functions are stored in the `grad_fn` attribute of the resulting tensor.

a few examples are:

- **Unary Operations**
    - `half()`: This operation converts the tensor's data and gradients to half precision (float16). The backward function simply copies the gradients from the output tensor to the input tensor.

    - `exp()`: This operation computes the element-wise exponential of the tensor. The backward function multiplies the gradients of the output tensor with the original tensor data to compute the gradients of the input tensor.

- **Binary Operations**

    - `__add__()`: This operation performs element-wise addition of two tensors, taking broadcasting into account. The backward function differs based on whether the input tensors have the same shape or different shapes:

        - Same shape: The gradients of the output tensor are directly propagated to the input tensors.
        - Different shapes: The gradients are summed along the broadcasted axes before being propagated to the input tensors.


by defining these backward functions for each operation, the Tensor class enables automatic differentiation. During the backward pass, the gradients are propagated through the computational graph, and the `grad` attribute of each tensor is updated with the computed gradients.


> note: this code is a simplified version, and a complete implementation of the Tensor class would require additional operations, error handling, and optimizations. But, the goal here is to understand and learn.



⚠️ *This is still a work in progress*