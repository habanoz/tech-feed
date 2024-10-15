---
layout: single
title: "Demystifying Neural Network Activations In Pytorch"
categories:
  - workshop

tags:
  - pytorch
  - backpropagation
  - transformer
  - gpt2
  - 
mathjax: true
---

Backpropagation is a crucial part of the deep learning training process. It allows us to compute gradients and update our model parameters. This post is not about explaining how backpropagation works. There are many excellent resources online that explain backprop in depth. For a few recommended materials you mays see [1,2].

Training large neural networks requires a lot of GPU memory and activation memory is a significant contributor to the memory consumption.

In this post I will try to demystify the activation memory in neural networks. It is difficult to find good materials explaining the nature of activation memory. I hope this post will be useful.


* This post assumes a basic understanding of how backpropagation works.


## What is activation memory ?

Consider the following linear transformations which are very common in neural networks:

$$z = Wx$$

$$y = z + b$$

The partial derivatives are:

$$\frac{\partial z}{\partial x} = W$$

$$\frac{\partial z}{\partial W} = x$$

$$\frac{\partial y}{\partial z} = 1$$

$$\frac{\partial y}{\partial b} = 1$$

We need to find derivatives of `x` and `y` wrt. `W`, which gives how sensitive is y to the W and x. By chain rule:


$$\frac{\partial y}{\partial x} = \frac{\partial y}{\partial z} \cdot \frac{\partial z}{\partial x} = \frac{\partial y}{\partial z} \cdot W$$

$$\frac{\partial y}{\partial W} = \frac{\partial y}{\partial z} \cdot \frac{\partial z}{\partial W} = \frac{\partial y}{\partial z} \cdot x$$

Note that we need derivative of `y` wrt. `z` to calculate partial derivatives. In this example they are obvious, but in a deep neural network it is usually not the case. 

Here is a simple multiplication implementation for pytorch:

```python
class Multplication(Function):
    @staticmethod
    def forward(ctx, W, x):
        ctx.save_for_backward(W, x)
        output = W*x
        return output

    @staticmethod
    def backward(ctx, grad_output):
        W, x = ctx.saved_tensors
        grad_W = grad_output * x
        grad_x = grad_output * W
        return grad_W, grad_x
```

Note the `grad_output` parameter. It is the gradient of the output generated in the forward pass. For our example, in math terms, it is:

$$\frac{\partial y}{\partial z}$$

This is the expression that we needed to calculate partial derivatives above. Now lets repeat what we have learnt: In order to calculate partial derivatives of we need the gradient of the output and some variables. Output gradient is provided by pytorch. In multiplication case, variables are operands, namely x and W. For other operations, variables change. Which variables are necessary for gradient calculation depends on the operation. e.g. for derivative of sum operation no variable is needed because derivative of sum is just 1. 

Since we need input variables in backward pass to calculate partial derivatives, we need to save them in forward pass. This is the source of activation memory. 

## Pytorch Tensors

In pytorch everything is a tensor. From memory point of view, network weights, input or activations are all the same. They are tensors saved in the memory.

Input tensors, typically, are not modified. They are not adjusted to improve the network output. Hence input tensors do not require gradients. But weights and all activations do require gradients. For activations, if one of the inputs to the operation requires gradient, then the resulting activation also requires gradients.

An input to an operation can be one of weight, input or activation e.g. result from another operation. 

What happens when a weight is used in an operation? What will happen if the weight tensor needs to be saved for backward pass? Will occupy more memory space? The answer is no! Pytorch will create a new torch but it will point to the same memory location of the original weight tensor, thus no additional memory will be used.

### Torch CUDA Caching Allocator

Pytorch uses a sophisticated CUDA allocator to manage CUDA memory efficiently. The idea is that `cudaMalloc` and `cudaFree` operations takes significant amount of time and introduce synchronization between CPU and GPU. In order avoid `cudaMalloc` and `cudaFree` calls as much as possible, pytorch requests block of memory from CUDA and tries to reuse them during the lifetime of the program.

The allocator rounds up allocation requests to avoid oddly shaped allocations which are likely to cause fragmentation. In default mode, allocations are rounded up to multiple of 512 bytes. 

For more information about the CUDA caching allocator, your may see [3].

## Computation Graph

During forward pass, pytorch creates a computation graph to allow execution of backward pass. For each forward call, a node that includes call to backward function is added to the computation graph. 

This computation graph is core to the Pytorch's backpropagation implementation. Loss value is the root of the graph, network weights are leaves of the graph. Intermediate nodes correspond to backward pass functions that calculates output gradients. 

![Linear Layer - Computation Graph]({{site.baseurl}}/assets/images/torch-activation-memory-linear.png)

## References

1- [CS231n Winter 2016: Lecture 4: Backpropagation, Neural Networks 1](https://www.youtube.com/watch?v=i94OvYb6noo)

2- [The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0)

3- [A guide to PyTorch's CUDA Caching Allocator](https://zdevito.github.io/2022/08/04/cuda-caching-allocator.html)

4- [Scaling Laws Notebook by Karpathy](https://github.com/karpathy/nanoGPT/blob/master/scaling_laws.ipynb)