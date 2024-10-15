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

Here is the corresponding computation graph for a linear transformation defined above. Green node represent the root (output tensor), blue nodes represent leaf nodes (weight tensors), orange nodes shows tensors saved for backward pass and gray nodes represents backward pass functions.

Note that `x` is not part of the graph since it does not require gradients.

![Linear Layer - Computation Graph]({{site.baseurl}}/assets/images/torch-activation-memory-linear.png)

In the graph, you can see that multiplication operation saves `w` tensor for backward pass. Conversely, `x` is NOT saved, because it is not part of the graph. Also note that, Add operation does not save any tensors, because its derivative is 1 and does not depend on inputs.

You should also notice the AccumulateGrad nodes. These nodes are connected to the leaf nodes and used for accumulating gradients during backward pass.

In pytorch, there is a special layer for multiplication followed by an addition: `nn.Linear`. We can plugin `nn.Linear` layer to achieve them same output with less code.

![NN Linear Layer - Computation Graph]({{site.baseurl}}/assets/images/torch-activation-memory-nn-linear.png)

Note that instead of two functions, `nn.Linear` introduces single function: AddmmBakward0. Similarly, AddmmBakward0 caches only the weight tensor (mat1). Dark green node indicates that the output tensor (light green tensor) is a view of the output of the Addmm operation. Rest of the graph looks similar.  


## Automatic Mixed Precision Training (AMP)

Another interesting aspect to see at computation graph would be see how Automatic Mixed Precision Training (AMP) looks. 

Neural network weights are stored as 4 byte floats. Float32 is called full precision. Modern GPUs performs matrix multiplication much faster at half precision, float16. However, float16 is not enough for some operations. Using solely float16 significantly degrades training performance. AMP is strikes a balance between full precision and half precision. In AMP, some operations use full precision while others use half precision.

Here is how the computation graph in mixed mode looks like. Note `ToCopyBackward0` functions. They indicate that tensors are down scaled to half-precision.

![NN Linear Layer - Computation Graph - MP]({{site.baseurl}}/assets/images/torch-activation-memory-nn-linear-mp.png)

Unfortunately, torchviz does not display precision information. But we can obtain this information by traversing the computation graph. Here is a BFS code to traverse the graph and print function and tensor data.

```python
from collections import deque
import torch

def is_tensor(t):
    return isinstance(t, torch.Tensor)

def shape(t:torch.Tensor):
    return list(t.size())

def bsf_print(y, named_parameters=None, print_saved_tensors=True):
    
    named_parameter_pairs = list(named_parameters)
    accounted_address= set()
    parameter_index = dict()

    if named_parameters:
        parameter_index = {tensor:name for name, tensor in named_parameter_pairs}
        accounted_address = { t.untyped_storage().data_ptr() for n,t in named_parameter_pairs}

    queue = deque([y.grad_fn])
    visited = set()
    
    print("")
    print("Computation graph nodes:")

    while queue:
        node = queue.popleft()
        if node in visited:
            continue

        visited.add(node)

        if "AccumulateGrad" in node.name():
            tensor_var = node.variable
            assert is_tensor(tensor_var)
            assert tensor_var.requires_grad
            
            # do not add this to activation calculations
            accounted_address.add(tensor_var.untyped_storage().data_ptr())

            if tensor_var in parameter_index:
                print(f"* AccumulateGrad - {parameter_index[tensor_var]} - {list(tensor_var.size())} - dtype: {str(tensor_var.dtype):8} - Addr: {tensor_var.untyped_storage().data_ptr():13}")
            else:
                print(f"* AccumulateGrad - 'Name not known' - {list(tensor_var.size())} - dtype: {str(tensor_var.dtype):8} - Addr: {tensor_var.untyped_storage().data_ptr():13}")
        else:
            print(f"- {node.name()}")
        
        saved_tensor_data = [(atr[7:], getattr(node, atr)) for atr in dir(node) if atr.startswith("_saved_")]
        if saved_tensor_data and (print_saved_tensors):
            tensor_data = [data for data in saved_tensor_data if is_tensor(data[1])]

            # handle tensors
            if tensor_data and print_saved_tensors:
                for data in tensor_data:
                    t = data[1]
                    print(f"[{data[0]:>13}] - dtype: {str(t.dtype)[6:]:8} - Shape: {str(shape(t)):<18} - Addr: {t.untyped_storage().data_ptr():13} - NBytes: {t.untyped_storage().nbytes():>12,} - Size: {t.dtype.itemsize*t.size().numel():>12,}")
        
        # add children to the queue
        queue.extend([next_fn_pair[0] for next_fn_pair in node.next_functions if next_fn_pair[0] is not None and next_fn_pair[0] not in visited])
```

Now print the graph of the `y` tensor.

```python
bsf_print(y, named_parameters=w.named_parameters())
```

AccumulatedGrad nodes represents the weight tensors. Note that weights are in full precision. `mat1` tensor saved by `AddmmBackward0` function is down-scaled version of `weight` tensor of the linear layer. Note that `mat1` tensor is in half precision.

```text
Computation graph nodes:
- ViewBackward0
- AddmmBackward0
[         mat1] - dtype: float16  - Shape: [1, 1]             - Addr: 98196682887744 - NBytes:            2 - Size:            2
- ToCopyBackward0
- TBackward0
* AccumulateGrad - bias - [1] - dtype: torch.float32 - Addr: 98196682467904
- ToCopyBackward0
* AccumulateGrad - weight - [1, 1] - dtype: torch.float32 - Addr: 98196682648256
```

If you uncomment `nn.LayerNorm` line and view the tensors, you will see that LayerNorm is sensitive to precision thus keeps its values in full precision.

## References

1- [CS231n Winter 2016: Lecture 4: Backpropagation, Neural Networks 1](https://www.youtube.com/watch?v=i94OvYb6noo)

2- [The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0)

3- [A guide to PyTorch's CUDA Caching Allocator](https://zdevito.github.io/2022/08/04/cuda-caching-allocator.html)

4- [Scaling Laws Notebook by Karpathy](https://github.com/karpathy/nanoGPT/blob/master/scaling_laws.ipynb)