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


## What is activation memory

Consider the following linear transformation function which is very common in neural networks:

y = Wx


The partial derivatives are:

$$\frac{\partial y}{\partial x} = W$$

$$\frac{\partial y}{\partial W} = x$$

## References

1- [CS231n Winter 2016: Lecture 4: Backpropagation, Neural Networks 1](https://www.youtube.com/watch?v=i94OvYb6noo)

2- [The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0)

3- [The Llama 3 Herd of Models](https://arxiv.org/pdf/2407.21783)

4- [Scaling Laws Notebook by Karpathy](https://github.com/karpathy/nanoGPT/blob/master/scaling_laws.ipynb)