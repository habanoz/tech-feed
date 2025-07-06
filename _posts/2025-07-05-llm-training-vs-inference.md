---
layout: single
title: "LLM Training vs Serving in Terms of Memory Access Patterns"
classes: wide
categories:
  - llm

tags:
  - llm
  - training
  - inference
  - memory
mathjax: true
---

This article investigates differences of LLM training vs serving in terms of memory access patterns. 

Serving is to load the model and allow the users send generation requests. LLM generation involves multiple inference steps. In each step a token is generated. Then the generated token is prepended to input prompt to generate a new token in the following step. This process is referred as auto regressive decoding. 

For the sake of simplicity this article focus on single GPU case.

## Purpose

Let $V$ be the vocabulary of all possible tokens. Let a sequence of tokens be $x_{1:t}=(x_1,x_2,…,x_t)$, where $x_i∈V$. The LLM outputs a probability mass function (PMF) over all possible next tokens conditioned on the input sequence:

$$P(x_{t+1}=x \mid x_1,x_2,…,x_t)$$

for all $x∈V$. 

Using this PMF, it is possible to calculate $P(x_{t+1} \mid x_1,x_2,…,x_t)$ which gives probability of outputting token $x_{t+1}$ given the previous tokens $x_{1:t}$


An LLM consists of multiple layers of transformers which are bunch of matrices. The purpose of the training process is to find values of matrix elements such that the LLM can output PDFs that are close to actual PDFs. The purpose of the inference is to produce a PDF for a given input sequence.

> Please note that inference does not lead to a text completion. Inference only generates one token. To achieve a completion, it is necessary to run inference and sample from the pmf multiple times using a specific sampling strategy until a termination condition is met. This process is known as decoding. 

## Objectives

During training, massive amounts of text is consumed. Training throughput is defined as amount of tokens consumed during training. It is important to achieve maximum throughput achievable. This ensures high GPU utilization and low total training time. Low throughput on the other hand leads to longer training times and higher costs. 

During serving, throughput also matters. Serving throughput is defined as amount of tokens produced. It is important to have high GPU utilization for cheaper serving. Latency is defined as total time waited to obtain a completion text for the given prompt. 

During serving, requests can be interactive or batch-oriented. For interactive requests, latency is very important.

## Attention Calculation

Scaled dot-product attention:

$Attention(Q, K, V) = softmax(\frac{Q K^T}{\sqrt{d_k}})V $

Where:
- $Q = x W^Q$ -> (Dimensions: Batch Size (B) x Sequence Length (T) x Hidden Dimension (D))
- $K = x W^K$ -> (B x T x D)
- $V = x W^V$ -> (B x T x D)

In python:

```python
## Forward function for training
# x.shape -> (B, T, D)

q = self.query(x)  # -> shape: (B, T, D)
k = self.key(x)    # -> shape: (B, T, D)
v = self.value(x)  # -> shape: (B, T, D)

att = (q @ k.transpose(-2, -1)) # -> (B, T, D) * (B, D, T) -> (B, T, T)
att = att * (1.0 / math.sqrt(k.size(-1))) # scale
att = F.softmax(att, dim=-1)
y = att @ v # (B, T, T) x (B, T, D) -> (B, T, D)

```

### Training

During training the data is readily available. Once a batch is finished, the processing of the next batch immediately begins. All batches are in the same shape, meaning they have the same length, T. Attention is calculated for all input tokens which has $T^2$ complexity where each of the T tokens performs attention over all previous tokens. 

During serving, input data may not be readily available. Request sizes can vary significantly. GPU requires processed batches to be of the same length. If different, they are left padded to the maximum length in the batch.   

### Inference

![Prefill vs Decoding]({{site.baseurl}}/assets/images/sarathi-serve.png)
> Source [2]

Generation has two distinct phases: prefill and decode. 

During prefill phase, attention over the whole sequence is calculated where each token attends to all tokens in the sequence. Prefill phase is compute bound with $O(T^2)$ complexity, similar to how attention is calculated during training. Unlike training, K and V matrices are stored in so called KV cache to be used in decoding phase. 

Decode phase involves multiple decode steps. Note that Q and K values are only calculated for the last token generated. K and V values for the past tokens are retrieved from the KV cache. 

```python
## Forward function for decoding
# x.shape -> (B, 1, D)
q = self.query(x)  # -> shape: (B, 1, D)
k = self.key(x)    # -> shape: (B, 1, D)
v = self.value(x)  # -> shape: (B, 1, D)

# add newly calculated matrices to the cache
self.cache_k[:bsz, cur_pos : cur_pos + 1] = k
self.cache_v[:bsz, cur_pos : cur_pos + 1] = v

k = self.cache_k[:bsz, : cur_pos + 1] # get all past keys
v = self.cache_v[:bsz, : cur_pos + 1] # get all past values

att = (q @ k.transpose(-2, -1)) # -> (B, 1, D) * (B, D, T) -> (B, 1, T)
att = att * (1.0 / math.sqrt(k.size(-1))) # scale
att = F.softmax(att, dim=-1)
y = att @ v # (B, 1, T) x (B, T, D) -> (B, 1, D)

```

Also note that for decoding attention formulation can be expressed in the following terms:

$Attention(Q_t, K_{1:t}, V_{1:t}) $

This formulation provides a hint on why key and values are cached but not queries.  

While the math during decoding phase is similar, its complexity is $O(T)$. Each new token only needs to attend to previous tokens. Since less calculations are done, unlike prefill phase, decode phase is memory bound. 

## Memory Transfer

Matrix multiplication is done in tensor cores. Tensor cores need the tensors to be transferred from HBM memory to shared memory. 

A GPU has two main resources: FLOPS and Memory Bandwidth. FLOPS is the amount of floating point operations can be done while bandwidth determines amount of data that can be transferred from/to shared memory. A nice thing about the GPU is that it is possible to overlap computation and transfer. It is possible to transfer next weights while doing multiplication in tensor cores. 

If tensor cores have very few computations to make they become idle until next weights are transferred. In this case the operation is said to be memory bound. If tensor cores have a lot of computations to make they will not be able to process next weights even if their transfer is completed. In this case the operation is said to be compute bound.
The optimal case is where computation in tensor cores and memory transfer is completed at the same time. 

The optimal batch size can help the balance the computation and transfer time to achieve high GPU utilization thus high throughput. 

Training and Prefill operations are compute bound. They can utilize GPU well. However, decode operation is memory bound. Without proper batching GPU utilization will be low, the time will be spent on waiting weights to be transferred. GPU flops will be wasted. While this is acceptable in terms of latency, it is not desirable in terms of throughput and cost of inference. 

## Memory Usage

### Training 

LLM Training is an optimization process. Involves two passes: forward pass and backward pass. 

During the forward pass, activations are calculated and save for backward pass. At the end of the forward pass the loss is calculated.

During the backward pass, training uses the chain rule of calculus to compute the gradient of the loss with respect to model parameters. This requires:

$ \frac{\partial L}{\partial \theta} = \frac{\partial L}{\partial a} . \frac{\partial a}{\partial \theta} $

Where:
    - L is the loss
    - a is the activation (output of some layer)
    - θ is a parameter (e.g., a weight)

Optimizer uses the gradients to update model weights. Most optimizers keep their internal states. For example AdamW optimizer keeps two state variables: mean and variance of the past gradients.

AdamW Update Rule:

$\theta \gets \theta - η \frac{m_t}{\sqrt{v_t}} - \lambda . \theta $

Where:
- η: learning rate
- $m_t$​: first moment estimate (mean of gradients)
- $v_t$: second moment estimate (variance of gradients)
- λ: weight decay (L2 regularization)

So during training, GPU memory is used by model weights, activations, gradients and optimizer states. 

```python
param_mem = N_params × 2 # usually stored in fp16/bf16

# B = batch size
# L = number of layers
# D = model hidden dimension
# factor = accounts for intermediate activations (usually 3–5× for attention, MLP, etc.)
activation_memory ≈ B × L × D × N_layers × dtype_size × factor

grad_mem = N_params × 4 bytes # usually stored in fp32
opt_state_mem = N_params × 2 × 4 bytes  # usually stored in fp32, Adam keeps two states

total_mem ≈ param_mem + activation_mem + grad_mem + opt_state_mem
```

### Inference

During LLM inference only forward pass is invoked. Since backward pass is not invoked, states used during backward pass are not needed e.g. activations, optimizer states, gradients. LLM inference uses KV cache to avoid recomputation of key and values during subsequent decoding steps. 

```python
param_mem = N_params × 2 # usually stored in fp16/bf16

# D = hidden dimension
# L = number of layers
# 2 bytes for each value (fp16/bf16)
# 2 for K and V
kv_mem = 2 x 2 x B x T x D x L 

total_mem ≈ param_mem + kv_mem
```

## References

1- [Attention is all you need](https://arxiv.org/pdf/1706.03762)

2- [Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve](https://arxiv.org/pdf/2403.02310)

3- [Matrix multiplication with Tensor Cores](https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/#Matrix_multiplication_with_Tensor_Cores)

4- [Simple Attention Implementation](https://github.com/habanoz/lm_model_notebooks/blob/main/9_multi-headed-attention-lm.ipynb)

5- [Simple Attention Implementation - Make More](https://github.com/karpathy/makemore/blob/master/makemore.py)

6- [Attention Implementation - Llama3 Reference](https://github.com/meta-llama/llama3/blob/main/llama/model.py)