---
layout: single
title:  "QLORA: Efficient Finetuning of Quantized LLMs"
classes: wide

categories:
  - notes

tags:
  - paper
  - llm
  - fine-tunning
mathjax: true
---

Contributions:

**4-bit NormalFloat (NF4)**:  An information theoretically optimal quantization data type for normally distributed data that yields better empirical results than 4-bit Integers and 4-bit Floats.

**Double Quantization** A method that quantizes the quantization constants, saving an average of about 0.37 bits per parameter (approximately 3 GB for a 65B model). 

**Paged Optimizers** using NVIDIA unified memory to avoid the gradient checkpointing memory spikes that occur when processing a mini-batch with a long sequence length.

With Qlora it becomes possible to fine-tune a 65B model which required 780GB of GPU memory with a single 48GB GPU.

A pretrained model is quantized to 4-bit. Then a set of learnable low-rank adapters weights are added. Adapters are tuned by backpropagation.

## Introduction

Finetunning Observations:

**Data quality**  is far more important than dataset size, (For exmaple: a 9k sample dataset (OASST1) outperformed a 450k sample dataset (FLAN v2, subsampled) on chatbot performance, even when both are meant to support instruction following generalization.)

**Dataset suitability** matters more than size for a given task. (For example:  strong Massive Multitask Language Understanding (MMLU) benchmark performance does not imply strong Vicuna chatbot benchmark performance and vice versa)

## Background

### Block-wise k-bit Quantization

Quantization is process of discretizing an input from a representation that holds more information to a represenation that holds less information. e.g. 32-bit floats to 8-bit ingegers. 

Assume input elements are in the form of tensors. To ensure entire range of low bit data type is used, thwe input data type is rescaled into target data type range through normalization by the absolute maximum of the input elements. 

Following figure shows how an 32-bit floating point tensor is quantized into 8-bit tensor with range [-127, 127].


![data-pre-processing]({{site.baseurl}}/assets/images/qlora-quant-1.png)

The problem is that if a large magnitude value occurs in the input tensor, then quantization bins are not utilized well. e.g. few or none numbers are quantized into some bins. 

To prevent outlier issue, a comman approach is to chunk input tensor into block that are independently quantized, with a seperate quantization constant c. For n different blocks we will have n different quantization constants.

### Low-rank Adapters

Model parameters are fixed. Adapter parameters are updated during gradient descent. 

![data-pre-processing]({{site.baseurl}}/assets/images/qlora-adapter-2.png)

### Memory Requirement of Parameter-Efficient Finetuning

Activation gradients uses significant amount of memory. This is more obvious in Lora. 

For example, a 7B model trained on FLAN v2 with a batch size of 1, lora input gradients have a memory footprint of 567MB while lora parameters take up only 26MB. (In typical scenario where LoRA weights equivalent to commonly used 0.2% of the original model weights ).

With gradient checkpointing, the input gradients reduce to 18MB. In contrast, quantized base model takes up 5048 MB of memory. 

To conclude, amount of lora parameters should not be tuned for porpuse of saving memory as lora parameters are inportant for performance of the final model.

## References
1. [Qlora paper](https://arxiv.org/pdf/2305.14314.pdf)
2. [Qlora code repository](https://github.com/artidoro/qlora)
3. [Bits and Bytes code repository](https://github.com/TimDettmers/bitsandbytes)