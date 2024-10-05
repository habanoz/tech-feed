---
layout: single
title: "LLM Scaling Laws"
categories:
  - workshop

tags:
  - llama
  - scaling laws
  - llm
mathjax: true
---

Training language models is an expensive business and it is important to plan carefully ahead of training. This post will discuss the scaling laws of LLMs and how to plan ahead for training.

Scaling laws tries to answer the question: Given a compute budget what is the optimal amount of tokens to train a model? 

## Kaplan scaling laws

First popular study on scaling laws is done by Jared Kaplan of OpenAI [1].

![Kaplan - Scaling Laws for Neural Language Models - Table 1]({{site.baseurl}}/assets/images/kaplan-nn-scaling-laws-table-1.png)

Table 1 shows formula for computing FLOPs per token for forward pass. Since number of parameters of a large language model is significantly greater than context length, context dependent terms is generally omitted. 
As another approximation, backward pass FLOPs are assumed to be 2 times the forward pass. 

Consequently relation between total number of FLOPs C, total tokens D and number of parameters N is given by:

$$ C = 6 * N * D $$

Kaplan scaling laws assumes a power law relationship between optimal number of parameters and compute and a power law relationship between optimal number of tokens and compute. According to Table 6 of [1] the scaling laws can be expressed as:

$$ N \propto C^{0.73} $$

$$ D \propto C^{0.27} $$

These scaling laws claim that if number of parameters is increased, it is necessary to increase dataset size. Increase in dataset size is less significant.


## Chinchilla scaling laws

Later in [2] those findings are invalidated. Chinchilla scaling laws assumes similar power-law relationships but claims that dataset size should be increased at the same rate with the model size. (See Table 2 of [2]).

![Hoffmann-Scaling laws-Table 2](hoffmann-scaling-laws-table-2.png)

[2] models training loss as a function of dataset size and model size.

![Hoffmann-Scaling laws-Equation 2](hoffmann-scaling-laws-equation-2.png)

Estimations for the unknown values are given in appendix D.2 of [2].

![Hoffmann-Scaling laws-Equation 10](hoffmann-scaling-laws-equation-10.png)

Using the values it is possible to formulate optimal number of tokens and parameters as in equation 4 of [2].

![Hoffmann-Scaling laws-Equation 4](hoffmann-scaling-laws-equation-4.png)

However, using the estimated parameter values from appendix D.2 of [2], I was unable to recrate results in Table 3 of [2].

![Hoffmann-Scaling laws-Equation 4 - Example 1](hoffmann-scaling-laws-equation-4-ex-1.png)

But according to Table 3 of [2], model size should have been 67B, dataset size should have been 1.5T. This mismatch is probably due to lack of precision in the given parameter values. In this form these numbers are useless. 


## Llama 3 Scaling Laws

In [3], llama paper introduces its own scaling laws with similar assumptions. The power relationship between compute and dataset size is given in:

$$ N^{*}(C) = AC^{\alpha} $$

In the paper, $$(\alpha, A)$$ is given as (0.53, 0.29). Unfortunately, with these values it is not possible to recreate their projection of 402B parameters and 16.55T tokens.

Also they show deviation from FLOP approximation of Kaplan [1]. They seem to use a slightly different approximation:

$$ C = 5.711 * N * D $$ 

which can be easily found by using their compute budget, and projected parameter count and dataset size. From this point on, one can leave A as-is and try to obtain a better estimation of α . 

$$ N^{*}(C) = AC^{\alpha} $$

$$ 16.55x10^{12} = 0.299x(3.8x10^{25})^{\alpha} $$

$$ 16.55x10^{12} / 0.299 = (3.8x10^{25})^{\alpha} $$

$$ log(16.55x10^{12} / 0.299) = \alpha x log(3.8x10^{25}) $$

$$ \alpha = 0.5372651710628614 $$

With this value, it is possible to match values in the paper:

![Llama-Scaling Laws-Correction](llama3-scaling-laws-correction-1.png)


## References

1- [Scaling Laws for Neural Language Models](https://arxiv.org/pdf/2001.08361)

2- [Training Compute-Optimal Large Language Models](https://arxiv.org/pdf/2203.15556)

3- [The Llama 3 Herd of Models](https://arxiv.org/pdf/2407.21783)