---
layout: single
title:  "LLAMA 2 Paper Analaysis"
classes: wide

categories:
  - analysis

tags:
  - llama 2
  - papaer
  - rlhf
  - llm
  - training
mathjax: true
---

## Pretraining

### Data
- 2 trillion tokens: for a good perfermance cost trade off
- up-sampling factual sources

### Training
- prenormalization using RMSNorm
- SwiGLU activation function
- Rotary positional embeddings
- Grouped query attention

#### Hyperparameters
- AdamW optimizer with $\beta_1 = 0.9$, $\beta_2 = 0.95$, $\eps = 10^-5$
- $$\eps = 10^-5$$

## References
1. [Github Source Code](https://github.com/habanoz/crawl-for-vector-db)
