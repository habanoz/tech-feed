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
- Grouped query attention (for bigger models 34B and 70B)
- BPE tokenizer (SentencePiece implementation). Numbers are split into individual digits. Use bytes to decompose unknown UTF-8 characters. Vocabulary size is 32k tokens.

#### Hyperparameters
- AdamW optimizer with $$\beta_1 = 0.9$$, $$\beta_2 = 0.95$$, $$eps = 10^{-5}$$
- Cosine learning rate schedule
- warmpup of 2000 steps
- wight decay 0.1
- gradient clipping 0f 1.0
- Global batch size of 4M
- LR for 7B and 13B is $$3.0x10^{-4}$$
- LR for 34B and 70B is $$1.5x10^{-4}$$
- Learning rate decay down to 10%
- Context length 4k


## Fine-tuning


## References
1. [Github Source Code](https://github.com/habanoz/crawl-for-vector-db)
