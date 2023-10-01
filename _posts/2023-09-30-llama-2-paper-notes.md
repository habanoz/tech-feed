---
layout: single
title:  "LLAMA 2 Paper Notes"
classes: wide

categories:
  - notes

tags:
  - llama 2
  - paper
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

### Supervised Fine-Tuning (SFT)
- bootstrap using public instruction tuning data
- Third-party SFT data is available from many different sources, but we found that many of these have insufficient diversity and quality — in particular for aligning LLMs towards dialogue-style instructions.
- Focused first on collecting several thousand examples of high-quality SFT data.
- By setting aside millions of examples from third-party datasets and using fewer but higher-quality examples from vendor-based annotation efforts, results notably improved. (Quality Is All You Need)
- A limited set of clean instruction-tuning data can be sufficient to reach a high level of quality
- SFT annotations in the order of tens of thousands was enough to achieve a high-quality result. Stopped annotating SFT after collecting a total of 27,540 annotations.
- Another bservation is that different annotation platforms and vendors can result in markedly different down-stream model performance, highlighting the importance of data checks even when using vendors to source annotations.
- A surprising finding is that the outputs sampled from the resulting SFT model were often competitive with SFT data handwritten by human annotators, suggesting that authors could reprioritize and devote more annotation effort to preference-based annotation for RLHF.

#### Fine-Tuning Details

- Cosine learning rate schedule an initial learning rate of 2 × 10−5
- a weight decay of 0.1
- a batch size of 64
- a sequence length of 4096 tokens.
- each sample consists of a prompt and an answer. To ensure the model sequence length is properly filled, concatenated all the prompts and answers from the training set. A special token is utilized to separate the prompt and answer segments.
- Utilized an autoregressive objective and zero-out the loss on tokens from the user prompt, so as a result, backpropagated only on answer tokens. 
- fine-tuned the model for 2 epochs.

### Reinforcement Learning with Human Feedback (RLHF)

RLHF procedure is needed to further align model with human preferences and istruction following.  

#### Human Preference Data Collection

- Binary comparison protocol is used.
- Annotators write a prompt then choose between two sampled responses (e.g. different temperature, different model variants) based on provided criteria. Annotators also label the degree of preference: significantly better, better, slightly better, or negligibly better/ unsure. 
- Two types of annotations are collected: helpfulness and safety.
- Safety annotations also include safety labeling: 1) the preferred response
is safe and the other response is not, 2) both responses are safe, and 3) both responses are unsafe, with
18%, 47%, and 35% of the safety dataset falling into each bin, respectively.
- Human annotations are collected in a weekly basis. Annotations are used to improve reward model. Better reward models are used to train better chat models. 
- Over 1M binary comparisons are collected in 14 batches (according to Table 26 of the paper)

![table26-preference-data]({{site.baseurl}}/assets/images/llama2-table-26.png)

#### Reward Modeling

A reward models takes chat history and generates a scalar reward to indicate how well the generated answer does. Safety and helpfulness are competing objectives. It is easier to train separate reward models for each objective.

Reward model is initialized from pretrained chat model checkpoints. This ensures that language model and reward model has the same knowledge.

I am not sure why it is indicated to be a "pretrained chat model", it should be more clear to say "pretrained model", because a pretrained model becomes a chat model only after SFT stage.

![Tasks]({{site.baseurl}}/assets/images/llama2-table-6.png)

*Training Objectives*: 

$$\mathcal{L}_{ranking}=-log(\sigma( r_{\theta}(x,y_c) - r_{\theta}(x,y_r) - m(r) ))$$

Where $$r_{\theta}(x,y_c)$$ is scalar score output for prompt $$x$$ and completion $$y$$ with model weights $$\theta$$. $$y_c$$ is preferred answer. $$y_r$$ is rejected answer. margin component $$m(r)$$ is a discrete function of preference rating.

Training procedure update reward function weights such that difference between accepted and rejected answer scores is maximized. Margin component tries to further move apart scores especially for dissimilar responses. Margin component is novel and added by the authors to the loss function. It is shown that margin component imroves Helpfulness reward model accuracy especially on samples
where two responses are more separable. However it needs to be used with caution. 

![margins]({{site.baseurl}}/assets/images/llama2-table-27.png)

Table 27 of the paper shows different margin values tried.

![margin-effects]({{site.baseurl}}/assets/images/llama2-table-28.png)

Table 28 of the paper shows affect of margin on model performance. Small margin helps model. Large margin improves model on more separable responses while degrades model on more similar responses.

Figure 27 of the paper shows affect of margin component on reward model output density. As margin increases, model generates more marginal scores. A large margin causes large distribution shift in reward model which may affect PPO performance which is sensitive to reward distribution change. Authors note to invest more in reward calibration future work.  

![reward-shift-by-margin]({{site.baseurl}}/assets/images/llama2-figure-27.png)

## References
1. [Github Source Code](https://github.com/habanoz/crawl-for-vector-db)
