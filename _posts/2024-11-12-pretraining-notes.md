---
layout: single
title: "Haber GPT2"
classes: wide
categories:
  - llm

tags:
  - gpt2
  - rope
  - pretraining

---

HaberGPT2 is a decoder only language model with 100M parameters. It is trained from scratch using turkish news. This post shares details about training process.

## Data

The model is trained using `habanoz/news-tr-1.8M` dataset[1]. See [2] on how dataset is generated.

## Tokenizer

The Tokenizer was trained on the `habanoz/news-tr-1.8M` dataset using the "sentencepiece" library[3]. 

A relatively small vocabulary size of 8192 was chosen to prioritize faster training speeds, resulting in a 33% increase in throughput compared to a 32K vocabulary size. This smaller vocabulary enables the use of larger micro batch sizes, making the model training more efficient. However, the impact of vocabulary size on the performance of downstream tasks was not investigated.

| Vocab Size | Tokens        | Relative |
|------------|---------------|----------|
| 32K        | 871.296.394   | 81%      |
| 16K        | 955.567.472   | 89%      |
| 8K         | 1.072.343.909 | 100%     |

Table shows number of tokens for `habanoz/news-tr-1.8M` dataset using each tokenizers. Relative column is relative to 8K tokenizer. With 8k vocabulary, `habanoz/news-tr-1.8M` dataset contains 1B tokens. 

## Architecture

I have trained many 10M and 40M parameter models to find optimum architecture for throughput. Starting from the original GPT2 architecture, I have decided to include following changes:

- **Rotary Positional Embedding (ROPE)**[4]: ROPE harms throughput while improving validation loss. To mitigate throughput degradation, I have pre-computed and cached rotation matrices. With ROPE throughput is down by 10% while validation loss decreases by 2%. Despite significant impact on throughput, I have decided to continue with throughput.

- **Group Query Attention (GQA)**: GQA improves throughput with no visible impact on validation loss. Throughput gain over ROPE baseline is about 2%. I used half the number of value heads.

- **Deep Layers**: Deep models are known to outperform wide models[5]. Deeper models are more capable at finding complex relationships between tokens while wider models have more capacity to learn facts. For smaller models depth is more preferable.  

- **Embedding Sharing**: Token embedding weights and output layer weights are shared. While embedding sharing has a small performance impact, it is a common practice for small models[5,9]. For large models, output layer size is negligible and output layer is not shared with token embeddings. 

Notably, I did not use SwiGLU[7] activation, which was employed at [5], because during my tests I observed significant throughput degradation. 

I also experimented with Squared Relu activation[8], and found it to hurt throughput. I decided not to use Squared Relu. My later experiments with `torch.compile` reveled actually it improves the throughput.

RMSNorm[6] was also found to hurt throughput which is surprising because RMSNorm is simpler alternative to the LayerNorm. My guess is the GPU I used is old and lacks optimized kernels for RMSNorm in contrast to LayerNorm. (Torch compile does not help with RMSNorm.)

Architecture Details:

- Number of Hidden Layers: 32

- Model dimension : 512

- Number of Heads: 8 (4 kq heads, 8 value heads)

- Sequence Length: 512

- Vocabulary Size: 8192

- Total Parameters: 108.04M

- Total Parameters w/o Embeddings: 108.03M


## Training

The model is trained using 11B tokens (11 epochs). Each sample is prefixed with BOS token and suffixed with EOS token. Samples are concatenated to fit sequence length. 

- Optimizer: AdamW (beta1 0.9, beta2 0.95)

- Learning Rate: 0.0018

- Learning Rate Scheduler: Cosine schedular (2000 warmup steps, Decay to 10% of initial LR)

- Batch size: 552960 (0.5M)

- Number of Steps: 20K


2 Tesla T4 GPUs are used in mixed precision mode. Training took 147 hours (roughly 6.5 days). 

### Sequence length limitation

Sequence length is only 512, which is not typical in LLM training after original GPT1 model. Attention calculation has quadratic complexity. The popular solution is to use Flash attention which is only available for more recent GPUS starting from volta series. Since tesla series of GPUs are supported by flash attention, I cannot increase sequence length.

## Results

Final validation loss is 2.09, perplexity is 8.07. A test notebook is provided to test the model [11].

The model can output coherent completions in wide variety of topics included in the training set. However, the model does not produce and newlines. Sometimes outputs are not logical continuations. Sometimes, especially in greedy decoding, model tends to repeat sentences. 

My conclusions:

- News dataset is not a quality distribution to learn the language. There are unnecessary repetitions and long sentences. 

- Sequence length limitation maybe a major obstacle for model output quality.

- Training for 10 epochs is too much. It is shown that multi-epoch training causes degradation[10]. A larger dataset is needed to avoid reusing tokens. 

I share model weights [12], dataset[1], training code[13] and testing notebook[11] for practitioners.

## References

1- [News-tr-1.8M Dataset](https://huggingface.co/datasets/habanoz/news-tr-1.8M)

2- [Building news-tr-1.8M Dataset]({{site.baseurl}}/workshop/collecting-1_8_M_news_documents/)

3- [Sentencepiece repository](https://github.com/google/sentencepiece)

4- [ROFORMER: ENHANCED TRANSFORMER WITH ROTARY POSITION EMBEDDING](https://arxiv.org/pdf/2104.09864)

5- [MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases](https://arxiv.org/pdf/2402.14905)

6- [RMSNorm](https://arxiv.org/pdf/1910.07467)

7- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)

8- [Primer: Searching for Efficient Transformers for Language Modeling](https://arxiv.org/pdf/2109.08668)

9- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

10-[o Repeat or Not To Repeat:Insights from Scaling LLM under Token-Crisis](https://arxiv.org/pdf/2305.13230)

11-[Test Notebook](https://github.com/habanoz/nb_gpu_trainer/blob/main/evaluate_model_100m.ipynb)

12-[HaberGPT2-100M](https://huggingface.co/habanoz/haber-gpt-2.1-100M-8k-v1.08)

13-[Trainer Repository](https://github.com/habanoz/nb_gpu_trainer)