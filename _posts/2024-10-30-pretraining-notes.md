---
layout: single
title: "Transformer Pre-training Notes"
categories:
  - math

tags:
  - gpt3
  - llama
  - deep-seek
  - qween
  - chihcilla
  - gopher
  - paper notes
  - pretraining
---


## GPT3

### Architecture

| Model Name   | nparams     | nlayers | dmodel | nheads | dhead | Batch | Learning Rate |
|--------------|-------------|---------|--------|--------|-------|-------|---------------|
| GPT-3 Small  | 125M        | 12      | 768    | 12     | 64    | 0.5M  | 6.0 × 10−4    |
| GPT-3 Medium | 350M        | 24      | 1024   | 16     | 64    | 0.5M  | 3.0 × 10−4    |
| GPT-3 Large  | 760M        | 24      | 1536   | 16     | 96    | 0.5M  | 2.5 × 10−4    |
| GPT-3 XL     | 1.3B        | 24      | 2048   | 24     | 128   | 1M    | 2.0 × 10−4    |
| GPT-3 2.7B   | 2.7B        | 32      | 2560   | 32     | 80    | 1M    | 1.6 × 10−4    |
| GPT-3 6.7B   | 6.7B        | 32      | 4096   | 32     | 128   | 2M    | 1.2 × 10−4    |
| GPT-3 13B    | 13.0B       | 40      | 5140   | 40     | 128   | 2M    | 1.0 × 10−4    |
| GPT-3        | 175B(GPT-3) | 96      | 12288  | 96     | 128   | 3.2M  | 0.6 × 10−4    |

We use the same model and architecture as GPT-2 [ RWC+19 ], including the modified initialization, pre-normalization, and reversible tokenization described therein, with the exception that we use alternating dense and locally banded sparse attention patterns in the layers of the transformer, similar to the Sparse Transformer [ CGRS19 ].

### Training Details

To train all versions of GPT-3:

- All models were trained for a total of 300 billion tokens.

- Adam with β1 = 0.9, β2 = 0.95, and epsilon = 10−8

- Clip the global norm of the gradient at 1.0

- Cosine decay for learning rate down to 10% of its value, over 260 billion tokens (after 260 billion tokens, training continues at 10% of the original learning rate)

- There is a linear LR warmup over the first 375 million tokens

- Gradually increase the batch size linearly from a small value (32k tokens) to the full value over the first 4-12 billion tokens of training, depending on the model size. 

- Data are sampled without replacement during training (until an epoch boundary is reached) to minimize overfitting.

- Weight decay of 0.1 to provide a small amount of regularization [LH17].

During training we always train on sequences of the full nctx = 2048 token context window, packing multiple
documents into a single sequence when documents are shorter than 2048, in order to increase computational efficiency.
Sequences with multiple documents are not masked in any special way but instead documents within a sequence
are delimited with a special end of text token, giving the language model the information necessary to infer that
context separated by the end of text token is unrelated. This allows for efficient training without need for any special
sequence-specific masking.

## Gopher

### Architecture

| Model       | Layers | nHeads | KVSize | dmodel | LR       | Batch    |
|-------------|--------|--------|--------|--------|----------|----------|
| 44M         | 8      | 16     | 32     | 512    | 6x10-4   | 0.25M    |
| 117M        | 12     | 12     | 64     | 768    | 6x10-4   | 0.25M    |
| 417M        | 12     | 12     | 128    | 1,536  | 2x10-4   | 0.25M    |
| 1.4B        | 24     | 16     | 128    | 2,048  | 2x10-4   | 0.25M    |
| 7.1B        | 32     | 32     | 128    | 4,096  | 1.2x10-4 | 2M       |
| Gopher 280B | 80     | 128    | 128    | 16,384 | 4x10-5   | 3M -> 6M |

- The feed-forward size is always 4xdmodel.


We use the autoregressive Transformer architecture detailed in Radford et al. (2019) with two modifications: we use **RMSNorm** (Zhang and Sennrich, 2019) instead of LayerNorm (Ba et al., 2016), and we use the **relative positional encoding** scheme from Dai et al. (2019) rather than absolute positional encodings. 

Relative encodings permit us to evaluate on longer sequences than we trained on, which improves the modelling of articles and books as shown in Section D.6. We tokenize the text using SentencePiece (Kudo and Richardson, 2018) with a vocabulary of 32,000 and use a byte-level backoff to support open-vocabulary modelling. The Gopher model card (Mitchell et al., 2019) is included in Appendix B.

### Training Details

- 300 billion tokens with a 2048 token context window

- Adam optimizer

- Warm-up the learning rate from 10-7 to the maximum learning rate over the first 1500 steps

- Decay LR 10x using a cosine schedule

- As we increase model size, we decrease the maximum learning rate and increase the number of tokens in each batch, as shown in Table 1.

- Increase Gopher’s batch size from three to six million tokens per batch during training

- Clip gradients based on the global gradient norm using a clipping value of 1. For the 7.1B model and for Gopher we reduce this to 0.25 for improved stability.

We incorporate the bfloat16 numerical format to reduce memory and increase training throughput. Models smaller than 7.1B are trained with mixed precision float32 parameters and bfloat16 activations (Micikevicius et al., 2018), while 7.1B and 280B use bfloat16 activations and parameters. bfloat16 parameters are updated using stochastic rounding to maintain stability (Gupta et al., 2015). We subsequently found that stochastic rounding does not fully recover mixed precision training performance; more details can be found in Appendix C.

#### Lessons Learned

While at smaller scales we found pre-training with Adafactor to be stable and performant, at large scales we found that Adafactor resulted in reduced performance compared to Adam along with increased number of instabilities.

fp32 everywhere acts as a baseline– while it has the largest memory footprint, no compromises are made in numerical representation relative to the other methods. We find that bfloat16 parameters with a float32 copy stored in the partitioned optimiser state is indistinguishable in performance yet offers a reduced memory footprint and a 1.4x speed improvement.

In all configurations, we use fp32 for computing the attention softmax and the softmax cross-entropy in the loss. This stabilizes low-precision training with almost zero runtime cost on TPU. All methods using bfloat16 offer a similar 1.4x speed improvement over fp32 everywhere.


## Chincilla

### Training for Scaling Law Experiments

- Maximum learning rate of 2 x 10-4 for the smallest models and 1.25 x 10-4 for the largest models

- Learning rate drops by a factor of 10x during training, using a cosine schedule.

We make the assumption that the cosine cycle length should be approximately matched to the number of training steps. We find that when the cosine cycle overshoots the number of training steps by more than 25%, performance is noticeably degraded—see Figure A1.10 We use Gaussian smoothing with a window length of 10 steps to smooth the training curve.

### Differences between Chinchilla and Gopher

| Model          | Layers | nHeads | K/V Size | dmodel | LR     | Batch Size |
|----------------|--------|--------|----------|--------|--------|------------|
| Gopher 280B    | 80     | 128    | 128      | 16,384 | 4x10-5 | 3M -> 6M   |
| Chinchilla 70B | 80     | 64     | 128      | 8,192  | 1x10-4 | 1.5M -> 3M |


- Specifically, Gopher was trained with Adam (Kingma andBa, 2014) whereas Chinchilla was trained with AdamW.

- Chinchilla stored a higher-precision copy of the weights in the sharded optimiser state.

- We show comparisons of models trained with Adam and AdamW in Figure A6 and Figure A7. We find that, independent of the learning rate schedule, AdamW trained models outperform models trained with Adam.

- Interestingly, a model trained with AdamW only passes the training performance of a model trained with Adam around 80% of the way through the cosine cycle, though the ending performance is notably better– see Figure A7.

- The feed-forward size is always set to 4xdmodel. Note that we double the batch size midway through training for both Chinchilla and Gopher.

- We train Chinchilla with a slightly modified SentencePiece (Kudo and Richardson, 2018) tokenizer that does not apply NFKC normalisation. The vocabulary is very similar– 94.15% of tokens are the same as those used for training Gopher. We find that this particularly helps with the representation of mathematics and chemistry, for example.

- Whilst the forward and backward pass are computed in bfloat16, we store a float32 copy of the weights in the distributed optimiser state (Rajbhandari et al., 2020). See Lessons Learned from Rae et al. (2021) for additional details.


## GPT-NeoX-20B

### Architecture

GPT-NeoX-20B, a 20 billion parameter autoregressive Transformer language model trained on the Pile (Gao et al., 2020) dataset, and detail the main architectural differences between GPT-NeoX-20B and GPT-3—most notably the change in tokenizer, the addition of Rotary Positional Embeddings, the parallel computation of attention and feed-forward layers, and a different initialization scheme and hyperparameters.

- Rotary Positional Embeddings : Rotary embeddings are a form of static relative positional embeddings. In brief, they twist the embedding space such that the attention of a token at position m to token at position n is linearly dependent on m − n. 

While Su et al. (2021) apply rotary embeddings to every embedding vector, we follow Wang and Komatsuzaki (2021) and instead apply it only to the first 25% of embedding vector dimensions. Our initial experiments indicate that this strikes the best balance of performance and computational efficiency.

- Parallel Attention + FF Layers : We compute the Attention and Feed-Forward (FF) layers in parallel4 and sum the results, rather than running them in series. This is primarily for efficiency purposes, as each residual addition with op-sharding requires one all-reduce in the forward pass and one in the backwards pass (Shoeybi et al.,2020). By computing the Attention and FFs in parallel, the results can be reduced locally before per- forming a single all-reduce. In Mesh Transformer JAX (Wang, 2021), this led to a 15% throughput increase, while having comparable loss curves with running them in series during early training.

A tied layer norm the way Wang and Komatsuzaki (2021) does:
x + Attn(LN1(x)) + FF(LN1(x))

- Initialization : For the Feed-Forward output layers before the residuals, we used the initialization scheme introduced in Wang (2021), 2/L√d . For all other layers, we use
the small init scheme from Nguyen and Salazar (2019), √ 2 / d+4d .

- All Dense Layers : While GPT-3 uses alternating dense and sparse layers using the technique introduced in Child et al. (2019), we instead opt to exclusively use dense layers to reduce implementation complexity.

We opted to use the values from Brown et al. (2020) (GPT-3) to guide our choice of hyperparameter. 


### Training Details

- Interpolate between the learning rates of their 13B and 175B models to arrive at a learning rate of 0.97E−5 . 

- Based on the results of smaller scale experiments, we select a weight decay of 0.01 

- use the same batch size as OpenAI’s 175B model–approximately 3.15M tokens, or 1538 contexts of 2048 tokens each, and train for a total of 150,000 steps, decaying the learning rate with a cosine schedule to 10% of its original value at the end of training.

- We use the AdamW (Loshchilov and Hutter,2019) optimizer, with beta values of 0.9 and 0.95 respectively, and an epsilon of 1.0E−8. We extend AdamW with the ZeRO optimizer, to reduce memory consumption by distributing optimizer states across ranks. 

- Tensor parallel size of 2, and a pipeline parallel size of 4


## PALM

### Architecture

| Model     | Layers | #Heads | dmodel | #Parameters | Batch Size        |
|-----------|--------|--------|--------|-------------|-------------------|
| PaLM 8B   | 32     | 16     | 4096   | 8.63        | 256 → 512         |
| PaLM 62B  | 64     | 32     | 8192   | 62.50       | 512 → 1024        |
| PaLM 540B | 118    | 48     | 18432  | 540.35      | 512 → 1024 → 2048 |

- The feed-forward size dff is always 4 × dmodel and attention head size is always 256.

- SwiGLU Activation -- We use SwiGLU activations (Swish(xW ) · xV ) for the MLP intermediate activations because they have been shown to significantly increase quality compared to standard ReLU, GeLU, or Swish activations (Shazeer, 2020). Note that this does require three matrix multiplications in the MLP rather than two, 

- Parallel Layers -- We use a “parallel” formulation in each Transformer block (Wang & Komatsuzaki, 2021), rather than the standard “serialized” formulation. Specifically, the standard formulation can be written as:
y = x + MLP(LayerNorm(x + Attention(LayerNorm(x)))

Whereas the parallel formulation can be written as:
y = x + MLP(LayerNorm(x)) + Attention(LayerNorm(x))

- Multi-Query Attention

- RoPE Embeddings -- We use RoPE embeddings (Su et al., 2021) rather than absolute or relative position embeddings, since RoPE embeddings have been shown to have better performance on long sequence lengths

- Shared Input-Output Embeddings – We share the input and output embedding matrices, which is done frequently (but not universally) in past work.

- No Biases – No biases were used in any of the dense kernels or layer norms. We found this to result in increased training stability for large models.

- Vocabulary – We use a SentencePiece (Kudo & Richardson, 2018a) vocabulary with 256k tokens, which was chosen to support the large number of languages in the training corpus without excess tokenization. The vocabulary was generated from the training data, which we found improves training efficiency. The vocabulary is completely lossless and reversible, which means that whitespace is completely preserved in the vocabulary (especially important for code) and out-of-vocabulary Unicode characters are split into UTF-8 bytes, with a vocabulary token for each byte. Numbers are always split into individual digit tokens (e.g., “123.5 → 1 2 3 . 5”)

### Training Details


- Weight initialization : The kernel weights are initialized with "fan-in variance scaling". The input embeddings are initialized to E ∼ N (0, 1), since layer normalization is not applied to the embeddings.

- Optimizer : The model was trained with the Adafactor optimizer (Shazeer & Stern, 2018), without factorization

- Optimization hyperparameters – We use an Adafactor learning rate of 10−2 for the first 10,000 steps, which is then decayed at a rate of 1/√k, where k is the step number. We train with momentum of β1 = 0.9. The second-order moment interpolation value is computed as β2 = 1.0 − k−0.8, where k is the step number. We have found this to be more stable than the standard β2 = 0.99 when training large language models, because rare embedding tokens can have poorly estimated second moments over shorter windows. We use global norm gradient clipping (Pascanu et al. (2012)) with a value of 1.0 for all models. We use a dynamic weight decay of lr^2.0 during training, where lr is the current learning rate.

- Loss function – The model is trained with the standard language modeling loss function, which is the average log probability of all tokens without label smoothing. We additionally use an auxiliary loss of z loss = 10−4 · log2 Z to encourage the softmax normalizer log (Z) to be close to 0, which we found increases the stability of training.

- Sequence length – A sequence length of 2048 was used for all models. Input examples are concatenated together and then split into sequences of exactly 2048 tokens, so that there are no padding tokens, but examples may be split in the middle. Input examples are differentiated from one another with a special [eod] token.

- Batch size – For all models, we increase the batch size during training. For the largest model, we use batch size 512 (1M tokens) until step 50k, then double it to 1024 (2M tokens) until step 115k, and finally double again it to 2048 (4M tokens) until training is complete at step 255k. The smaller models followed similar schedules. The reason for using such batch size schedule is twofold: (1) smaller batch sizes are more sample efficient (i.e., better loss as a function of tokens seen) earlier in training, while larger batch sizes are beneficial later in training due to better gradient estimates (Smith et al., 2018; McCandlish et al., 2018), and (2) larger batch sizes result in larger matrix multiplication dimensions, which increases TPU efficiency.

- Dropout – The model was trained without dropout, although dropout of 0.1 is used for finetuning in most cases.

#### Training Efficiency

- Rematerialization

- PaLM achieves high accelerator utilization because of its parallelism strategy and several other factors, including XLA TPU compiler optimizations, and the use of “parallel layers” (see Section 2). We believe PaLM represents a significant step forward in LLM training efficiency.


## LLAMA

### Architecture

| params | dimension | nheads | nlayers | LR     | batch | tokens |
|--------|-----------|--------|---------|--------|-------|--------|
| 6.7B   | 4096      | 32     | 32      | 3.0e−4 | 4M    | 1.0T   |
| 13.0B  | 5120      | 40     | 40      | 3.0e−4 | 4M    | 1.0T   |
| 32.5B  | 6656      | 52     | 60      | 1.5e−4 | 4M    | 1.4T   |
| 65.2B  | 8192      | 64     | 80      | 1.5e−4 | 4M    | 1.4T   |

We leverage various improvements that were subsequently proposed, and used in different models such as PaLM.

Pre-normalization [GPT3]. To improve the training stability, we normalize the input of each transformer sub-layer, instead of normalizing the output. We use the RMSNorm normalizing function, introduced by Zhang and Sennrich (2019).

SwiGLU activation function [PaLM]. We replace the ReLU non-linearity by the SwiGLU activation function, introduced by Shazeer (2020) to improve the performance. We use a dimension of 2/3 4d instead of 4d as in PaLM.

Rotary Embeddings [GPTNeo]. We remove the absolute positional embeddings, and instead, add rotary positional embeddings (RoPE), introduced by Su et al. (2021), at each layer of the network.

### Training Details

Our training approach is similar to the methods described in previous work (Brown et al., 2020; Chowdhery et al., 2022), and is inspired by the Chinchilla scaling laws (Hoffmann et al., 2022).

Tokenizer. We tokenize the data with the byte-pair encoding (BPE) algorithm (Sennrich et al.,2015), using the implementation from SentencePiece (Kudo and Richardson, 2018). Notably, we split all numbers into individual digits, and fallback to bytes to decompose unknown UTF-8 characters.

For most of our training data, each token is used only once during training, with the exception of the Wikipedia and Books domains, over which we perform approximately two epochs.

- AdamW optimizer: β1 = 0.9, β2 = 0.95

- cosine learning rate schedule, such that the final learning rate is equal to 10% of the maximal learning rate.

- weight decay of 0.1

- gradient clipping of 1.0

- 2,000 warmup steps 

## LLAMA 2

### Architecture

LLAMA 1

| Params | SeqLen | GQA | Tokens | LR         |
|--------|--------|-----|--------|------------|
| 7B     | 2k     | X   | 1.0T   | 3.0 × 10−4 |
| 13B    | 2k     | X   | 1.0T   | 3.0 × 10−4 |
| 33B    | 2k     | X   | 1.4T   | 1.5 × 10−4 |
| 65B    | 2k     | X   | 1.4T   | 1.5 × 10−4 |


LLAMA 2

| Params | SeqLen | GQA | Tokens | LR         |
|--------|--------|-----|--------|------------|
| 7B     | 4k     | X   | 2.0T   | 3.0 × 10−4 |
| 13B    | 4k     | X   | 2.0T   | 3.0 × 10−4 |
| 34B    | 4k     | ✓   | 2.0T   | 1.5 × 10−4 |
| 70B    | 4k     | ✓   | 2.0T   | 1.5 × 10−4 |

We adopt most of the pretraining setting and model architecture from Llama 1. We use the standard transformer architecture (Vaswani et al., 2017), apply pre-normalization using RMSNorm (Zhang and Sennrich, 2019), use the SwiGLU activation function (Shazeer, 2020), and rotary positional embeddings (RoPE, Su et al. 2022). The primary architectural differences from Llama 1 include increased context length and grouped-query attention (GQA). 

### Training Details

- AdamW optimizer (Loshchilov and Hutter, 2017), with β1 = 0.9, β2 = 0.95, eps = 10−5.

- Cosine learning rate schedule, with warmup of 2000 steps, and decay final learning rate down to 10% of the peak learning rate. 

- A weight decay of 0.1 and gradient clipping of 1.0

## LLAMA 3

### Architecture


|                    | 8B     | 70B      | 405B   |
|--------------------|--------|----------|--------|
| Layers             | 32     | 80       | 126    |
| Model Dimension    | 4,096  | 8192     | 16,384 |
| FFN Dimension      | 14,336 | 28,672   | 53,248 |
| Attention Heads    | 32     | 64       | 128    |
| Key/Value Heads    | 8      | 8        | 8      |
| Peak Learning Rate | 3×10−4 | 1.5×10−4 | 8×10−5 |


We pre-train a model with 405B parameters on 15.6T tokens using a context window of 8K tokens. This standard pre-training stage is followed by a continued pre-training stage that increases the supported context window to 128K tokens.

We make a few small modifications compared to Llama 2:

- We use grouped query attention (GQA; Ainslie et al. (2023)) with 8 key-value heads to improve inference speed and to reduce the size of key-value caches during decoding.

- We use an attention mask that prevents self-attention between different documents within the same sequence. We find that this change had limited impact during in standard pre-training, but find it to be important in continued pre-training on very long sequences

- We use a vocabulary with 128K tokens. Our token vocabulary combines 100K tokens from the tiktoken3 tokenizer with 28K additional tokens to better support non-English languages.

- We increase the RoPE base frequency hyperparameter to 500,000. This enables us to better support longer contexts; Xiong et al. (2023) showed this value to be effective for context lengths up to 32,768

- Activation Function SwiGLU

- Vocabulary Size 128,000

- Positional Embeddings RoPE (θ = 500,000)

### Training Details


The recipe used to pre-train Llama 3 405B consists of three main stages: (1) initial pre-training, (2) long-context pre-training, and (3) annealing

#### Initial Pre-Training

- AdamW with a peak learning rate of 8×10−5

- a linear warm up of 8,000 steps

- a cosine learning rate schedule decaying to 8×10−7 over 1,200,000 steps

- lower batch size early in training to improve training stability, and increase it subsequently to improve efficiency.

  - an initial batch size of 4M tokens and sequences of length 4,096

  - double these values to a batch size of 8M sequences of 8,192 tokens after pre-training 252M token
  - double the batch size again to 16M after pre-training on 2.87T tokens

#### Adjusting the data mix

- increased the percentage of non-English data during pre-training to improve the multilingual performance of Llama 3

- upsample mathematical data to improve the model’s mathematical reasoning performance

- added more recent web data in the later stages of pre-training to advance the model’s knowledge cut-of

- downsampled subsets of the pre-training data that were later identified as being lower quality

#### Long Context Pre-Training

- Train on long sequences to support context windows of up to 128K tokens

- We increase the supported context length in increments, pre-training until the model has successfully adapted to the increased context length:

  - model performance on short-context evaluations has recovered completely

  - the model perfectly solves “needle in a haystack” tasks up to that length

In Llama 3 405B pre-training, we increased context length gradually in six stages, starting from the original 8K context window and ending in the final 128K context window. This long-context pre-training stage was performed using approximately 800B training tokens.


#### Annealing

During pre-training on the final 40M tokens, we linearly annealed the learning rate to 0, maintaining a context length of 128K tokens. During this annealing phase, we also adjusted the data mix to upsample data sources of very high quality; see Section 3.1.3. Finally, we compute the average of model checkpoints (Polyak (1991) averaging) during annealing to produce the final pre-trained model

Annealing Data: Empirically, we find that annealing (see Section 3.4.3) on small amounts of high-quality code and mathematical
data can boost the performance of pre-trained models on key benchmark. Akin to Li et al. (2024b), we perform annealing with a data mix that upsamples high-quality data in select domains. 

We find that annealing improved the performance of a pre-trained Llama 3 8B model on the GSM8k and MATH validation sets by 24.0% and 6.4%, respectively. 

However, the improvements on the 405B model are negligible, suggesting that our flagship model has strong in-context learning and reasoning capabilities and does not require specific in-domain training samples to obtain strong performance


#### Scaling law experiments:

- compute budgets between 6 × 1018 FLOPs and 1022 FLOPs. 

- At each compute budget, we pre-train models ranging in size between 40M and 16B parameters, using a subset of model sizes at each compute budget. 

In these training runs:

- cosine learning rate schedule with a linear warmup for 2,000 training steps. 

- The peak learning rate is set between 2 × 10−4 and 4 × 10−4 depending on the size of the model. 

- We set the cosine decay to 0.1 of the peak value. 

- The weight decay at each step is set to 0.1 times the learning rate at that step.

- We use a fixed batch size for each compute scale, ranging between 250K and 4M.


## QWEN

### Architecture

We have adopted the recent open-source approach of training large language models, LLaMA (Touvron et al.,2023a), which is widely regarded as the top open-source LLM.

- Embedding and output projection : Based on preliminary experimental findings, we have opted for the untied embedding approach instead of tying the weights of input embedding and output projection. This decision was made in order to achieve better performance with the price of memory costs.

- Positional embedding : We have chosen RoPE (Rotary Positional Embedding) (Su et al., 2021) as our preferred option for incorporating positional information into our model. RoPE has been widely adopted and has demonstrated success in contemporary large language models, notably PaLM (Chowdhery et al., 2022; Anil et al., 2023) and LLaMA (Touvron et al., 2023a;b). In particular, we have opted to use FP32 precision for the inverse frequency matrix, rather than BF16 or FP16, in order to prioritize model performance and achieve higher accuracy.

- Bias : For most layers, we remove biases following Chowdhery et al. (2022), but we add biases in the QKV layer of attention to enhance the extrapolation ability of the model (Su, 2023b).

- Pre-Norm & RMSNorm : In modern Transformer models, pre-normalization is the most widely used approach, which has been shown to improve training stability compared to post-normalization. Recent research has suggested alternative methods for better training stability, which we plan to explore in future versions of our model. Additionally, we have replaced the traditional layer normalization technique described in (Ba et al., 2016) with RMSNorm (Jiang et al., 2023). This change has resulted in equivalent performance while also improving efficiency.

- Activation function : We have selected SwiGLU (Shazeer, 2020) as our activation function, a combination of Swish (Ramachandran et al., 2017) and Gated Linear Unit (Dauphin et al., 2017). Our initial experiments have shown that activation functions based on GLU generally outperform other baseline options, such as GeLU (Hendrycks & Gimpel, 2016). As is common practice in previous research, we have reduced the dimension of the feed-forward network (FFN) from 4 times the hidden size to 8/3 of the hidden size.

### Training Details

- To create batches of data, we shuffle and merge the documents, and then truncate them to the specified context lengths

- Context lengths of 2048

- Flash Attention

- AdamW optimizer β1 = 0.9, β2 = 0.95, and ϵ = 10−8. 

- cosine learning rate schedule with a specified peak learning rate for each model size. The learning rate is decayed to a minimum learning rate of 10% of the peak learning rate. 

- All the models are trained with BFloat16 mixed precision for training stability.


#### Context Length Extension

Training-free techniques that are solely applied during inference to extend the context length of the model. One of the key techniques we have used is NTK-aware interpolation (bloc97, 2023). 

Unlike position interpolation (PI) (Chen et al., 2023a) which scales each dimension of RoPE equally, NTK-aware interpolation adjusts the base of RoPE to prevent the loss of high-frequency information in a training-free manner.

To further improve performance, we have also implemented a trivial extension called dynamic NTK-aware interpolation, which is later formally discussed in (Peng et al., 2023a). It dynamically changes the scale by chunks, avoiding severe performance degradation. These techniques allow us to effectively extend the context length of Transformer models without compromising their computational efficiency or accuracy.

QWEN additionally incorporates two attention mechanisms: LogN-Scaling (Chiang & Cholak, 2022; Su, 2023a) and window attention (Beltagy et al., 2020). LogN-Scaling rescales the dot product of the query and value by a factor that depends on the ratio of the context length to the training length, ensuring that the entropy of the attention value remains stable as the context length grows. Window attention restricts the attention to a limited context window, preventing the model from attending to tokens that are too far away.

We also observed that the long-context modeling ability of our model varies across layers, with lower layers being more sensitive in context length extension compared to the higher layers. To leverage this observation, we assign different window sizes to each layer, using shorter windows for lower layers and longer windows for higher layers.


## QWEN 2

### Architecture

| Configuration         | 0.5B   | 1.5B   | 7B     | 72B    | 57B-A14B |
|-----------------------|--------|--------|--------|--------|----------|
| Hidden Size           | 896    | 1,536  | 3,584  | 8,192  | 3,584    |
| # Layers              | 24     | 28     | 28     | 80     | 28       |
| # Query Heads         | 14     | 12     | 28     | 64     | 28       |
| # KV Heads            | 2      | 2      | 4      | 8      | 4        |
| Head Size             | 64     | 128    | 128    | 128    | 128      |
| Intermediate Size     | 4,864  | 8,960  | 18,944 | 29,568 | 2,560    |
| # Routed Experts      | -      | -      | -      | -      | 64       |
| # Activated Experts   | -      | -      | -      | -      | 8        |
| # Shared Experts      | -      | -      | -      | -      | 8        |
| Embedding Tying       | True   | True   | False  | False  | False    |
| Vocabulary Size       | 151646 | 151646 | 151646 | 151646 | 151646   |
| # Trained Tokens      | 12T    | 7T     | 7T     | 7T     | 4.5T     |

- Grouped Query Attention : We adopt Grouped Query Attention (GQA, Ainslie et al., 2023) instead of conventional multi-head attention (MHA). GQA optimizes KV cache usage during inference, significantly enhancing throughput. Detailed KV head configurations for various model sizes are reported in Section 2.2.3.

- Dual Chunk Attention with YARN : To expand the context window of Qwen2, we implement Dual Chunk Attention (DCA, An et al., 2024), which segments long sequences into chunks of manageable lengths. If the input can be handled in a chunk, DCA produces the same result as the original attention. Otherwise, DCA facilitates effective capture of relative positional information between tokens within and across chunks, thereby improving long context performance. Moreover, we also employ YARN (Peng et al., 2023) to rescale the attention weights for better length extrapolation.

Moreover, we follow Qwen with the usage of SwiGLU (Dauphin et al., 2017) for activation, Rotary Positional Embeddings (RoPE, Su et al., 2024) for positional embedding, QKV bias (Su, 2023) for attention, RMSNorm (Jiang et al., 2023b) and pre-normalization for training stability.


### Training Details

#### Long Context Training

To enhance the long-context capability of Qwen2, we augmented the context length from 4,096 tokens to 32,768 tokens during the concluding phase of pre-training. This expansion was complemented by the introduction of a significantly increased volume of high-quality, lengthy data. In conjunction with these enhancements, we modified the base frequency of RoPE from 10,000 to 1,000 000 to optimize performance in long-context scenarios (Xiong et al., 2023).

To fully leverage the model’s length extrapolation potential, we adopted the YARN mechanism (Peng et al., 2023) and the Dual Chunk Attention mechanism (An et al., 2024). These strategies enable the model to process sequences of up to 131,072 tokens while maintaining high performance, as evidenced by minimal perplexity degradation in preliminary experiments.


## DeepSeek LLM

### Architecture

| Params | #layers | dmodel | #heads | #kv_heads | SeqLength | Batch | LR     | Tokens |
|--------|---------|--------|--------|-----------|-----------|-------|--------|--------|
| 7B     | 30      | 4096   | 32     | 32        | 4096      | 2304  | 4.2e-4 | 2.0T   |
| 67B    | 95      | 8192   | 64     | 8         | 4096      | 4608  | 3.2e-4 | 2.0T   |


The micro design of DeepSeek LLM largely follows the design of LLaMA (Touvron et al., 2023a,b), adopting a Pre-Norm structure with RMSNorm (Zhang and Sennrich, 2019) function and using SwiGLU (Shazeer, 2020) as the activation function for the Feed-Forward Network (FFN), with an intermediate layer dimension of 8/3 d model. It also incorporates Rotary Embedding (Su et al., 2024) for positional encoding. To optimize inference cost, the 67B model uses Grouped-Query Attention (GQA) (Ainslie et al., 2023) instead of the traditional Multi-Head Attention (MHA).

However, in terms of macro design, DeepSeek LLM differs slightly. Specifically, DeepSeek LLM 7B is a 30-layer network, while DeepSeek LLM 67B has 95 layers. These layer adjustments, while maintaining parameter consistency with other open-source models, also facilitate model pipeline partitioning to optimize training and inference.

Unlike most works using Grouped-Query Attention (GQA), we expanded the 67B model’s parameters in network depth rather than the common practice of widening the intermediate width of FFN layers, aiming for better performance. Detailed network specifications can be found in Table 2.

### Training Details

- initialized with a standard deviation of 0.006

- AdamW optimizer, with the following hyperparameters: 𝛽1 = 0.9, 𝛽2 = 0.95, and weight_decay = 0.1.

- The gradient clipping 1.0 

- A multi-step learning rate scheduler is employed during pre-training instead of the typical cosine scheduler. Specifically, the learning rate of the model reaches its maximum value after 2000 warmup steps, and then decreases to 31.6% of the maximum value after processing 80% of the training tokens. It further reduces to 10% of the maximum value after 90% of the tokens.

#### Scaling Laws with Different Data

An interesting observation from the analysis is that the optimal model/data scaling-up allo-
cation strategy across these three datasets showed consistency with data quality. As illustrated
in Table 4, as data quality improves, the model scaling exponent 𝑎 gradually increases, while the
data scaling exponent 𝑏 decreases, which suggests that the increased compute budget should be
allocated more to the model instead of the data. This finding might also explain the significant
differences in optimal model/data scaling-up allocation observed in earlier studies of scaling
laws.

## DeepSeek-V2

### Architecture

For attention, we design MLA, which utilizes low-rank key-value joint compression to eliminate the bottleneck of inference-time key-value cache, thus supporting efficient inference. For FFNs, we adopt the DeepSeekMoE architecture (Dai et al., 2024), a high-performance MoE architecture that enables training strong models at an economical cos

- Multi-Head Latent Attention: Boosting Inference Efficiency

- DeepSeekMoE: Training Strong Models at Economical Costs

Model:
- All learnable parameters are randomly initialized with a standard deviation of 0.006

- We set the number of Transformer layers to 60 and the hidden dimension to 5120

- In MLA, we set the number of attention heads to 128 and the per-head dimension to 128. The KV compression dimension is set to 512, and the query compression dimension is set to 1536. For the decoupled queries and key, we set the per-head dimension to 64.

- Following Dai et al. (2024), we substitute all FFNs except for the first layer with MoE layers.

  - Each MoE layer consists of 2 shared experts and 160 routed experts, where the intermediate hidden dimension of each expert is 1536. Among the routed experts, 6 experts will be activated for each token.

  -  we employ additional RMS Norm layers after the compressed latent vectors, and multiply additional scaling factors at the width bottlenecks

- Under this configuration, DeepSeek-V2 comprises 236B total parameters, of which 21B are activated for each token.

### Training Details

- AdamW optimizer 𝛽1 = 0.9, 𝛽2 = 0.95, and weight_decay = 0.1. 

- The learning rate is scheduled using a warmup-and-step-decay strategy (DeepSeek-AI, 2024). Initially, the learning rate linearly increases from 0 to the maximum value during the first 2K steps. Subsequently, the learning rate is multiplied by 0.316 after training about 60% of tokens, and again by 0.316 after training about 90% of tokens. 

- The maximum learning rate is set to 2.4 × 10−4

- gradient clipping norm is set to 1.0.

- batch size is gradually increased from 2304 to 9216 in the training of the first 225B tokens, and then keeps 9216 in the remaining training. 

- We set the maximum sequence length to 4K, and train DeepSeek-V2 on 8.1T tokens.

- We leverage pipeline parallelism to deploy different layers of a model on different devices, and for each layer, the routed experts will be uniformly deployed on 8 devices (𝐷 = 8). As for the device-limited routing, each token will be sent to at most 3 devices (𝑀 = 3). As for balance losses, we set 𝛼1 to 0.003, 𝛼2 to 0.05, and 𝛼3 to 0.02. We employ the token-dropping strategy during training for acceleration, but do not drop any tokens for evaluation.

#### Long Context Extension


After the initial pre-training of DeepSeek-V2, we employ YaRN (Peng et al., 2023) to extend the default context window length from 4K to 128K.

YaRN was specifically applied to the decoupled shared key k𝑅 𝑡 as it is responsible for carrying RoPE (Su et al., 2024). For YaRN, we set the scale 𝑠 to 40, 𝛼 to 1, 𝛽 to 32, and the target maximum context length to 160K. Under these settings, we can expect the model to respond well for a context length of 128K. Slightly diverging from original YaRN, due to our distinct attention mechanism, we adjust the length scaling factor to modulate the attention entropy. The factor √𝑡 is computed as √𝑡 = 0.0707 ln 𝑠 + 1, aiming at minimizing the perplexity.

We additionally train the model for 1000 steps, with a sequence length of 32K and a batch size of 576 sequences. Although the training is conducted solely at the sequence length of 32K, the model still demonstrates robust performance when being evaluated at a context length of 128K.


## MobileLLM

### Architecture

| Model             | #Layers | #Params | ARC-e | ARC-c | BoolQ | PIQA | SIQA | HellaSwag | OBQA | WinoGrande | Avg. |
|-------------------|---------|---------|-------|-------|-------|------|------|-----------|------|------------|------|
| Cerebras-GPT-111M | 10      | 111M    | 35.8  | 20.2  | 62.0  | 58.0 | 39.8 | 26.7      | 29.0 | 48.8       | 40.0 |
| LaMini-GPT-124M   | 12      | 124M    | 43.6  | 26.0  | 51.8  | 62.7 | 42.1 | 30.2      | 29.6 | 49.2       | 41.9 |
| Galactica-125M    | 12      | 125M    | 44.0  | 26.2  | 54.9  | 55.4 | 38.9 | 29.6      | 28.2 | 49.6       | 40.9 |
| OPT-125M          | 12      | 125M    | 41.3  | 25.2  | 57.5  | 62.0 | 41.9 | 31.1      | 31.2 | 50.8       | 42.6 |
| GPT-neo-125M      | 12      | 125M    | 40.7  | 24.8  | 61.3  | 62.5 | 41.9 | 29.7      | 31.6 | 50.7       | 42.9 |
| Pythia-160M       | 12      | 162M    | 40.0  | 25.3  | 59.5  | 62.0 | 41.5 | 29.9      | 31.2 | 50.9       | 42.5 |
| RWKV-169M         | 12      | 169M    | 42.5  | 25.3  | 59.1  | 63.9 | 40.7 | 31.9      | 33.8 | 51.5       | 43.6 |
| MobileLLM-125M    | 30      | 125M    | 43.9  | 27.1  | 60.2  | 65.3 | 42.4 | 38.9      | 39.5 | 53.1       | 46.3 |
| MobileLLM-LS-125M | 30      | 125M    | 45.8  | 28.7  | 60.4  | 65.7 | 42.9 | 39.5      | 41.1 | 52.1       | 47.0 |
| Cerebras-GPT-256M | 14      | 256M    | 37.9  | 23.2  | 60.3  | 61.4 | 40.6 | 28.3      | 31.8 | 50.5       | 41.8 |
| OPT-350M          | 24      | 331M    | 41.9  | 25.7  | 54.0  | 64.8 | 42.6 | 36.2      | 33.3 | 52.4       | 43.9 |
| Pythia-410M       | 24      | 405M    | 47.1  | 30.3  | 55.3  | 67.2 | 43.1 | 40.1      | 36.2 | 53.4       | 46.6 |
| RWKV-430M         | 24      | 430M    | 48.9  | 32.0  | 53.4  | 68.1 | 43.6 | 40.6      | 37.8 | 51.6       | 47.0 |
| BLOOM-560M        | 24      | 559M    | 43.7  | 27.5  | 53.7  | 65.1 | 42.5 | 36.5      | 32.6 | 52.2       | 44.2 |
| Cerebras-GPT-590M | 18      | 590M    | 42.6  | 24.9  | 57.7  | 62.8 | 40.9 | 32.0      | 33.2 | 49.7       | 43.0 |
| MobileLLM-350M    | 32      | 345M    | 53.8  | 33.5  | 62.4  | 68.6 | 44.7 | 49.6      | 40.0 | 57.6       | 51.3 |
| MobileLLM-LS-350M | 32      | 345M    | 54.4  | 32.5  | 62.8  | 69.8 | 44.1 | 50.6      | 45.8 | 57.2       | 52.1 |

Table 9: Detailed architecture specifications of MobileLLM. "Emb Dim" denotes the embedding dimension and "Hidden Dim" represents the dimension inside the feed-forward network.

| Model          | #Layer | #Head | #KV-Head | Emb Dim | Hidden Dim | #Params |
|----------------|--------|-------|----------|---------|------------|---------|
| MobileLLM-125M | 30     | 9     | 3        | 576     | 1536       | 124.6M  |
| MobileLLM-350M | 32     | 15    | 5        | 960     | 2560       | 345.3M  |
| MobileLLM-600M | 40     | 18    | 6        | 1152    | 3072       | 603.1M  |
| MobileLLM-1B   | 54     | 20    | 5        | 1280    | 3584       | 1.0B    |
| MobileLLM-1.5B | 54     | 25    | 5        | 1600    | 4352       | 1.5B    |

- Contradictory to the scaling law (Kaplan et al., 2020), we demonstrate that depth is more important than width for small LLMs. A deep-and-thin model structure excels in capturing abstract concepts, resulting in superior final performance.

- We revisit embedding sharing methods (Zhang et al., 2022) and implement grouped query attention (Ainslie et al., 2023) in small LLMs to maximize weight utilization.

- We propose immediate block-wise weight sharing. In scenarios where memory movement is the latency bottleneck, weight sharing between two adjacent blocks avoids weight movement, requiring only computing the block twice and incurring minimal latency overhead.

Improving sub-billion:

- Adopting SwiGLU FFN (Dauphin et al., 2017);

- forcing lanky (deep and thin) architectures

- revisiting embedding sharing method (Zhang et al., 2022)

- utilizing grouped query attention (Chowdhery et al., 2023)

- an immediate block-wise layer-sharing method

### Training Details

- Models trained 480k iterations on 1T tokens.

- Adam optimizer (Kingma & Ba, 2014) with a weight decay of 0.1

- Initial learning rate is set to 2e-3 and follows a cosine learning-rate decay strategy

- Batch size 32 GPU, 32 batch, total batch size 1024

## SmolLM

### Architecture

| #param | #layers | #head | #kv-head | #d_emb | #h_dim | lr   | batch |
|--------|---------|-------|----------|--------|--------|------|-------|
| 135M   | 30      | 9     | 3        | 576    | 1536   | 3e-3 | 1M    |
| 362M   | 32      | 15    | 5        | 960    | 2560   | 3e-3 | 1M    |
| 1.71B  | 24      | 32    | 32       | 2048   | 8192   | 5e-4 | 2M    |


Incorporating Grouped-Query Attention (GQA) and prioritizing depth over width, similar to MobileLLM.

The 1.7B parameter model uses a more traditional architecture. For all three models we use embedding tying and a context length of 2048 tokens. 

This context length can be further extended with some long context fine-tuning.

We used a tokenizer trained on the Smollm Corpus with a vocab size of 49152. 

### Training Details

- 135M and 360M models, each trained on 600B tokens from Smollm-Corpus

- 1.7B model, trained on 1T tokens from Smollm-Corpus

- trapezoidal learning rate scheduler with a cooldown phase equal to 20% of the total training time.

