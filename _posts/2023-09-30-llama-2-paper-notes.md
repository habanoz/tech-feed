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

This article includes my notes on Llama 2 paper. All images if not stated oherwise are from the paper.

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

Similar to instructGpt paper, binary ranking loss is used, with addition of a margin component. 

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

*Data Composition*: 

Public human preference data is used to bootstrap reward models. Then they are kept in collected human preference data. Authors do not comment on benefit of public human preference data except for their hopes for better generalization. They only state that they did not observe any negative transfer. 

The Helpfulness reward model is trained on all Meta Helpfulness data, combined with an equal parts of the remaining data uniformly sampled from Meta Safety and from the open-source datasets. 

The Meta Safety reward model is trained on all Meta Safety and Anthropic Harmless data, mixed with Meta Helpfulness and open-source helpfulness data in a 90/10 proportion. 10% helpfulness data is found to be beneficial for the accuracy on samples where both the chosen and rejected responses were deemed safe.

*Training Details*:

- 1 epoch over training data. Training longer may cause overfitting.
- Learning rate $$5X10^{-6}$$ for LLAMA-70B and $$1X10^{-5}$$ for the rest.
- Cosine learning rate scheduler down to 10% percent of initial larning rate.
- Warm-up of 3% of total number of steps, with a maximum of 5.
- Effective batch size is kept fixed at 512 pairs, or 1024 rows per batch.


*Reward Model Results*:

![reward-results]({{site.baseurl}}/assets/images/llama2-table-7-8.png)

Authors note that each model performs better in its own domain, proving competition between the objectives. Also they note that reward model performs worse for similar responses which is not surprising.


*Scaling Trends.*: 

![reward-scaling]({{site.baseurl}}/assets/images/llama2-figure-6.png)

Larger models benefit more from similar amounts of data. Authors note that the scaling performance did not show plateaue given the existing volume of data annotation used for training which indicates that there is room for more improvement with more annotations. 

Authors also note that reward model performance is the most important proxy for chat model performance. While evaluating a chat model is open researh topic there is not debate over reward ranking. Thus any improvement on the reward model means improvement in the chat model. 


#### Iterative Fine-Tuning

Human preference data is collected as batches. As new batches arrive, they are used to train better reward models and collect more prompts. As a result multiple RLHF models are trained: RLHF-v1 to RLHF-v5. Two different algorithm are applied:
- Rejection sampling fine-tunning.
- Proximal Policy Optimization (PPO).

*Rejection Sampling*: 

In Reject Sampling K outputs are sampled from the model given a prompt. Best ouput is selected using the reward model. 

At each iteration, all prompts are used to generate K outputs and best outputs are selected. Then selected outputs from the all previous iterations are used to finetune the model similar to SFT stage.

![rejection-sampling-benefit]({{site.baseurl}}/assets/images/llama2-figure-7.png)

Figure 7 of the paper shows that as K increases maximum score for the generated responses increase. High temperature value helps generating more variations. 

![rejection-sampling-benefit]({{site.baseurl}}/assets/images/llama2-figure-8.png)

Figure 8 of the paper shows the relationship between temparature and maximum reward by number of samples. RLHF models exhibit different temperature behaviour than SFT models.

Rejection sampling is only applied with Llama 2-Chat 70B. All smaller models are fine-tuned on rejection sampled data from the larger model, thus distilling the large-model capabilities into the smaller ones.

*PPO*:

Pretrained language model is regarded as the policy to optimize using the following objective:

$$ \arg \max_{\pi} E_{p \sim D, g \sim \pi} [R(g|p)] $$

where $$p$$ is prompt sampled from dataset $$D$$, $$g$$ is generation from the policy $$\pi$$. Reward function is:

$$ R(g|p) = \tilde{R}_c (g|p)  - \beta D_{KL} (\pi_{\theta}(g|p) ||  \pi_{0}(g|p)) $$

which similar to instructGPT paper includes a penaly term that penalizes divergance from the initial policy.  

Reward function $$R_c$$ is a combination of safety and helpfulness reward scores. 

![ppo-reward]({{site.baseurl}}/assets/images/llama2-ppo-reward.png)

Safety reward score is used if a prompt is known to produce unsafe responses or a response obtains a safety reward less than 0.15. Otherwise helpfulness reward score is used. Reward is transformed by using whiten function (which makes covariance 1) (shown by reversing the sigmoid with the logit function) in order to increase stability and balance properly with the KL penalty term (β) above.

PPO hyperparamers:
- AdamW optimizer with $$\beta_1 = 0.9$$, $$\beta_2 = 0.95$$ and $$eps = 10^{-5}$$.
- Weight decay of 0.1
- Gradient clip of 1.0
- Constant learning rate of $$10^{-6}$$
- PPO batch size of 512
- a PPO clip threshold of 0.2
- a mini-batch size of 64
- take one gradient step per mini-batch.
- For the 7B and 13B models, we set β = 0.01 (KL penalty), and for the 34B and 70B models, we set β = 0.005.

Models are trained between 200 and 400 iterations. Held-out prompts are used for early-stopping. Each PPO iteration for 70B model took average of 330 seconds. They used techniques shortly mentioned in the paper to increase batch size and speed up the process.

### System Message for Multi-Turn Consistency

Initial models tend to ignore system prompt after a few turns. They propose Gatt to solve the issue.

**GAtt Method.**:

Ghost attention techique is not clearly explained in the paper. As a result I observed different interpretations. I will update this section if I come up with a better explanation.

Gatt is stated to be inspired from Context Distillation method desribed in LEARNING BY DISTILLING CONTEXT paper. But what is context distillation. 

#### Context distillation 

Context distillation method is used to internalize details from rich detailed instructions into the model so that the model can generate answer to prompts with less details. 

![context-distillation]({{site.baseurl}}/assets/images/llama2-context-distillation-figure1.png)
Image is from LEARNING BY DISTILLING CONTEXT paper.

There are two models as it is the classical distillation framework: a teacher and a student. The difference is in classical distillation model weights are different. In context distillation, student and teacher has the same weights. Teacher and student differs in the instructions they are given. 

A raw input is sampled from distribution D. It is used to build teach and student prompts. Teacher prompt has more details like explanation and examples. Student prompt is simpler it only has minimal instructions and raw input. Teacher model is also allowed to generate chain of reasoning tokens which is called a scratchpad in the paper. 

Given the teacher prompt, teacher model generates a response. Actual output is extracted from the response, e.g. scratchpad is stripped. Then student model is fine-tuned using simpler prompts and actual outputs from the teacher. 


#### How Gatt works?

The paper does not expose a lot of information about the process. But we can derive following information.

"*Assume we have access to a multi-turn dialogue dataset between two persons (e.g., a user and an assistant), with a list of messages [u1, a1, . . . , un, an], where un and an correspond to the user and assistant messages for turn n, respectively.*"

This part is clear. Basically; we have a dialogue dataset D with multiple turns e.g. n turns. 

"*Then, we define an instruction, inst, that should be respected throughout the dialogue. For example, inst could be “act as.” We can then synthetically concatenate this instruction to all the user messages of the conversation.*"

Now we have a synthetically created new dataset D' such that [u1+inst, a1, . . . , un+inst, an].

"*Next, we can sample from this synthetic data using the latest RLHF model.*"

This section is not clear. How should we sample ? I will assume we should sample next user message from D' using the latest RLHF model using a special prompt that is not disclosed in the paper e.g. given the context-dialogue generate a question that may be asked by the user. Now we have D'' such that [u1+inst, a1, . . . , un+inst, an, u(n+1),]. It is also not clear whether "u(n+1)" includes the instruction. But i expect it to include the instruction because all previous example utterances include it. Also if it is not included there is no reason to expect "a(n+1)" respect the instruction. 

"*We now have a context-dialogue and the sample with which to fine-tune a model, in a process analogous to Rejection Sampling*"

This section mentions that fine-tuning process to apply resembles Rejection Sampling. Remember in Rejection Sampling we sample K completions and select the best wrt. reward model. Now we have dataset D''' such that [u1+inst, a1, . . . , un+inst, an, u(n+1),a(n+1)] 
where a(n+1) is best among K a(n+1) candidates.

"*Instead of augmenting all context-dialogue turns with the instruction, we can drop it in all but the first turn, but this
would lead to a mismatch at training time between the system message, i.e., all the intermediate assistant messages that come before the last turn, and our sample*"

 In training time we want to drop all instructions from user messages from u2 to un. But this would lead to a mismatch between all the intermediate assistant messages that come before the last turn, and our sample. Why? Probably because we sampled u(n+1) using context-dialogue messages e.g. [u1+inst, a1, . . . , un+inst, an]. I cannot see how but this is my interpretation. 

 "* To fix this issue, which could hurt the training, we simply set the loss to 0 for all the tokens from the previous turns, including assistant messages.*"

 In training we drop all instructions except for the first user message and zero out loss for all tokens contained in the context dialogue. Now we have [u1+inst, a1, . . . , un, an, u(n+1),a(n+1)]. And loss is only calculated for last turn messages e.g. u(n+1),a(n+1). Since first user message includes the instruction generated model response has to follow the same instruction (in supervised fine tunning it has to match a(n+1)), it learns to follow instruction regardless of number of dialog turns, which is in essence similar to context distillation. 

 However, we assumed u(n+1) to include the instruction. If that is the case, then a(n+1) has to follow the instruction regardless of u1 having the instruction. So, my interpretation cannot be correct. 

Also note that, we added instructions to a given dialog. Let's say the instruction is act like Sheakspear, but the answer was generated without the instruction, so it does not act like Sheakspear. Is not this a problem? So it is necessary to zero out the loss for context-dialogue tokens without any further reason e.g. the reason mentioned in the paper. Another problem is, all user messages include an instruction that is not followed by the assitance answers then this my encourage RLHF model to generate an answer that ignores the instruction.

I will update this section as new information comes out or my understanding improves.

## References
1. [LLAMA2 Paper](https://arxiv.org/pdf/2307.09288.pdf)
2. [Gatt by Philipp Schmid](https://twitter.com/_philschmid/status/1692222511612637201)
3. [LEARNING BY DISTILLING CONTEXT](https://arxiv.org/pdf/2209.15189.pdf)
4. [LEARNING BY DISTILLING CONTEXT: a video from first author](https://www.youtube.com/watch?v=IKtAFLUAYvM&t=1756s)