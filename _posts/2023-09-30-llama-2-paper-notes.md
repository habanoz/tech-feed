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

![gpu-times]({{site.baseurl}}/assets/images/llama2-table-2.png)

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

Gatt is stated to be inspired from Context Distillation method desribed in LEARNING BY DISTILLING CONTEXT paper. But what is context distillation. 

#### Context distillation 

Context distillation method is used to internalize details from rich detailed instructions into the model so that the model can generate accurate answers to prompts with less details. 

![context-distillation]({{site.baseurl}}/assets/images/llama2-context-distillation-figure1.png)
Image is from LEARNING BY DISTILLING CONTEXT paper.

There are two models as it is the classical distillation framework: a teacher and a student. The difference is in classical distillation model weights are different. In context distillation, student and teacher has the same weights. Teacher and student differs in the instructions they are given. 

A raw input is sampled from distribution D. It is used to build teach and student prompts. Teacher prompt has more details like explanation and examples. Student prompt is simpler it only has minimal instructions and raw input. Teacher model is also allowed to generate chain of reasoning tokens which is called a scratchpad in the paper. 

Given the teacher prompt, teacher model generates a response. Actual output is extracted from the response, e.g. scratchpad is stripped. Then student model is fine-tuned using simpler prompts and actual outputs from the teacher. 


#### How Gatt works?

I believe the explanation in the paper is not very good. Thanks to a post from one of the authors, Xavier Martinet, to the huggingface community page, we can have a better understanding of Gatt.

Suppose we have a N turn dialogue: [u1, a1, ..., un, an]. Select an instruction, e.g.  "Act as if you were Napoleon who like playing cricket."(taken from the post). Now concatenate the instruction to all user messages and sample the assistant responses. Now we have a N turn dialogue [u1+inst, a1', ..., un+inst, an']. Note that assistant messages are now denoted with a prime which indicates they are sampled from RLHF model after instruction concataneted to user message. 

In training time, instructions are removed from the user messages. Instead it is prepended as a system message and we have [inst, u1, a1', ..., un, an']. Each sample S in the dialogue is used to fine tune the model. Loss for the token before the sample S is zeroed out. 

For example, assume N is 3, we would have a synthetic dataset like [inst, u1, a1, u2, a2, u3, a3]. We can use this dialgue to fine-tune our model 3 times:
-  Use [inst, u1] to predict a1, only compute loss using a1 and predicted a1.
-  Use [inst, u1, a1, u2] to predict a2, only compute loss using a2 and predicted a2.
-  Use [inst, u1, a1, u2, a2, u3] to predict a3, only compute loss using a3 and predicted a3.

#### Gatt Results

Gatt is applied after RLHF V3 stage. When a new instruction used, one that is not used in training, Gatt is found to be consistent upto 20+ turns. 

Gatt is also applied to a LLAMA 1 model that is fine-tuned to have 4096 token context window (base model only has a token window of 2048). The paper suggests evidence on usability of Gatt as a general mothod for long context attention. (See Appendix A.3.5)


### RLHF Results

#### Model-Based Evaluation

The paper notes that human evaluation is gold standard in model evaluation but also states problems with it. It proposes a model based evaluation approach. 

In model based evaluation, reward improvement is used to select best-performing models among several ablations at each iteration from RLHF-V1 to V5. Then major model versions were validated with human evaluations.

**How Far Can Model-Based Evaluation Go?**: They compared reward model results with human annototators preferences (3 annotators, 7-point Likert scale based annotations ). Figure 29 of the paper suggests that reward model is well aligned with human preferences.

Additionally they trained a new reward model that is trained on diverse open-source Reward Modeling datasets. New reward model is used to ensure that meta reward models do not divert from human preferences.

As a last verification step, they compared resuls of the last model and new model. In next annotation iteration, both models are used to sample from for diversity and comparison on new prompts. 

![reward-model-human-preference-alingment]({{site.baseurl}}/assets/images/llama2-figure-29.png)

**Progression of Models**:   

In figure 11 of the paper, harmlessness and helpfullness of llama-2 chat models compared to ChatGPT is shown. Judgenements are based on meta reward models (which indicates bias) and GPT-4 (ChatGPT and Llama 2-Chat outputs appeared in GPT-4 prompt are randomly swapped to avoid any bias). 

Validation set contains 1586 safety and 584 helpfulness prompts.

#### Human Evaluation

Finally human evaluation is used to judging the models. 4000 single and multi-turn prompts are used. Llama-2-chat models are compared to Falcon, MPT, Vicuna, ChatGPT and PaLM.   

![human-evaluation-helpfulness]({{site.baseurl}}/assets/images/llama2-figure-12.png)

- Llama 2-Chat 70B model is competitive with ChatGPT.
- Llama 2-Chat 70B model outperforms PaLM-bison chat model by a large percentage

**Inter-Rater Reliability (IRR)**: In this sub-section authors notes some limitations of human evaluation. In summary they note objectivity of the result and state that evaluation on a different set of prompts or with different instructions could result in different results.

## Safety

### Safety in Pretraining

One interesting point in Llama2 pretraining is that toxic data is not removed from pretraining dataset for a better downstream generalization. The paper presents how some groups that common source of bias are represented in the pre-training dataset. 

Pre-training dataset mostly contains English data (89.7%). Rest of the data contains code (8.38%) and other languages. 

![evoluation of llama2-chat]({{site.baseurl}}/assets/images/llama2-figure-11.png)

**Safety Benchmarks for Pretrained Models**: Safety evaluation has three dimensions:
- Truthfulness: TruthfulQA is used to measure how well LLM generates factually reliable answers
- Toxicity: Toxigen is used to measure amount of toxic output generation.
- Bias: BOLD is used to measure the sentiment in model generations may vary with demographic attributes.

![llama2-chat-safety-truthfulness]({{site.baseurl}}/assets/images/llama2-table-44.png)
![llama2-chat-safety-toxicity]({{site.baseurl}}/assets/images/llama2-table-45.png)
![llama2-chat-safety-race]({{site.baseurl}}/assets/images/llama2-table-46.png)
![llama2-chat-safety-gender]({{site.baseurl}}/assets/images/llama2-table-47.png)

### Safety Fine-Tuning

#### Safety Categories and Annotation Guidelines

Annotators are instructed to create adverserial prompts along two dimensions:
- A risk category: A topic an LLM an potentialy create unsafe content.  There are tree main risk categories: illicit and criminal activities (e.g., terrorism, theft, human trafficking); hateful and harmful activities (e.g., defamation, self-harm, eating disorders discrimination); and unqualified advice (e.g., medical advice, financial advice, legal advice). 

- An attack vector: A question style to cover different prompts that can produce unsafe output. For example:  psychological manipulatio (e.g., authority manipulation), logic manipulation (e.g., false premises), syntactic manipulation (e.g., misspelling), semantic manipulation (e.g., metaphor), perspective manipulation (e.g., role playing), non-English languages, and others.

Then best practices for safe and helpful model responses are defined:
- The model should first address immediate safety concerns if applicable
- Address the prompt by explaining the potential risks to the user
- Provide additional information if possible. 

Annotator responses cannot do following:
- Promote or enable criminal activities.
- Promote or enable dangerous behaviors to the user or other people.
- Contain, promote or enable offensive and abusive behavior towards the user or other people.
- Contain, promote or enable sexually explicit content. 

#### Safety Supervised Fine-Tuning

Collected annotations are used to fine-tune the models. 

#### Safety RLHF

In early work authors observe that fine-tuned models observed to generate safe outputs. The interesting observation is that when the model outputs safe responses, they are often more detailed than what the average annotator writes, which shows generalization capability of the LLMs.  As a result, they only collected a few thousands supervised annotations.

RLHF is applied for more nuanced responses and harder jail breaking. 

**Better Long-Tail Safety Robustness without Hurting Helpfulness**: Safety is inherently a long-tail problem. 

Two models are trained to compare effect of safety rlhf: one without adversarial prompts in the RLHF stage and one with them. Results clearly show benefit of safety prompts in RLHF. Another observation is that helpfulness score distribution is preserved after safety tuning with RLHF. 

![llama2-safety-rlhf]({{site.baseurl}}/assets/images/llama2-figure-14.png)

Following example shows benefit of RLHF.

![llama2-safety-rlhf]({{site.baseurl}}/assets/images/llama2-table-12.png)

**Impact of Safety Data Scaling**: To better understand how the amount of safety data affects model performance,   authors increased amount of safety data used in RLHF stage, starting from 0 up to ∼0.1M samples, while keeping helpfulness data fixed (∼0.9M samples).

They observe that when amount of safety data is increased, the model’s performance on handling adversarial prompts improves dramatically, with a lighter tail in the safety reward model score distribution. Meanwhile, the mean helpfulness score remains constant. It is hypothesized that this is because there is a sufficiently large amount of helpfulness training data.

![llama2-safet-data-amount]({{site.baseurl}}/assets/images/llama2-figure-15.png)

**Measure of False Refusal**: A model with more safety mitigation answers certain questions in a more conservative manner. A study is conducted to measure false refusal to quantify the frequency that the model incorrectly refuses to answer non-adversarial prompts e.g. false refusal as the model incorrectly refusing to answer legitimate user prompts due to irrelevant safety concerns.

A classifier is trained for detecting refusals and used on responses for two test datasets: helpfulness dataset and a borderline test set consisting of 210 samples.

With more safety data used, false-refusal rates increases. For helpfulness dataset it remains at 0.05%. However, for borderline dataset, which is hard, increases upto 28%.

![false-refusal]({{site.baseurl}}/assets/images/llama2-figure-33.png)


#### Context Distillation for Safety

Authors observed that the safety capabilities of LLMs can be efficiently enhanced by prefixing the model with a safety preprompt (e.g., “You are a safe and responsible assistant”). They use pre-prompts to generate improved safe answers from the models and used the generated data (without pre-prompt) for context distillation to improve safety responses.  

Figure 39 of the paper shows example safety pre-prompts. Those examples can be useful as a system prompt to the llama2-chat models.

![safety-pre-prompts]({{site.baseurl}}/assets/images/llama2-figure-39.png)


**Context Distillation with Answer Templates**: Annotators provided assign risk categories  for the safety prompts. This allows providing answer templates based on identified risk categories. Figure 16 of the paper shows affect of context distillation with and without answer templates. Answer templates seems to improve reward scores further.  

**Rejecting Context Distillation Errors with the Safety Reward Model**:  Performing safety context distillation for helpful prompts can degrade model performance and lead to more false refusals. Therefore, safety context distillation is only applied to adversarial prompts.

If model responses are good, safety context distillation can still cause degredation. Therefore, context distillation is only used on prompts where it improves the results, based on the safety reward model.   

![safety-context-distillation]({{site.baseurl}}/assets/images/llama2-figure-16.png)


### Red Teaming

Red teaming involved 350 people from  various domains (ybersecurity, elec-tion fraud, social media misinformation, legal, policy, civil rights, ethics, software engineering, machine learning, responsible AI, and creative writing.). 

After each red teaming exercise, analysis are performed oncollected data, including dialogue length, risk area distribution, histogram of topic of misin-formation (where appropriate), and rated degree of risk. Lessons are taken as a guide to help model safet training improvements. Probably some of the data from the exercise are added fine-tunning dataset. 


#### Safety Evaluation of Llama 2-Chat

**Safety Human Evaluation**: 2000 adversarial prompts are collected for safety human evaluation. Raters judged the models according to 5-point likert scale (5- no violation and helpful, 1- severe violation). Figure 17 of the paper shows the results.

![safety-measure]({{site.baseurl}}/assets/images/llama2-figure-17.png)

- 1 and 2 are used as violation. 
- Violation percentage is used as the main evaluation metric.
- Mean rating is used as a supplement.
- Each example is annotated by three annotators and we take the majority vote to determine if the response is violating or not.
- Used Gwet’s AC1/2 statistic to measure inter-rater reliability (IRR).
- Multi-turn conversations are more prone to inducing unsafe responses

![violation-percentage]({{site.baseurl}}/assets/images/llama2-figure-18.png)

**Truthfulness, Toxicity, and Bias**: The fine-tuned Llama 2-Chat is far more better than the pretrained Llama 2 in terms of truthfulness (50.18 → 64.14 for 70B) and toxicity (24.60 → 0.01 for 70B). 

![llm-evaluations]({{site.baseurl}}/assets/images/llama2-table-14.png)

- Even with proficient annotators, each write with significant variation.  
- A model fine-tuned on SFT annotations learns this diversity, includingü unfortunately, poor annotations.
- The model’s performance is capped by the writing abilities of the most skilled annotator. 
- Humans are better in comparing two outputs. Humans are not that great at writing annoations.
- During annotation, the model has the potential to explore into writing trajectories that even the
best annotators may not explore. Even then humans can effectively compare them.
- "while we may not all be accomplished artists, our ability to appreciate and critique art remains intact"
- Superior writing abilities of LLMs, as manifested in surpassing human annotators in certain tasks, are fundamentally driven by RLHF.
- Consequently, the reward mechanism swiftly learns to assign low scores to undesirable tail-end
distribution and aligns towards the human preference.
- This phenomena can be seen in Figure 20 of the paper, where it is shown that the worst answers are progressively removed, shifting the distribution to the right.
- Supervised data may no longer be the gold standard, and this evolving circumstance compels a re-evaluation of the concept of "supervision".

## Discussion

### Learnings and Observations

**Beyond Human Supervision**: Authors admit that many of the researhes were skeptic about RL. However, RL turned out to be highly effective considering its cost and time effectiveness. They attribute success of RLHF to synergy it fosters between humans and LLMs
throughout the annotation process.

![reward-distribution-shift]({{site.baseurl}}/assets/images/llama2-figure-20.png)

**In-Context Temperature Rescaling**: Temperature appears to be influenced by RLHF as it is shown in Figure 8 of the paper. Shifts are not similar for all prompt types, as shown in figure 21 of the paper. In creative prompts, RLHF behaves somewhat similar to SFT, as temperature increase diversity increase. But in factual prompts, RLHF prompts despite increased temperature remains to be factual and less diverse. 

![rlhf-temperature-adapt]({{site.baseurl}}/assets/images/llama2-figure-21.png)

**Llama 2-Chat Temporal Perception**: To teach concept of time, 1000 SFT examples related to specific dates are collected e.g. How long How long ago did Barack Obama become president? Each example contained two types of metadata:
- Query date: Response changes according to query date. For example, the answer to “How long ago did Barack Obama become president?” would be different in 2020 compared to 2025.
- Event date:  A point in time prior to which the question would be nonsensical. For example, asking “How long ago did Barack Obama become president?” would not make sense before he actually became president. 

![time-awareness]({{site.baseurl}}/assets/images/llama2-figure-22.png)

This observation suggest that LLMs can have a better understanding of time than it is previously assumed. This is an interesting finding given that LLMs are trained using next token prediction on data that is randomly shuffled without any regard to chronological context. 


**Tool Use Emergence**: As it is shown in Figure 23 of the paper, despite never having been trained to use tools, the model demonstrates the capability to utilize a sequence of tools in a zero-shot context.

### Limitations and Ethical Considerations

Llama 2 chat is subject to limitations of auto-regressive LLMs:
- Knowledge cut-off
- Hallucination
- Potential for non-factual generation such as unqualified advice,

Llama2 is trained primarily on English data. Its abilities in other languages are limited.

Llama2 may generate false refusal of prompts due to overly cautios safety considerations.


## References
1. [LLAMA2 Paper](https://arxiv.org/pdf/2307.09288.pdf)
2. [Gatt by Philipp Schmid](https://twitter.com/_philschmid/status/1692222511612637201)
3. [Community page on Hugging Face](https://huggingface.co/papers/2307.09288)
4. [LEARNING BY DISTILLING CONTEXT](https://arxiv.org/pdf/2209.15189.pdf)
5. [LEARNING BY DISTILLING CONTEXT: a video from first author](https://www.youtube.com/watch?v=IKtAFLUAYvM&t=1756s)