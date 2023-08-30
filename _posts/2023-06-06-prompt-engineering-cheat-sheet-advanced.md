---
layout: single
title:  "Advanced Prompt Engineering Cheat Sheet"
classes: wide

categories:
  - cheatsheet

tags:
  - prompt engineering
  - llm

toc: true
toc_label: "My Table of Contents"
toc_icon: "cog"
---

# Techinques


## TELeR: A General Taxonomy of LLM Prompts for Benchmarking Complex Tasks

![Teler Figure 1]({{site.baseurl}}/assets/images/teler-figure-1.png)

According to the paper; LLM prompts for complex
tasks are categorized along the following four dimensions.

1- Turn: Based on the number of turns used while prompting a language model in order to perform a complex task, prompts can be either single-turn or multi-turn.
2- Expression: Based on the expression style of the overall task as well as the associated sub-tasks, prompts can be either question-style or instruction-style.
3- Role: Based on whether a proper system role is defined in the LLM system before providing the actual prompt, prompts can have system-role either defined or undefined.
4- Level of Details: Based on whether particular aspects of the goal task definition is present or absent in the directive, prompts are divided into six distinct levels. Here, aspects refer to clear goal definition, sub-task division, explanation seeking, providing few-shot examples, etc. 

By definition, Level “0” means minimal details, i.e., no aspects/no directive; while level “5” means the highest level of details where the directive includes clear goals, distinct sub-tasks/steps, an explicit requirement of explanation/justification, well-defined criteria for evaluation and/or few-shot examples.

### Example Use Case 1 :  Meta-Review Generation

Meta-reviewing is a critical part of the scientific peer-review process and is generally a complex task that involves summarizing expert reviews from multiple reviewers (Shen et al., 2022, 2023)


We also assume that three review-ers have reviewed the manuscript and provided their comments as the data for the meta-review generation task. Suppose that these three reviewer comments are denoted by R1, R2, R3.

- Level 0 Prompt: <R1, R2, R3>

- Level 1 Prompt: Prepare a meta-review by summarizing the following reviewer comments: <R1,R2, R3>

- Level 2 Prompt: Prepare a meta-review by summarizing the following reviewer comments. 
The final output should highlight the core contributions of the manuscript, common strengths/weaknesses mentioned by multiple reviewers, suggestions for improvement, and missing references (if any). The review texts are provided below: <R1, R2, R3>

- Level 3 Prompt: Prepare a meta-review by answering the following questions from the reviewer comments (provided after the questions). 

1. Based on the reviewer’s comments, what are the core contributions made by this manuscript?
2. What are the common strengths of this work, as mentioned by multiple reviewers?
3. What are the common weaknesses of this work, as highlighted by multiple reviewers?
4. What suggestions would you provide for improving this paper?
5. What are the missing references mentioned by the individual reviews?

The review texts are below: <R1, R2, R3>

- Level 4 Prompt: Level 3 Text + Provide justification for your response in detail by explaining why you made the choices you actually made.

- Level 5 Prompt: Level 3 Text + Provide justification for your response in detail by explaining why you made the choices you actually made. A good output should be coherent, highlight major strengths/issues mentioned by multiple reviewers, be less than 400 words in length, and finally, the response should be in English only.


### Example Use Case 2: Narrative Braiding

Narrative braiding, also known as “interweaving” or “multi-perspective storytelling” is a literary technique that involves the parallel telling of multiple storylines that eventually converge and intersect (Bancroft, 2018). 

We also assume two alternative narratives are available that describe the same event as our data for the braiding task, and the goal is to create a final braided narrative. The two alternative narratives are denoted by N1, and N2.

- Level 0: <N1, N2>

- Level 1: Braid a single coherent story from the following alternative narratives: <N1, N2>

- Level 2: Braid a single coherent story from the following alternative narratives. The final narrative should highlight the common information provided by both narratives, interesting unique information provided by each individual narrative, and conflicting information (if any) conveyed in these narratives. The input alternative narratives are provided below: <N1, N2>

- Level 3: Braid a single coherent story from the following alternative narratives provided later by performing the following tasks.
1. Extract overlapping clause pairs from both narratives and paraphrase them.
2. Extract unique clauses from each narrative and identify the interesting ones.
3. Extract conflicting clause pairs conveyed in both narratives and resolve the conflict (if possible).
4. Generate paragraphs from overlapping unique-conflicting clauses and merge them into a single document.
5. Reorder sentences of the merged document into a detailed, coherent story.
6. Summarize the detailed story into a concise braided narrative.
The alternative narratives are below: <N1, N2>
 
 
- Level 4: Level 3 Text + Provide justification for your response in detail by explaining why your response contains certain information and discards other information.

- Level 5: Level 3 Text + Provide justification for your response in detail by explaining why your response contains certain information and discards other information. A good output should be coherent, highlight overlapping-unique-conflicting information provided by individual narratives, be less than 1000 words in length, and finally, the response should be in English only.

### W&B Lecture Example

Taken From [Notebook](https://github.com/wandb/edu/blob/main/llm-apps-course/notebooks/02.%20Generation.ipynb)

A Level 5, complex directive that includes the following:

- Description of high-level goal
- A detailed bulleted list of sub-tasks
- An explicit statement asking LLM to explain its own output
- A guideline on how LLM output will be evaluated
- Few-shot examples

#### System template

```text
You are a creative assistant with the goal to generate a synthetic dataset of Weights & Biases (W&B) user questions.
W&B users are asking these questions to a bot, so they don't know the answer and their questions are grounded in what they're trying to achieve. 
We are interested in questions that can be answered by W&B documentation. 
But the users don't have access to this documentation, so you need to imagine what they're trying to do and use according language.
```

#### Prompt Template

```text
Here are some examples of real user questions, you will be judged by how well you match this distribution.
***
{QUESTIONS}
***
In the next step, you will read a fragment of W&B documentation.
This will serve as inspiration for synthetic user question and the source of the answer. 
Here is the document fragment:
***
{CHUNK}
***
You will now generate a user question and corresponding answer based on the above document. 
First, explain the user context and what problems they might be trying to solve. 
Second, generate user question. 
Third, provide the accurate and concise answer in markdown format to the user question using the documentation. 
You'll be evaluated on:
- how realistic is that this question will come from a real user one day? 
- is this question about W&B? 
- can the question be answered using the W&B document fragment above? 
- how accurate is the answer?
Remember that users have different styles and can be imprecise. You are very good at impersonating them!
Use the following format:
CONTEXT: 
QUESTION: 
ANSWER: 
Let's start!
```

## Chain-of-Thought Prompting

Cot encourages LLM to generate thought steps to explain about its reasoning process before coming to a conclusion about the answer. 

### Few-shot COT Prompting
[Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)

### Zero-shot COT Prompting
[Large Language Models are Zero-Shot Reasoners](https://arxiv.org/abs/2205.11916)

### Automatic Chain-of-Thought (Auto-CoT)
[Automatic Chain of Thought Prompting in Large Language Models](https://arxiv.org/abs/2210.03493)

### Self Consistency
[Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171)
An extension of few-show COT prompting. Sample multiple answers and chose the answer that is most commonly output. Suitable for math and reasoning tasks.

### Tree of Thoughts 
[Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601)
An Extension to Self Consistency. Instead of sampling multiple whole reasoning tracjectory at once, sample multiple reasoning steps and evaluate them. Continue expanding on promising thoughts. An LLM is used to evaluate thoughts. Two tree traversal algorithms can be used: DFS and BFS.

### ReAct Prompting
[ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
Can be thought as an extension to few-shot COT. LLM is prompted to generate a thought and an action based on the reasoning. Then action is taken and result is used as an observation. Those thougt, action, observation steps continue until model is convinced that it knows the result.  

#### Reflexion
[Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/pdf/2303.11366.pdf)
 Reflexion is an approach that resembles to RL. A LLM (evaluator) evaluates actions proposed by the LLM agent(Actor). Then Another LLM (Self-Reflexion) provides verbal feedbacks to improve LLM agent answers in subsequent trials.


## Retrieval Augmented Generation (RAG)

RAG involves retrieving documents about a question and using retrieved documents to answer the question. Documents typically comes from a vector store whichs stores embedding vectors beloning to documents.

### Forward-Looking Active REtrieval augmented generation (FLARE)
[Active Retrieval Augmented Generation](https://arxiv.org/pdf/2305.06983.pdf)

Flare is an extension to RAG. FLARE is based on the obversation that a model ouputs low probabiltiy tokens when it is not sure about the topic. 

In FLARE, model is allowed to generate a partial response. If partial response involves tokens that has low probability, output generation is paused. LLM is asked to generate new questions about the partial output. Answers to new questions are then are added to context. Enriched context is used to generate a more confident answer.


## Recursively Criticize and Improve 
[Language Models can Solve Computer Tasks](https://arxiv.org/pdf/2303.17491.pdf)

In critic step, ask LLM to find problems about its answer. Then improvement step, ask LLM to use the critique to improve its answer. This process can be recursive.

## References
1. Santu, S. K., & Feng, D. (2023). TELeR: A General Taxonomy of LLM Prompts for Benchmarking Complex Tasks. ArXiv. /abs/2305.11430
2. [COT](https://www.promptingguide.ai/techniques/cot)
3. [ Tree of Thoughts: Deliberate Problem Solving with Large Language Models (Full Paper Review) ](https://www.youtube.com/watch?v=ut5kp56wW_4&t=20s)
4. [Understanding ReACT with LangChain](https://www.youtube.com/watch?v=Eug2clsLtFs)
5. [Retrieve as you generate with FLARE](https://python.langchain.com/docs/use_cases/question_answering/how_to/flare)
6. [ GPT 4 Can Improve Itself - (ft. Reflexion, HuggingGPT, Bard Upgrade and much more) ](https://www.youtube.com/watch?v=5SgJKZLBrmg)