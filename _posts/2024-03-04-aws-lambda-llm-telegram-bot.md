---
layout: single
title:  "Deploying LLM Powered Telegram Bots to AWS Lambda"
classes: wide
categories:
  - analysis

tags:
  - python
  - llm
  - aws
  - lambda
  - OpenAI
  - Quantized
  - Gpt4All
---

During a conversation in a Telegram group, one of my friends used an unpleasant word. Although I didn’t appreciate the choice of language, I also didn’t want to be that guy. It made me wish our group had a moderator to intervene. This thought led me to explore LLM (Large Language Model) powered Telegram bots.

As pricing and model capabilities change over time this analysis is only relevant for the being, as of early 2024.

## Introduction

Lambda functions work in response to an event e.g. a chat message. AWS provides a free tier for lambda functions which seems to be enough for a personal bot use case. Those attributes make AWS lambda an interesting option for deploying an LLM-powered telegram bot. However, AWS Lambda provides a constrained execution environment. It is only possible to allocate 10GB of memory and 10GB of disk space to a function.

This post will not be based on any specific use case. Instead, the general applicability of LLM-based telegram bots to AWS Lambda is investigated. Two options are taken as worthy of exploration:

- Self-contained implementation where a quantized LLM is loaded within the AWS Function. The contained LLM is used to reply the incoming messages.
- Remote API-based implementation where a proprietary LLM served over an API is utilized. The API is used to reply the incoming messages. 

To keep things simple, the lambda function does not keep chat history. Since LLM interaction only involves the last message, the bot is not a real chatbot. 

## Self-Contained Implementation

Large Language Models (LLMs) are notorious for their memory, disk space, and GPU requirements. However, thanks to quantization techniques, it’s now possible to reduce the size of an LLM with minimal loss of quality. Gpt4All is an interesting tool that provides quantized language models that can be used in CPU-only environments.

For instance, consider the quantized `mistral-7b-instruct` model file, which is 3.8GB. In contrast, the regular `mistral-7b-instruct` model file weighs approximately 15GB (without delving into the details of the significant size difference). The quantized version of `mistral-7b-instruct` can fit within an AWS Lambda function.

During testing, 5.5GB of memory was sufficient to load the model and generate text using 100 input tokens, resulting in 100 output tokens. However, Telegram imposes a 60-second request limit, leaving the Lambda function with approximately 50 seconds to generate a response (accounting for any loading time). While using two Lambda functions could mitigate this limitation, this approach is NOT explored. Therefore, 100 tokens represent the practical maximum for this configuration. Generating more tokens may not yield a significant difference.

As memory and execution times increase, Lambda functions become costlier and the free tier resources deplete relatively quickly. With this setup, you can respond to 666 messages per month using free tier resources. Handling 5000 messages (equivalent to processing 1M tokens per month) costs approximately $20. In contrast, processing 1M tokens with GPT-3.5-Turbo costs only $1.

This configuration works well for short prompts and brief output messages until the AWS Lambda free tier resources are exhausted.

One important caveat: I do not know of any language models of this size that perform well in Turkish. Consequently, this deployment option may not provide an optimal Turkish language experience.


## Proprietary LLM API-Based Impl

Utilizing a language model over an API is more suitable for resource-constrained lambda environments. The Langchain library allows easy switching between different API providers. This study utilizes the OpenAI GPT-3.5-Turbo model. 

GPT-3.5-Turbo model is an efficient and capable option. Assuming it takes 5 seconds to generate a response, with a 1GB configuration, which is probably too much, It is possible to process 80K requests per month within the AWS lambda free tier. For 200 tokens per request, it costs only $1.6 per month.

This approach scales much better and supports the Turkish language. 

## Conclusion

Self-contained implementation is feasible and can be suitable for various use cases. However, an API-based implementation offers greater scalability and capability. Moreover, the API-based approach becomes more cost-efficient, particularly after the AWS Lambda free tier resources are exhausted.

You can find detailed open-source implementations of both options in my GitHub repository. 

## References
1. [Gpt4All](https://gpt4all.io/index.html)
2. [Mistral 7B Instruct](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)
3. [Contained Deployment](https://github.com/habanoz/telegram-bot-gpt4all-lambda)
4. [API Deployment](https://github.com/habanoz/telegram-bot-llm-openai-lambda)