---
layout: single
title:  "Block Oriented Web Page Crawler for Semantic Search"

categories:
  - analysis

tags:
  - crawling
  - llm
  - langchain

---

## Introduction

Langchain offers WebBaseLoader to load web pages and scrap text from them. 
This is useful for semantic search, as you can build and store text embeddings in a vector database.
Then the vector db is used in conjunction with an LLM to find answer to web page relevant questions in a conversational manner.    

This article tries to find a crawler that is more suitable for semantic search.

### Objectives

Web pages contain a lot of redundat text. 
Pages commonly shares a similar structure which result in repetetion of text.

A better crawler should eliminate redundant text. 
This would enhance the question answering by reducing the embedding generation and LLM querying costs without sacrificing utility.

### Problems with Classical Approach

The classical approach combines text content from a webpage. 
It produces a long block of text. 
A long block of text is then can/should be split into smaller chunks using text splitters.

Splitting a text into smaller segments may result in losing some of the context, as the splitting algorithm may not be able to determine the relevance of different parts of the text.

One solution which is used commonly to mitigate adverse effect of splitting is overlapping chunks. 
Some of the text from the previos chunk is used in later chunk. 
However, overlap size is not easy to tune. 





## References
1. [Deep Learning AI Course](https://learn.deeplearning.ai/chatgpt-prompt-eng/lesson/1/introduction)
