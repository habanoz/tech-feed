---
layout: single
title:  "Langchain Cheat Sheet"
categories:
  - cheatsheet

tags:
  - langchain
  - python
  - llm

toc: true
toc_label: "My Table of Contents"
toc_icon: "cog"
---

## Models

```python
from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(temperature=0.0)


```

## Prompts

```python

from langchain.prompts import ChatPromptTemplate

template_text = "<> {style} <> text: {text} <>"
prompt_template = ChatPromptTemplate.from_template(template_text)

style_input = "<>"
text_input = "<>"

messages = prompt_template.format_messages (style=style_input, text=text_input)

response = llm(customer_messages)
print(response.content)

```

## Parsers

Output formatting without parser:

```python
from langchain.prompts import ChatPromptTemplate

template_text = "<> field1:<> field2:<> field3:<> <> text: {text}<>"
prompt_template = ChatPromptTemplate.from_template(template_text)

text_input = "<>"
messages = prompt_template.format_messages(text=text_input)

response = llm(messages)
print(response.content)
```

Corresponding code with output parser:

```python
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

field1_schema = ResponseSchema(name="field1", description="desc1")
field2_schema = ResponseSchema(name="field2", description="desc2")
field3_schema = ResponseSchema(name="field3", description="desc3")
response_schemas = [field1_schema, field2_schema, field3_schema]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

template_text = "<> field1:<> field2:<> field3:<> <> text: {text}<> {format_instructions} <>"
prompt = ChatPromptTemplate.from_template(template=template_text)

text_input = "<>"
messages = prompt.format_messages(text=text_input format_instructions=format_instructions)

response = llm(messages)
output_dict = output_parser.parse(response.content)

print(output_dict.get('field1'))
print(output_dict.get('field2'))
print(output_dict.get('field3'))
```


## Memory

#### ConversationBufferMemory

```Python
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

llm = ChatOpenAI(temperature=0.0)
memory = ConversationBufferMemory()
chain = ConversationChain(
    llm=llm, 
    memory = memory,
    verbose=True
)

chain.predict(input="prompt 1")
chain.predict(input="prompt 2")

print(memory.buffer)
memory.load_memory_variables({})  # contains prompt1,2 and ai responses

memory.save_context({"input": "prompt3"}, {"output": "ai response3"})
memory.load_memory_variables({}) # contains prompt1,2,3 and ai responses
```

#### ConversationBufferWindowMemory, ConversationTokenBufferMemory and ConversationSummaryBufferMemory

```python
from langchain.memory import ConversationBufferWindowMemory

window_size = 1
memory = ConversationBufferWindowMemory(k=window_size)

memory.save_context({"input": "prompt1"}, {"output": "ai response1"})
memory.save_context({"input": "prompt2"}, {"output": "ai response2"})

memory.load_memory_variables({}) # windows size is 1: keeps just last conversation (prompt2 and response 2)

# token limit
memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=30)
memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100)

```

## References
1. [Test Code](https://github.com/habanoz/java-go-rest-app-compare)
2. [Spring Native](https://docs.spring.io/spring-boot/docs/current/reference/html/native-image.html)
3. [Micronaut Graalvm Application](https://guides.micronaut.io/latest/micronaut-creating-first-graal-app-gradle-java.html)
4. [K6 Tool](https://k6.io/open-source/)