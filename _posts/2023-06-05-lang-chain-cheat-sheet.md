---
layout: single
title:  "Langchain Cheat Sheet"
classes: wide

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

### ConversationBufferMemory

```python
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

### Memory with constraints

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

## Chains

### LLMChain

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

llm = ChatOpenAI(temperature=0.9)
prompt = ChatPromptTemplate.from_template("<>{input}<>")

chain = LLMChain(llm=llm, prompt=prompt)
input_text = "<>"
chain.run(input_text)

```

### SimpleSequentialChain

Each chain has 1 input and 1 output.

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import SimpleSequentialChain

llm = ChatOpenAI(temperature=0.9)

# chain 1
prompt1 = ChatPromptTemplate.from_template("<>{input}<>")
chain1 = LLMChain(llm=llm, prompt=prompt1)

# chain 2
prompt2 = ChatPromptTemplate.from_template("<>{input2}<>")
chain2 = LLMChain(llm=llm, prompt=prompt2)

chain = SimpleSequentialChain(chains=chain1, chain2],verbose=True)

input_text = "<>"
chain.run(input_text)
```

### SequentialChain

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import SequentialChain

llm = ChatOpenAI(temperature=0.9)

prompt1 = ChatPromptTemplate.from_template("<>{intput_text}<>")
chain1 = LLMChain(llm=llm, prompt=prompt1, output_key="output_text1")

prompt2 = ChatPromptTemplate.from_template("<>{output_text1}<>")
chain2 = LLMChain(llm=llm, prompt=prompt2, output_key="output_text2")

prompt3 = ChatPromptTemplate.from_template("<>{intput_text}<>")
chain3 = LLMChain(llm=llm, prompt=prompt3, output_key="output_text3")

prompt4 = ChatPromptTemplate.from_template("<>{output_text2}<>{output_text3}<>")
chain4 = LLMChain(llm=llm, prompt=prompt4, output_key="output_text4")

chain = SequentialChain(
    chains=[chain1, chain2, chain3, chain4],
    input_variables=["intput_text"],
    output_variables=["output_text1", "output_text2","output_text4"],
    verbose=True
)

input_text = "<>"
chain(input_text)

```

### Router Chain

```python
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain,RouterOutputParser
from langchain.prompts import PromptTemplate

llm = ChatOpenAI(temperature=0)

template_text1 = "<>{input}<>"
template_text2 = "<>{input}<>"
template_text3 = "<>{input}<>"

prompt1 = ChatPromptTemplate.from_template(template=template_text1)
chain1 = LLMChain(llm=llm, prompt=prompt1)

chains = {'domain1':chain1, 'domain2':chain2, 'domain3':chain3}
domains_str = "\n".join(['domain1: Good for domain1', 'domain2: Good for domain2', 'domain3: Good for domain3'])

prompt_default = ChatPromptTemplate.from_template("{input}")
chain_default = LLMChain(llm=llm, prompt=prompt_default)s

router_tamplate_text_0 = """
<>

<< FORMATTING >>
Return a markdown code snippet with a JSON object formatted to look like:
```json
{{{{
    "destination": string \ name of the prompt to use or "DEFAULT"
    "next_inputs": string \ a potentially modified version of the original input
}}}}

REMEMBER: "destination" MUST be one of the candidate prompt \
names specified below OR it can be "DEFAULT" if the input is not\
well suited for any of the candidate prompts.
REMEMBER: "next_inputs" can just be the original input \
if you don't think any modifications are needed.

<< CANDIDATE PROMPTS >>
{domains}

<< INPUT >>
{{input}}

<< OUTPUT (remember to include the ```json)>>
"""

router_tamplate_text = router_tamplate_text_0.format(
    domains=domains_str
)

router_prompt = PromptTemplate(
    template=router_tamplate_text,
    input_variables=["input"],
    output_parser=RouterOutputParser(),
)

router_chain = LLMRouterChain.from_llm(llm, router_prompt)

chain = MultiPromptChain(router_chain=router_chain, destination_chains=chains, default_chain=chain_default, verbose=True)

chain.run("<>")

```

## Indexes

### One Line Index Creation

```python
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.indexes import VectorstoreIndexCreator

loader = ## e.g. CSVLoader(file_path=file)

index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch
    # embedding=embeddings,
).from_loaders([loader])

query ="<>"
response = index.query(query)

```

## Walkthrough

```python
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA

loader = ## e.g. CSVLoader(file_path=file)
docs = loader.load()

embeddings = OpenAIEmbeddings()
db = DocArrayInMemorySearch.from_documents(
    docs, 
    embeddings
)

query = "<>"
llm = ChatOpenAI(temperature = 0.0)

# combine documents
selected_docs = db.similarity_search(query)
combined_docs = "".join([docs[i].page_content for i in range(len(selected_docs))])

response = llm.call_as_llm(f"{combined_docs} Question: <>")

# or use a chain
retriever = db.as_retriever()
qa_stuff = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever, 
    verbose=True
)

response = qa_stuff.run(query)

```

## References
1. first reference