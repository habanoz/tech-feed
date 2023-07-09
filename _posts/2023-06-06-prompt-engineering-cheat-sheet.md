---
layout: single
title:  "Prompt Engineering Cheat Sheet"
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

## Principles

### Write clear and specific intructions

#### Use delimiters

- Triple quotes: """
- Triple backticks: ```
- Triple dashes: ---
- Angle brackets: <>
- XML tags: <tag></tag>

Useful to avoid prompt injection.

```python
prompt = f"""
Summarize the text delimited by triple backticks \ 
into a single sentence.
```{text}```
```

#### Ask for structured output

```python
prompt = f"""
Generate a list of three made-up book titles along \ 
with their authors and genres. 
Provide them in JSON format with the following keys: 
book_id, title, author, genre.
"""
```

#### Check whether conditions are satisfied

Check assumptions required to do task.

```python
prompt = f"""
You will be provided with text delimited by triple quotes. 
If it contains a sequence of instructions, \ 
re-write those instructions in the following format:

Step 1 - ...
Step 2 - …
…
Step N - …

If the text does not contain a sequence of instructions, \ 
then simply write \"No steps provided.\"

\"\"\"{text_1}\"\"\"
"""
```
####  Few-shot prompting

Give successful examples of completing task and the ask model to perform the task.
```python
prompt = f"""
Your task is to answer in a consistent style.

<child>: Teach me about patience.

<grandparent>: The river that carves the deepest \ 
valley flows from a modest spring; the \ 
grandest symphony originates from a single note; \ 
the most intricate tapestry begins with a solitary thread.

<child>: Teach me about resilience.
"""
```

  
### Give the model to think.

#### Specify the steps required to complete a task

```python
prompt_1 = f"""
Perform the following actions: 
1 - Summarize the following text delimited by triple \
backticks with 1 sentence.
2 - Translate the summary into French.
3 - List each name in the French summary.
4 - Output a json object that contains the following \
keys: french_summary, num_names.

Separate your answers with line breaks.

Text:
```{text}```
"""

prompt_2 = f"""
Your task is to perform the following actions: 
1 - Summarize the following text delimited by 
  <> with 1 sentence.
2 - Translate the summary into French.
3 - List each name in the French summary.
4 - Output a json object that contains the 
  following keys: french_summary, num_names.

Use the following format:
Text: <text to summarize>
Summary: <summary>
Translation: <summary translation>
Names: <list of names in Italian summary>
Output JSON: <json with summary and num_names>

Text: <{text}>
"""
```

#### Instruct the model to work out its own solution before rushing to a conclusion

```python

prompt = f"""
Determine if the student's solution is correct or not.

Question:
I'm building a solar power installation and I need \
 help working out the financials. 
- Land costs $100 / square foot
- I can buy solar panels for $250 / square foot
- I negotiated a contract for maintenance that will cost \ 
me a flat $100k per year, and an additional $10 / square \
foot
What is the total cost for the first year of operations 
as a function of the number of square feet.

Student's Solution:
Let x be the size of the installation in square feet.
Costs:
1. Land cost: 100x
2. Solar panel cost: 250x
3. Maintenance cost: 100,000 + 100x
Total cost: 100x + 250x + 100,000 + 100x = 450x + 100,000
"""

```

Improve with following prompt:

```python
prompt = f"""
Your task is to determine if the student's solution \
is correct or not.
To solve the problem do the following:
- First, work out your own solution to the problem. 
- Then compare your solution to the student's solution \ 
and evaluate if the student's solution is correct or not. 
Don't decide if the student's solution is correct until 
you have done the problem yourself.

Use the following format:
Question:
```
question here
```
Student's solution:
```
student's solution here
```
Actual solution:
```
steps to work out the solution and your solution here
```
Is the student's solution the same as actual solution \
just calculated:
```
yes or no
```
Student grade:
```
correct or incorrect
```

Question:
```
I'm building a solar power installation and I need help \
working out the financials. 
- Land costs $100 / square foot
- I can buy solar panels for $250 / square foot
- I negotiated a contract for maintenance that will cost \
me a flat $100k per year, and an additional $10 / square \
foot
What is the total cost for the first year of operations \
as a function of the number of square feet.
``` 
Student's solution:
```
Let x be the size of the installation in square feet.
Costs:
1. Land cost: 100x
2. Solar panel cost: 250x
3. Maintenance cost: 100,000 + 100x
Total cost: 100x + 250x + 100,000 + 100x = 450x + 100,000
```
Actual solution:
"""

```

### Model Limitations: Hallucinations

Ask model to first find the relevant information from the text. Then ask it to use those quotes to answer the question. Having a way to trace the answer back to the source is helpful to reduce halluninations.

## Iterations

- Try something
- Analyze the result
- Clarify instructions


```python

prompt = f"""
Your task is to help a marketing team create a 
description for a retail website of a product based 
on a technical fact sheet.

Iteration1==>> Write a product description based on the information 
provided in the technical specifications delimited by 
triple backticks.

Iteration2==>> The description is intended for furniture retailers, 
so should be technical in nature and focus on the 
materials the product is constructed from.

Iteration3==>> At the end of the description, include every 7-character 
Product ID in the technical specification.

Iteration4==>> After the description, include a table that gives the 
product's dimensions. The table should have two columns.
In the first column include the name of the dimension. 
In the second column include the measurements in inches only.

Iteration5==>> Give the table the title 'Product Dimensions'.

Iteration6==>> Format everything as HTML that can be used in a website. 
Place the description in a <div> element.

Technical specifications: ```{fact_sheet_chair}```
"""

```

## Summarization

### Limits

```python
prompt = f"""
Your task is to generate a short summary of a product \
review from an ecommerce site. 

Summarize the review below, delimited by triple 
backticks, in at most 30 words. 

Review: ```{prod_review}```
"""
```

### Focus

```python
# focus on shipment and delivery
prompt = f"""
Your task is to generate a short summary of a product \
review from an ecommerce site to give feedback to the \
Shipping deparmtment. 

Summarize the review below, delimited by triple 
backticks, in at most 30 words, and focusing on any aspects \
that mention shipping and delivery of the product. 

Review: ```{prod_review}```
"""

#focus on price

prompt = f"""
Your task is to generate a short summary of a product \
review from an ecommerce site to give feedback to the \
pricing deparmtment, responsible for determining the \
price of the product.  

Summarize the review below, delimited by triple 
backticks, in at most 30 words, and focusing on any aspects \
that are relevant to the price and perceived value. 

Review: ```{prod_review}```
"""

```

### Extract Infomation

```python
prompt = f"""
Your task is to extract relevant information from \ 
a product review from an ecommerce site to give \
feedback to the Shipping department. 

From the review below, delimited by triple quotes \
extract the information relevant to shipping and \ 
delivery. Limit to 30 words. 

Review: ```{prod_review}```
"""
```

## Inferring

### Sentiment

```python
prompt = f"""
What is the sentiment of the following product review, 
which is delimited with triple backticks?

Review text: '''{lamp_review}'''
"""

# give a more concise answer
prompt = f"""
What is the sentiment of the following product review, 
which is delimited with triple backticks?

Give your answer as a single word, either "positive" \
or "negative".

Review text: '''{lamp_review}'''
"""
```

### Emotions

```python
prompt = f"""
Identify a list of emotions that the writer of the \
following review is expressing. Include no more than \
five items in the list. Format your answer as a list of \
lower-case words separated by commas.

Review text: '''{lamp_review}'''
"""

prompt = f"""
Is the writer of the following review expressing anger?\
The review is delimited with triple backticks. \
Give your answer as either yes or no.

Review text: '''{lamp_review}'''
"""
```

### Extract Information

```python
prompt = f"""
Identify the following items from the review text: 
- Item purchased by reviewer
- Company that made the item

The review is delimited with triple backticks. \
Format your response as a JSON object with \
"Item" and "Brand" as the keys. 
If the information isn't present, use "unknown" \
as the value.
Make your response as short as possible.
  
Review text: '''{lamp_review}'''
"""

## do multiple tasks at once
prompt = f"""
Identify the following items from the review text: 
- Sentiment (positive or negative)
- Is the reviewer expressing anger? (true or false)
- Item purchased by reviewer
- Company that made the item

The review is delimited with triple backticks. \
Format your response as a JSON object with \
"Sentiment", "Anger", "Item" and "Brand" as the keys.
If the information isn't present, use "unknown" \
as the value.
Make your response as short as possible.
Format the Anger value as a boolean.

Review text: '''{lamp_review}'''
"""
```

### Infer Topics

```python

prompt = f"""
Determine five topics that are being discussed in the \
following text, which is delimited by triple backticks.

Make each item one or two words long. 

Format your response as a list of items separated by commas.

Text sample: '''{story}'''
"""

prompt = f"""
Determine whether each item in the following list of \
topics is a topic in the text below, which
is delimited with triple backticks.

Give your answer as list with 0 or 1 for each topic.\

List of topics: art, science, health, natura

Text sample: '''{story}'''
"""

```

## Transforming

### Translation

```python

prompt = f"""
Translate the following English text to Spanish: \ 
```Hi, I would like to order a blender```
"""

prompt = f"""
Tell me which language this is: 
```Combien coûte le lampadaire?```
"""

prompt = f"""
Translate the following  text to French and Spanish
and English pirate: \
```I want to order a basketball```
"""

prompt = f"""
Translate the following text to Spanish in both the \
formal and informal forms: 
'Would you like to order a pillow?'
"""

```

### Tone

```python
prompt = f"""
Translate the following from slang to a business letter: 
'Dude, This is Joe, check out this spec on this standing lamp.'
"""
```

### Format Conversion

```python
data_json = { "resturant employees" :[ 
    {"name":"Shyam", "email":"shyamjaiswal@gmail.com"},
    {"name":"Bob", "email":"bob32@gmail.com"},
    {"name":"Jai", "email":"jai87@gmail.com"}
]}

prompt = f"""
Translate the following python dictionary from JSON to an HTML \
table with column headers and title: {data_json}
"""
```

### Spellcheck/Grammer Check

```python

text = [ 
  "The girl with the black and white puppies have a ball.",  # The girl has a ball.
  "Yolanda has her notebook.", # ok
  "Its going to be a long day. Does the car need it’s oil changed?",  # Homonyms
  "Their goes my freedom. There going to bring they’re suitcases.",  # Homonyms
  "Your going to need you’re notebook.",  # Homonyms
  "That medicine effects my ability to sleep. Have you heard of the butterfly affect?", # Homonyms
  "This phrase is to cherck chatGPT for speling abilitty"  # spelling
]
for t in text:
    prompt = f"""Proofread and correct the following text
    and rewrite the corrected version. If you don't find
    and errors, just say "No errors found". Don't use 
    any punctuation around the text:
    ```{t}```"""
    response = get_completion(prompt)
    print(response)

```

Show differences.

```python

text = f"""
Got this for my daughter for her birthday cuz she keeps taking \
mine from my room.  Yes, adults also like pandas too.  She takes \
it everywhere with her, and it's super soft and cute.  One of the \
ears is a bit lower than the other, and I don't think that was \
designed to be asymmetrical. It's a bit small for what I paid for it \
though. I think there might be other options that are bigger for \
the same price.  It arrived a day earlier than expected, so I got \
to play with it myself before I gave it to my daughter.
"""
prompt = f"proofread and correct this review: ```{text}```"
response = get_completion(prompt)
print(response)

from redlines import Redlines

diff = Redlines(text,response)
display(Markdown(diff.output_markdown))

```

Correct and transform.

```python
prompt = f"""
proofread and correct this review. Make it more compelling. 
Ensure it follows APA style guide and targets an advanced reader. 
Output in markdown format.
Text: ```{text}```
"""
```

## Emails

```python
prompt = f"""
You are a customer service AI assistant.
Your task is to send an email reply to a valued customer.
Given the customer email delimited by ```, \
Generate a reply to thank the customer for their review.
If the sentiment is positive or neutral, thank them for \
their review.
If the sentiment is negative, apologize and suggest that \
they can reach out to customer service. 
Make sure to use specific details from the review.
Write in a concise and professional tone.
Sign the email as `AI customer agent`.
Customer review: ```{review}```
Review sentiment: {sentiment}
"""

```


## Chatbot

### Orderbot


```python
def collect_messages(_):
    prompt = inp.value_input
    inp.value = ''
    context.append({'role':'user', 'content':f"{prompt}"})
    response = get_completion_from_messages(context) 
    context.append({'role':'assistant', 'content':f"{response}"})
    panels.append(
        pn.Row('User:', pn.pane.Markdown(prompt, width=600)))
    panels.append(
        pn.Row('Assistant:', pn.pane.Markdown(response, width=600, style={'background-color': '#F6F6F6'})))
 
    return pn.Column(*panels)

import panel as pn  # GUI
pn.extension()

panels = [] # collect display 

context = [ {'role':'system', 'content':"""
You are OrderBot, an automated service to collect orders for a pizza restaurant. \
You first greet the customer, then collects the order, \
and then asks if it's a pickup or delivery. \
You wait to collect the entire order, then summarize it and check for a final \
time if the customer wants to add anything else. \
If it's a delivery, you ask for an address. \
Finally you collect the payment.\
Make sure to clarify all options, extras and sizes to uniquely \
identify the item from the menu.\
You respond in a short, very conversational friendly style. \
The menu includes \
pepperoni pizza  12.95, 10.00, 7.00 \
cheese pizza   10.95, 9.25, 6.50 \
eggplant pizza   11.95, 9.75, 6.75 \
fries 4.50, 3.50 \
greek salad 7.25 \
Toppings: \
extra cheese 2.00, \
mushrooms 1.50 \
sausage 3.00 \
canadian bacon 3.50 \
AI sauce 1.50 \
peppers 1.00 \
Drinks: \
coke 3.00, 2.00, 1.00 \
sprite 3.00, 2.00, 1.00 \
bottled water 5.00 \
"""} ]  # accumulate messages


inp = pn.widgets.TextInput(value="Hi", placeholder='Enter text here…')
button_conversation = pn.widgets.Button(name="Chat!")

interactive_conversation = pn.bind(collect_messages, button_conversation)

dashboard = pn.Column(
    inp,
    pn.Row(button_conversation),
    pn.panel(interactive_conversation, loading_indicator=True, height=300),
)

## make a summary of previos order
messages =  context.copy()
messages.append(
{'role':'system', 'content':'create a json summary of the previous food order. Itemize the price for each item\
 The fields should be 1) pizza, include size 2) list of toppings 3) list of drinks, include size   4) list of sides include size  5)total price '},    
)
 #The fields should be 1) pizza, price 2) list of toppings 3) list of drinks, include size include price  4) list of sides include size include price, 5)total price '},    

response = get_completion_from_messages(messages, temperature=0)
print(response)

```

## TELeR: A General Taxonomy of LLM Prompts for Benchmarking Complex Tasks

According to the paper; LLM prompts for complex
tasks are categorized along the following four dimensions.

1- Turn: Based on the number of turns used while prompting a language model in order to perform a complex task, prompts can be either single-turn or multi-turn.
2- Expression: Based on the expression style of the overall task as well as the associated sub-tasks, prompts can be either question-style or instruction-style.
3- Role: Based on whether a proper system role is defined in the LLM system before providing the actual prompt, prompts can have system-role either defined or undefined.
4- Level of Details: Based on whether particular aspects of the goal task definition is present or absent in the directive, prompts are divided into six distinct levels. Here, aspects refer to clear goal definition, sub-task division, explanation seeking, providing few-shot examples, etc. 

By definition, Level “0” means minimal details, i.e., no aspects/no directive; while level “5” means the highest level of details where the directive includes clear goals, distinct sub-tasks/steps, an explicit requirement of explanation/justification, well-defined criteria for evaluation and/or few-shot examples.

### Example Use Case 1 :  Meta-Review Generation

![Teler Figure 1]({{site.baseurl}}/assets/images/teler-figure-1.png)

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

Taken From:
https://github.com/wandb/edu/blob/main/llm-apps-course/notebooks/02.%20Generation.ipynb

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

## References
1. [Deep Learning AI Course](https://learn.deeplearning.ai/chatgpt-prompt-eng/lesson/1/introduction)
