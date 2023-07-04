---
layout: single
title:  "Block Oriented Web Page Crawler for Semantic Search"
classes: wide

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

## Objectives

Web pages contain a lot of redundat text. 
Pages commonly shares a similar structure which result in repetetion of text.

A better crawler should eliminate redundant text. 
This would enhance the question answering by reducing the embedding generation and LLM querying costs without sacrificing utility.

## Problems with Classical Approach

The classical approach combines text content from a webpage. 
It produces a long block of text. 
A long block of text is then can/should be split into smaller chunks using text splitters.

Splitting a text into smaller segments may result in losing some of the context, as the splitting algorithm may not be able to determine the relevance of different parts of the text.

One solution which is used commonly to mitigate adverse effect of splitting is overlapping chunks. 
Some of the text from the previos chunk is prepended to later chunk. 
However, overlap size is not easy to tune. 

## Can We Do Better

Tags can be used to split text in a better way. 
Tags contains useful logical structuring of the HTML text. 

For example a DIV tag contains a block of text that may be releated. DIV boundaries thus can be used to split text such splitting is less sensivite to context loss. 

## The Algorithm

The algorithm has two parts: text scrapping and link mining.


### Text Scrapping

A depth first tree search algorithm is implemtened to traverse DOM nodes found in an HTML document. If a DIV block is met, a new split is created and all contained text is included. 

If a block of text is too long, it is splitted into overlapping chunks.

A set of previous text blocks is kept so that a text block is added only if it passes duplication check. Block based scrapping allows repetition check which is not possible with the plain scrapping.

### Link Mining

All links in a page are put into a list.
Links in the list are visited one by one and scrapped in the way.


## Example 

Following blocks shows two html text presumably belongs to two different page of a website. Content is copied from britannica.

```html
<body>
  <div>Britannica contains a lot of information.</div>
<div>

  <h1>Europe Countries</h1>
  <div>
    <h2>Germany
    </h2>
    <p>Germany, officially Federal Republic of Germany, German Deutschland or Bundesrepublik Deutschland, country of north-central Europe, traversing the continent’s main physical divisions, from the outer ranges of the Alps northward across the varied landscape of the Central German Uplands and then across the North German Plain.
    </p>
  </div>

  <div>
    <h2>France
    </h2>
    <p>France, officially French Republic, French France or République Française, country of northwestern Europe. Historically and culturally among the most important nations in the Western world, France has also played a highly significant role in international affairs, with former colonies in every corner of the globe.
    </p>
  </div>

</div>
</body>
```

```html
<body>
  <div>Britannica contains a lot of information.</div>
  <div>

  <h1>Asia Countries</h1>
  <div>
    <h2>China
    </h2>
    <p>China, Chinese (Pinyin) Zhonghua or (Wade-Giles romanization) Chung-hua, also spelled (Pinyin) Zhongguo or (Wade-Giles romanization) Chung-kuo, officially People’s Republic of China or Chinese (Pinyin) Zhonghua Renmin Gongheguo or (Wade-Giles romanization) Chung-hua Jen-min Kung-ho-kuo, country of East Asia. It is the largest of all Asian countries and has the largest population of any country in the world. 
    </p>
  </div>

  <div>
    <h2>India
    </h2>
    <p>India, country that occupies the greater part of South Asia. Its capital is New Delhi, built in the 20th century just south of the historic hub of Old Delhi to serve as India’s administrative centre. Its government is a constitutional republic that represents a highly diverse population consisting of thousands of ethnic groups and likely hundreds of languages.
    </p>
  </div>
  </div>
</body>
```

Chunk size is selected as 100 and overlap as 20.

### Plain Scrapping

```python
['\nBritannica contains a lot of information.\n\nEurope Countries\n\nGermany\n    \nGermany, officially Federal Republic of Germany, German Deutschland or Bundesrepublik Deutschland, country of north-central Europe, traversing the continent’s main physical divisions, from the outer ranges of the Alps northward across the varied landscape of the Central German Uplands and then across the North German Plain.\n    \n\n\nFrance\n ',
 'plands and then across the North German Plain.\n    \n\n\nFrance\n    \nFrance, officially French Republic, French France or République Française, country of northwestern Europe. Historically and culturally among the most important nations in the Western world, France has also played a highly significant role in international affairs, with former colonies in every corner of the globe.\n    \n\n\n',
 '\n    \n\n\n']

```

```python
['\nBritannica contains a lot of information.\n\nAsia Countries\n\nChina\n    \nChina, Chinese (Pinyin) Zhonghua or (Wade-Giles romanization) Chung-hua, also spelled (Pinyin) Zhongguo or (Wade-Giles romanization) Chung-kuo, officially People’s Republic of China or Chinese (Pinyin) Zhonghua Renmin Gongheguo',
 's Republic of China or Chinese (Pinyin) Zhonghua Renmin Gongheguo or (Wade-Giles romanization) Chung-hua Jen-min Kung-ho-kuo, country of East Asia. It is the largest of all Asian countries and has the largest population of any country in the world. \n    \n\n\nIndia\n    \nIndia, country that occupies the greater part of South Asia. Its capital is New',
 '   \nIndia, country that occupies the greater part of South Asia. Its capital is New Delhi, built in the 20th century just south of the historic hub of Old Delhi to serve as India’s administrative centre. Its government is a constitutional republic that represents a highly diverse population consisting of thousands of ethnic groups and likely hundreds of languages.\n    \n\n\n']
```

Note that common text mentioning Britannica is included in both pages. 

Also note that word france follows definition germany. And definition of france starts with some text from germany part.

Similarly definition of china includes parts from india definition. By mere coincidence last element contains information only about india. Note that some text is overlapping in 2nd and 3rd elements.



### Block Scrapping


```python
['Britannica contains a lot of information.',
 'Germany: Germany, officially Federal Republic of Germany, German Deutschland or Bundesrepublik Deutschland, country of north-central Europe, traversing the continent’s main physical divisions, from the outer ranges of the Alps northward across the varied landscape of the Central German Uplands and then across the North German Plain.',
 'France: France, officially French Republic, French France or République Française, country of northwestern Europe. Historically and culturally among the most important nations in the Western world, France has also played a highly significant role in international affairs, with former colonies in every corner of the globe.',
 'Europe Countries:']


```


```python
['China: China, Chinese (Pinyin) Zhonghua or (Wade-Giles romanization) Chung-hua, also spelled (Pinyin) Zhongguo or (Wade-Giles romanization) Chung-kuo, officially People’s Republic of China or Chinese (Pinyin) Zhonghua Renmin Gongheguo or (Wade-Giles romanization) Chung-hua Jen-min Kung-ho-',
 'India: India, country that occupies the greater part of South Asia. Its capital is New Delhi, built in the 20th century just south of the historic hub of Old Delhi to serve as India’s administrative centre. Its government is a constitutional republic that represents a highly diverse population consisting of thousands of ethnic groups and likely hundreds of languages.',
 'Asia Countries:',
 ' (Wade-Giles romanization) Chung-hua Jen-min Kung-ho-kuo, country of East Asia. It is the largest of all Asian countries and has the largest population of any country in the world.']
```

Note that common text is only included in the first page. 

Splits are shorter than they are in regular scrapping. As a result there are more chunks.

Sections defining countries are not overlapping. The only overlap occurs in definition of china because it is too long and needs to be split. 


## Benchmarks

To see behaviour of the proposed algorithm, a comparative study is conducted. 

Popular news portal bbc.com is targeted. Two approaches are compared under different settings.

In the first setting, number of urls fetched varied from 1 to 7.

In the second setting, number of urls is fixed and minimum number of chars filter is adjusted. If a split has less than filter allows it is not included in result. This filter does not have a meaningful effect in regular scrapping but has substantial effect in block based approach. This is because significant amount of blocks contain information that is too short to help answer any question. 

### Total Words by Number of URLs

![My image]({{site.baseurl}}/assets/images/crawler-urls-words.png)

### Total Number of Words by Minimum Length Filter

![My image]({{site.baseurl}}/assets/images/crawler-limit-total-words.png)

### Average Number of Words by Minimum Length Filter

![My image]({{site.baseurl}}/assets/images/crawler-limit-average-words.png)

## References
1. [Github Source Code](https://github.com/habanoz/crawl-for-vector-db)
