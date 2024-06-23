---
layout: single
title: "Counts vs Gradients - Developing A Bi-gram Language Model - A Case Study"
categories:
  - workshop

tags:
  - bi-gram models
  - language models
  - gradient descent
  - pytorch
---

I am inspired from the make more lecture of Andrej Karpathy. This post aims to showcase similarities between count based and gradient based methods to generate a bi-gram language model. This approach may help develop a better intuition on understanding how gradient based methods work. I encourage you to watch the make more lecture to understand the underlying concepts better.

Bi-gram models generate a statistical model that predicts the next character based on the previous character. Unlike the lecture which works on a dataset of names, I worked on a dictionary of Turkish words. 

## Collect Data

Let's download a dictionary of Turkish words.

```bash
!wget https://raw.githubusercontent.com/ncarkaci/TDKDictionaryCrawler/master/ortak_kelimeler.txt

with open("ortak_kelimeler.txt") as f:
    text = f.read()
  
words = text.split("\n")
```

Let's generate samples.

```python
xs, ys = [], []
for w in words:
    w = "."+w+"."
    for c1, c2 in zip(w, w[1:]):
        ix = ctoi[c1]
        iy = ctoi[c2]

        xs.append(ix)
        ys.append(iy)

xs = torch.tensor(xs)
ys = torch.tensor(ys)
```

## Count based bi-gram model

Count based bi-gram is very simple and intuitional. We count occurrences of each pair of characters and normalize them to get probabilities.

```python
# initialize count tensor
T = torch.zeros([30, 30], dtype=torch.int32)

# generate counts
for ix, iy in zip(xs, ys):
    T[ix][iy] += 1

# generate probabilities
probs = T / T.sum(dim=1, keepdim=True)
```

Let's see probabilities for the initial character.

```python
start_probs = probs[ctoi['.']]
start_probs
```

Here is a formatted (chars added) representation of the initial probabilities.

```
[(('a', '0.0730'), ('b', '0.0730'), ('c', '0.0157')),
 (('d', '0.0650'), ('e', '0.0362'), ('f', '0.0215')),
 (('g', '0.0432'), ('h', '0.0399'), ('i', '0.0401')),
 (('j', '0.0020'), ('k', '0.1182'), ('l', '0.0108')),
 (('m', '0.0646'), ('n', '0.0159'), ('o', '0.0183')),
 (('p', '0.0339'), ('r', '0.0149'), ('s', '0.0810')),
 (('t', '0.0667'), ('u', '0.0146'), ('v', '0.0139')),
 (('y', '0.0487'), ('z', '0.0117'), ('ç', '0.0329')),
 (('ö', '0.0157'), ('ü', '0.0067'), ('ğ', '0.0000')),
 (('ı', '0.0043'), ('ş', '0.0177'), ('.', '0.0000'))]
```

Let's look at the average negative log likelihood.

```python
nll_sum = 0
nll_count = 0
for ix,iy in zip(xs,ys):
    prob = probs[ix, iy]
    log_p = torch.log(prob)

    nll = -log_p # negative log likelihood

    nll_sum += nll # sum of negative log likelihoods
    nll_count += 1

loss = nll_sum / nll_count # average negative log likelihoods
print(loss.item())
```

Loss is `2.5226`. Note that this is the ideal loss. We will use this loss as target in the gradient based approach.

Here are some notes on the use of negative log likelihood. Our goal is to maximize the likelihood of our data. Maximizing likelihood is similar to maximizing log likelihood because log function is a strictly increasing function. Log likelihood is preferred because it is easy to work with logs. 

Maximizing log likelihood is equivalent to minimizing negative log likelihood. Note that log function takes negative values for probabilities. 

Negative log likelihood as a loss function gives values close to zero when probabilities are close to 1 which makes sense. Negative log likelihood as a loss function goes to infinity when probabilities are close to 0 which makes sense too.

[Check out the -log(x)](https://www.wolframalpha.com/input?i=-log%28x%29+from+0+to+1)

## Gradient Based bi-gram model

Gradient based model is a bit complex compared to the count based method. However they share similarities. Also it is important to note that gradient based method is much more powerful that can be applied to much more complicated problems.

Note that neural networks can not be used for counting. But we can use them to learn the parameters of a model that is similar to counting. W tensor is similar to T tensor of the count method that keeps the counts. W tensor values can be taught as log counts. one hot vector and W multiplication extracts log counts for the given input character. This log count is converted to pseudo-counts by using exponationation function. Pseudo-counts are then converted to probabilities by using normalization in the exact same manner with the count based method. Also note that exponentiation function and normalization together corresponds to the softmax function.

Use of one hot vectors is for indexing into the W vector. One hot vector has only one value set to 1 and all other values are set to 0. If we multiply a one hot vector with a tensor, the result will be the row of the tensor at that index, all other rows will be discarded. This is analogous to the count based method where we use an index into the T tensor.

Negative log likelihood is calculated in the same way. 

```python
# initialize weights tensor
W = torch.rand((30,30), requires_grad=True)

# apply gradient descent for 500 steps
for i in range(500):

    # forward
    xenc = F.one_hot(xs, num_classes=30).float()
    logits = xenc @ W  # log counts
    # softmax
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdim=True)
    # calculate loss
    loss = -probs[torch.arange(len(xs)), ys].log().mean()  # negative log likelihood

    if i % 10 ==0:
        print(f"Step {i} | loss: {loss.item():4f}")

    # backward
    W.grad = None
    loss.backward()

    # update weights
    W.data += -10 * W.grad
print(f"Step {i} | loss: {loss.item():4f}")
```

Loss decreases gradually. Training is stopped when the loss is sufficiently close to the loss of count based method.

```
Step 0 | loss: 3.459375
Step 10 | loss: 3.104548
Step 20 | loss: 2.929919
Step 30 | loss: 2.835742
Step 40 | loss: 2.776365
Step 50 | loss: 2.735428
Step 60 | loss: 2.705555
Step 70 | loss: 2.682837
Step 80 | loss: 2.664999
Step 90 | loss: 2.650638
Step 100 | loss: 2.638838
Step 110 | loss: 2.628977
Step 120 | loss: 2.620616
Step 130 | loss: 2.613439
Step 140 | loss: 2.607211
...
Step 470 | loss: 2.547258
Step 480 | loss: 2.546711
Step 490 | loss: 2.546187
Step 499 | loss: 2.545733
```

Now let's look at the probabilities for the initial character.

```python
xenc = F.one_hot(torch.tensor([ctoi['.']]), num_classes=30).float()
logits = xenc @ W
counts = logits.exp()
start_probs = counts / counts.sum(1, keepdim=True)
start_probs
```

Here is a formatted (chars added) representation of the initial probabilities.

```
[(('a', '0.0729'), ('b', '0.0729'), ('c', '0.0155')),
 (('d', '0.0649'), ('e', '0.0361'), ('f', '0.0213')),
 (('g', '0.0430'), ('h', '0.0398'), ('i', '0.0399')),
 (('j', '0.0026'), ('k', '0.1181'), ('l', '0.0106')),
 (('m', '0.0645'), ('n', '0.0158'), ('o', '0.0181')),
 (('p', '0.0338'), ('r', '0.0147'), ('s', '0.0808')),
 (('t', '0.0665'), ('u', '0.0145'), ('v', '0.0137')),
 (('y', '0.0486'), ('z', '0.0115'), ('ç', '0.0328')),
 (('ö', '0.0156'), ('ü', '0.0067'), ('ğ', '0.0015')),
 (('ı', '0.0044'), ('ş', '0.0176'), ('.', '0.0015'))]
```

So the probabilities are in the same ballpark with the count based method.


## Generate New Words

This how we can use the probabilities tensor to generate new words.

```python
torch.manual_seed(35)
for _ in range(5):
    ix = ctoi['.']
    while True:
        ix = torch.multinomial(probs[ix], num_samples=1, replacement=True).item()
        if ix == ctoi['.']:
            break
        print(itoc[ix], end='')
        
    print()
```
Generations:

```
atettı
ik
öncırısörarsı
kakl
satsabiçinorekom
```

This how we can use the bi-gram model weights tensor to generate new words.

```python
torch.manual_seed(35)
for _ in range(5):
    ix = ctoi['.']
    while True:
        xenc = F.one_hot(torch.tensor([ix]), num_classes=30).float()
        logits = xenc @ W
        counts = logits.exp()
        probs = counts / counts.sum(1, keepdim=True)
        ix = torch.multinomial(probs, num_samples=1, replacement=True).item()
        if ix == ctoi['.']:
            break
        print(itoc[ix], end='')
    print('')
```

Generations:

```
atettı
ik
öncırısörarsı
kakl
satsabiçinorekom
```

Note that we use manual seeds for reproducibility. When the same seed is given, bot methods generates similar words. 
In some cases, some words may differ slightly which is probably due to tiny differences between the probabilities generated by the two methods. (Maybe there are some subtle bugs).


## Conclusion

Bi-gram models generations are not perfect but they are a good starting point for text generation tasks. I observed much better generations with 3-gram or more grams. This workshop aims to show similarities between the count based and the gradient based methods.


## References

1- [Make More](https://www.youtube.com/watch?v=PaCmpygFfXo)
2- [TDKDictionaryCrawler](https://github.com/ncarkaci/TDKDictionaryCrawler)