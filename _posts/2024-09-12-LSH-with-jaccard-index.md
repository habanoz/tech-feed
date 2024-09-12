---
layout: single
title: "LSH with Jaccard Index"
categories:
  - workshop

tags:
  - minhash algorithm
  - lsh
  - locality sensitive hashing
---

Minhash algorithm can be used to detect near duplicate documents. Minhash algorithm works by calculating multiple hashes for different sections of a document n-grams. For each section hash value with the minimum value is selected. Selected hash values for the signature for each document. Signatures can be used to calculating Jaccard similarity between document pairs. Documents with similarity exceeding some threshold e.g. 0.8 are assumed near duplicate and removed from the dataset. 

Minhash algorithm works well. The issue is that for N documents, $$ N ^ 2 / 2 $$ comparisons are needed. 


## Locality Sensitive Hashing

Locality Sensitive Hashing (LSH) is an algorithm that is typically used together with Minhash algorithm to find near duplicate documents. The idea is that, instead of comparing document signatures, document signature is split into smaller buckets and buckets are compared for equality. It is possible to crate a set of bucket values for fast comparison. 

A document signature is split into `b` buckets, each bucket containing `r` hash values or rows. Consequently, a document signature has `b * r` hashes.


Suppose we have two documents to compare, A and B. Then J(A,B) is jaccard similarity between the documents:

- $$ J(A,B) $$ is the probability of two documents having the same hash value in a row. (Note that this probability is independent among rows)
- $$ J(A,B)^r $$ is the probability of two documents having the same hash value in all rows of bucket. 
- $$ 1- J(A,B)^r $$ is the complementary probability which gives the probability of two documents NOT having the same hash value in all rows of bucket. Which means they have at least one different hash value in a row.
- $$ (1- J(A,B)^r)^b $$ is the probability of having at least one different hash values in all buckets which does not give any matches.
- $$ 1- (1- J(A,B)^r)^b $$ gives the complementary probability of the previous step. It gives the probability that at least one of the buckets are equal.   

With `b=20` and `r=450` if `J(A,B)=0.8` then with 0.994 probability at least one bucket will be equal thus there will be a match and documents will be correctly identified as duplicates.

Recall will be high as it is very likely to detect duplicate documents.
Precision may be low because non-duplicate documents may match at least in one buckets. But it is possible to increase precision. 

## Datatrove Library

DataTrove is a library to process, filter and deduplicate text data at a very large scale. It is the library used to build FineWeb dataset by Huggingface.

DataTrove does calculate Jaccard similarity for matching document which suggest they focused on recall and skipped additional step similar to [2].

## DataTrove with Jaccard Index Step

I went ahead and added jaccard similarity calculation step for matching documents.

### Save Whole Signatures

DataTrove only keeps buckets. For a jaccard index calculation whole signature is needed. Thus first step is to save signatures to disk for quick access when needed.

```
if self.save_complete_sigs:
  with self.output_folder.open(f"sigs/{rank:02d}-{doc_idx}.sig.txt", mode="w") as sigfile:
    np.savetxt(sigfile, sig)
```

### Calculate Jaccard Index

In `MinhashDedupBuckets` step, calculate jaccard index if there is a match and a jaccard threshold is provided.


```python
def get_jaccard_index(self, doc1_file_stem, doc1_id, doc2_file_stem, doc2_id):
    
    doc1_sig = self.load_sig(doc1_file_stem,doc1_id)
    doc2_sig = self.load_sig(doc2_file_stem,doc2_id)
    
    similarity = np.sum(doc1_sig==doc2_sig)/doc1_sig.shape[0]
    
    return similarity

def load_sig(self, file_stem, doc_id):
    with self.input_folder.open(f"sigs/{file_stem}-{doc_id}.sig.txt", mode="r") as sigfile:
        return np.loadtxt(sigfile)

def is_jaccard_match(self, doc1_file_stem, doc1_id, doc2_file_stem, doc2_id):
    assert self.jaccard_threshold is not None
    return self.get_jaccard_index(doc1_file_stem, doc1_id, doc2_file_stem, doc2_id) > self.jaccard_threshold

***

  if last and last.sig == v.sig:
      if last.is_from_index():
        ***
      # if there isn't an index, or we are not only deduping in relation to the index
      elif not index_files or not self.only_dedup_in_index:
          # when there is match at the given bucket and a jaccard threshold is provided; calculate jaccard index
          if self.jaccard_threshold is None or self.is_jaccard_match(last.file_stem, last.doc_id, v.file_stem, v.doc_id):
              out_f.write(
                  struct.pack("<4I", int(last.file_stem), last.doc_id, int(v.file_stem), v.doc_id)
              )
              self.stat_update("total_jaccard_matches")
          self.stat_update("total_matches")
```

## Results

17K documents are used in deduplication tests.

Test configuration: 5-grams, `b=20` and `r=450`

There are 144M (17K*17K/2) document pairs. At each bucket, approximately 3560 matches are found. For each match, Jaccard Index is calculated. In total, 243 matches with jaccard similarity less then 0.8 is found. In other words, at most 243 documents were false positives. It is only 1.4 % of the documents.


## Conclusion

In my judgement, it is fair to ignore 1.4 % false positives to avoid spending time on jaccard calculations and saving duplicate complete signatures on the disk. So I abandoned the the idea.   


## References

1- [Deduplicating Training Data Makes Language Models Better](https://arxiv.org/pdf/2107.06499)

2- [The RefinedWeb Dataset for Falcon LLM](https://arxiv.org/pdf/2306.01116)

3- [DataTrove](https://github.com/huggingface/datatrove)

4- [The FineWeb Datasets](https://arxiv.org/pdf/2406.17557)