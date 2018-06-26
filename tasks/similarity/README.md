# LASER: application to multilingual similarity search

This codes shows how to embed an N-way parallel corpus (we
use the publicly available newstest2009 from WMT 2009), and
how to calculate the similarity search error rate for each language pair.

For each sentence in the source language, we calculate the closest sentence in
the joint embedding space in the target language. If this sentence has the same
index in the file, it is considered as correct, and as an error else wise.
Therefore, the N-way parallel corpus should not contain duplicates.

## Installation

* simply run the script
```bash
./sim.sh
```
  It downloads the data, calculates the sentence embeddings 
  and the similarity search error rate for each language pair.

## Results

You should get the following similarity search errors:

Confusion matrix:

|    |  de   |  en   |  es   |  fr   |
|----|-------|-------|-------|-------|
| de | 0.00% | 2.30% | 2.26% | 2.69% |
| en | 2.46% | 0.00% | 1.54% | 1.39% |
| es | 1.86% | 1.90% | 0.00% | 1.58% |
| fr | 2.73% | 1.27% | 1.39% | 0.00% |
