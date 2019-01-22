# LASER: application to multilingual similarity search

This codes shows how to embed an N-way parallel corpus (we
use the publicly available newstest2012 from WMT 2012), and
how to calculate the similarity search error rate for each language pair.

For each sentence in the source language, we calculate the closest sentence in
the joint embedding space in the target language. If this sentence has the same
index in the file, it is considered as correct, and as an error else wise.
Therefore, the N-way parallel corpus **should not contain duplicates.**

## Installation

* simply run the script `bash ./wmt.sh`
  to downloads the data, calculate the sentence embeddings 
  and the similarity search error rate for each language pair.

## Results

You should get the following similarity search errors:

|     |   cs  |   de  |   en  |   es  |   fr   |  avg  |  
|-----|-------|-------|-------|--------|-------|-------|
| cs  | 0.00% | 0.70% | 0.90% | 0.67%  | 0.77% | 0.76% |
| de  | 0.83% | 0.00% | 1.17% | 0.90%  | 1.03% | 0.98% |
| en  | 0.93% | 1.27% | 0.00% | 0.83%  | 1.07% | 1.02% |
| es  | 0.53% | 0.77% | 0.97% | 0.00%  | 0.57% | 0.71% |
| fr  | 0.50% | 0.90% | 1.13% | 0.60%  | 0.00% | 0.78% |
| avg | 0.70% | 0.91% | 1.04% | 0.75%  | 0.86% | 1.06% |
