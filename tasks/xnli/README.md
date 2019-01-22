# LASER: application to cross-lingual natural language inference

This codes shows how to use the multilingual sentence embedding for
cross-lingual NLI, using the XNLI corpus.

We train a NLI classifier on the English MultiNLI corpus, optimizing
the meta-parameters on the English XNLI development corpus.
We then apply that classifier to the test set for all 14 transfer languages.
The foreign languages development set is not used.

## Installation

Just run `bash ./xnli.sh`
which install XNLI and MultiNLI corpora, 
calculates the multilingual sentence embeddings,
trains the classifier and displays results.

The XNLI corpus is available [here](https://www.nyu.edu/projects/bowman/xnli/).

## Results

You should get the following results for zero-short cross-lingual transfer.
They slightly differ from those published in the initial version of the paper [2]
due to the change to PyTorch 1.0 and variations in random number generation, new optimization of meta-parameters, etc.

|   en  |   fr  |   es  |   de  |   el  |   bg  |   ru  |   tr  |   ar  |   vi  |   th  |   zh  |   hi  |   sw  |   ur  |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| 74.65 | 72.26 | 73.15 | 72.48 | 72.73 | 73.35 | 71.08 | 69.84 | 70.48 | 71.94 | 69.20 | 71.38 | 65.95 | 62.14 | 61.82 |

All numbers are accuracies on the test set

## References

Details on the corpus are described in this paper:

[1] Alexis Conneau, Guillaume Lample, Ruty Rinott, Adina Williams, Samuel R. Bowman, Holger Schwenk and Veselin Stoyanov,
[*XNLI: Cross-lingual Sentence Understanding through Inference*](https://aclweb.org/anthology/D18-1269),
EMNLP, 2018.

Detailed system description:

[2] Mikel Artetxe and Holger Schwenk,
[*Massively Multilingual Sentence Embeddings for Zero-Shot Cross-Lingual Transfer and Beyond*](https://arxiv.org/pdf/1812.10464),
arXiv, Dec 26 2018.
