# WikiMatrix: Mining 135M Parallel Sentences in 1620 Language Pairs from Wikipedia

The goal of this project is to mine for parallel sentences in the textual content of Wikipedia for all possible language pairs.


## Mined data
* 85 different languages, 1620 language pairs
* 134M parallel sentences, out of which 34M are aligned with English
* this [*table shows the amount of mined parallel sentences for most of the language pairs*](WikiMatrix-sizes.pdf)
* the mined bitext are stored on AWS and can de downloaded with the following command:
```bash
wget https://dl.fbaipublicfiles.com/laser/WikiMatrix/v1/WikiMatrix.en-fr.tsv.gz
```
Replace "en-fr" with the ISO codes of the desired language pair.
The language pair must be in alphabetical order, e.g. "de-en" and not "en-de".
The list of available bitexts and their sizes are given in the file [*list_of_bitexts.txt*](list_of_bitexts.txt).
Please do **not loop over all files** since AWs implements some [*limitations*](https://dl.fbaipublicfiles.com/README) to avoid abuse.

Use this command if you want to download all 1620 language pairs in one tar file (but this is 65GB!):
```bash
wget https://dl.fbaipublicfiles.com/laser/WikiMatrix/WikiMatrix.v1.1620_language_pairs.tar
```

## Approach

We use LASER's bitext mining approach and encoder for 93 languages [2,3].
We do not use the inter-language links provided by Wikipedia,
but search over all Wikipedia articles of each language.  We approach the
computational challenge to mine in almost 600 million sentences by using fast
indexing and similarity search with [*FAISS*](https://github.com/facebookresearch/faiss).
Prior to mining parallel sentences, we perform
sentence segmentation, deduplication and language identification.
Please see reference [1] for details.


## Data extraction and threshold optimization
We provide a tool to extract parallel texts from the the TSV files:
```bash
python3 extract.py \
  --tsv WikiMatrix.en-fr.tsv.gz \
  --bitext WikiMatrix.en-fr.txt \
  --src-lang en --trg-lang fr \
  --threshold 1.04
```
One can specify the threshold on the margin score.
The higher the value, the more likely the sentences are mutual translations, but the less data one will get.
**A value of 1.04 seems to be good choice for most language pairs.** Please see the analysis in the paper for
more information [1].

## Evaluation
To assess the quality of the mined bitexts, we trained neural MT system on all language pairs
for which we were able to mine at least 25k parallel sentences (with a margin threshold of 1.04).
We trained systems in both directions, source to target and target to source, and report BLEU scores
on the [*TED test*](https://github.com/neulab/word-embeddings-for-nmt) set proposed in [4].
This totals 1886 different NMT systems.
This [*table shows the BLEU scores for the most frequest language pairs*](WikiMatrix-bleu.pdf).
We achieve BLEU scores over 30 for several language pairs.

The goal is not to build state of the art systems for each language pair, but
to get an indication of the quality of the automatically mined data.  These
BLEU scores should be of course appreciated in context of the sizes of the
mined corpora.

Obviously, we can not exclude that the
provided data contains some wrong alignments even though the margin is large.
Finally, we would like to point out that we run our approach on all available
languages in Wikipedia, independently of the quality of LASER's sentence
embeddings for each one.


## License

The mined data is distributed under the Creative Commons Attribution-ShareAlike license.

Please cite reference [1] if you use this data.

## References

[1] Holger Schwenk, Vishrav Chaudhary, Shuo Sun, Hongyu Gong and Paco Guzman,
    [*WikiMatrix: Mining 135M Parallel Sentences in 1620 Language Pairs from Wikipedia*](https://arxiv.org/abs/1907.05791)
    arXiv, July 11  2019.

[2] Mikel Artetxe and Holger Schwenk,
    [*Margin-based Parallel Corpus Mining with Multilingual Sentence Embeddings*](https://arxiv.org/abs/1811.01136)
    arXiv, Nov 3 2018.

[3] Mikel Artetxe and Holger Schwenk,
    [*Massively Multilingual Sentence Embeddings for Zero-Shot Cross-Lingual Transfer and Beyond*](https://arxiv.org/abs/1812.10464)
    arXiv, Dec 26 2018.

[4] Ye Qi, Devendra  Sachan, Matthieu Felix, Sarguna Padmanabhan and Graham Neubig,
    [*When and Why Are Pre-Trained Word Embeddings Useful for Neural Machine Translation?*](https://www.aclweb.org/anthology/papers/N/N18/N18-2084/)
    NAACL, pages 529-535, 2018.
