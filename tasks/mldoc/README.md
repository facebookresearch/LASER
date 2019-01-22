# LASER: application to cross-lingual document classification

This codes shows how to use the multilingual sentence embedding for
cross-lingual document classification, using the MLDoc corpus [1].

We train a  document classifier on one language (e.g. English) and apply it then
to several other languages without using any resource of that language
(e.g. German, Spanish, French, Italian, Japanese, Russian and Chinese)

## Installation

* Please first download the MLDoc corpus from
  [here](https://github.com/facebookresearch/MLDoc)
  and install it in the directory MLDoc
* Calculate the multilingual sentence embeddings for all languages
  and train the classifier `bash ./mldoc.sh`

## Results

We use an MLP classifier with two hidden layers and Adam optimization.

You should get the following results for zero-short cross-lingual transfer
These results are in average better than those reported in [2] since the system has
been improved since publication.

| Train language |   En   |   De   |   Es   |   Fr   |   It   |   Ja   |   Ru   |   Zh  |
|----------------|--------|--------|--------|--------|--------|--------|--------|-------|
| English (en)   | 90.73  | 86.25  | 79.30  | 78.03  | 70.20  | 60.95  | 67.25  | 70.98 |
| German (de)    | 80.75  | 92.70  | 79.60  | 82.83  | 73.25  | 56.80  | 68.18  | 72.90 |
| Spanish (es)   | 69.58  | 79.73  | 88.75  | 75.30  | 71.10  | 59.65  | 59.83  | 61.70 |
| French (fr)    | 80.08  | 87.03  | 78.40  | 90.80  | 71.08  | 53.60  | 67.55  | 66.12 |
| Italian (it)   | 74.15  | 80.73  | 82.60  | 78.35  | 85.93  | 55.15  | 68.83  | 56.10 |
| Japanese (ja)  | 68.45  | 81.90  | 67.95  | 67.95  | 57.98  | 85.15  | 53.70  | 66.12 |
| Russian (ru)   | 72.60  | 79.62  | 68.18  | 71.28  | 67.00  | 59.23  | 84.65  | 65.62 |
| Chinese (zh)   | 77.95  | 83.38  | 78.38  | 75.83  | 70.33  | 55.25  | 66.62  | 88.98 |

All numbers are accuracies on the test set.

## References

Details on the corpus are described in this paper:

[1] Holger Schwenk and Xian Li,
    [*A Corpus for Multilingual Document Classification in Eight Languages*](http://www.lrec-conf.org/proceedings/lrec2018/pdf/658.pdf),
    LREC, pages 3548-3551, 2018.

Detailed system description:

[2] Mikel Artetxe and Holger Schwenk,
    [*Massively Multilingual Sentence Embeddings for Zero-Shot Cross-Lingual Transfer and Beyond*](https://arxiv.org/abs/1812.10464),
    arXiv, Dec 26 2018.
