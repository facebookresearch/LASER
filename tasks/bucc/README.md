# LASER: application to bitext mining

This codes shows how to use the multilingual sentence embeddings to mine
for parallel data in (huge) collections of monolingual data.

The underlying idea is pretty simple:
* embed the sentences in the two languages into the joint sentence space
* calculate all pairwise distances between the sentences.
  This is of complexity O(N\*M) and can be done very efficiently with
  the FAISS library [2]
* all sentence pairs which have a distance below a threshold
  are considered as parallel
* this approach can be further improved using a margin criterion [3] 

Here, we apply this idea to the data provided by the shared task of the BUCC 
[Workshop on Building and Using Comparable Corpora](https://comparable.limsi.fr/bucc2018/bucc2018-task.html).

The same approach can be scaled up to huge collections of monolingual texts
(several billions) using more advanced features of the FAISS toolkit.

## Installation

* Please first download the BUCC shared task data
  [here](https://comparable.limsi.fr/bucc2017/cgi-bin/download-data-2018.cgi)
  and install it the directory "downloaded"
* running the script
```bash
./bucc.sh
```

## Results

Optimized on the F-scores on the training corpus.
These results differ slighty from those published in [4] due to the switch from PyTorch 0.4 to 1.0.

| Languages | Threshold | precision | Recall | F-score |
|-----------|-----------|-----------|--------|---------|
|   fr-en   |  1.088131 |   91.52   |  93.32 |  92.41  |
|   de-en   |  1.092056 |   95.65   |  95.19 |  95.42  |
|   ru-en   |  1.093404 |   90.60   |  94.04 |  92.29  |
|   zh-en   |  1.085999 |   91.99   |  91.31 |  91.65  |

Results on the official test set are scored by the organizers of the BUCC workshop.


Below, we compare our approach to the [official results of the 2018 edition
of the BUCC workshop](http://lrec-conf.org/workshops/lrec2018/W8/pdf/12_W8.pdf) [1].
More details on our approach are provided in [2,3,4]

|               System | fr-en | de-en | ru-en | zh-en |
|----------------------|-------|-------|-------|-------|
|   Azpeitia et al '17 |  79.5 |  83.7 |   -   |   -   |
|   Azpeitia et al '18 |  81.5 |  85.5 |  81.3 |  77.5 |
|Bouamor and Sajjad '18|  76.0 |   -   |   -   |   -   |
|   Chongman et al '18 |   -   |   -   |   -   |  56   |
|            LASER [3] |  75.8 |  76.9 |   -   |   -   |
|            LASER [4] |  93.1 |  96.2 |  92.3 |  92.7 |

All numbers are F1-scores on the test set.

## Bonus

To show case the highly multilingual aspect of LASER's sentence embeddings,
we also mine for bitexts for language pairs which do not include English, e.g.
French-German, Russian-French or Chinese-Russian.
This is also performed by the script bucc.sh

Below the number of extracted parallel sentences for each language pair.

| src/trg | French | German | Russian | Chinese |
|---------|--------|--------|---------|---------|
| French  |  n/a   |  2795  |  3327   |  387    |
| German  |  2795  |  n/a   |  3661   |  466    |
| Russian |  3327  |  3661  |   n/a   |  664    |
| Chinese |   387  |   466  |   664   |  n/a    |


## References

[1] Pierre Zweigenbaum, Serge Sharoff and Reinhard Rapp,`
    [*Overview of the Third BUCC Shared Task: Spotting Parallel Sentences in Comparable Corpora*](http://lrec-conf.org/workshops/lrec2018/W8/pdf/12_W8.pdf),
    LREC, 2018.

[2] Holger Schwenk,
    [*Filtering and Mining Parallel Data in a Joint Multilingual Space*](https://arxiv.org/abs/1805.09822),
    ACL, July 2018

[3] Mikel Artetxe and Holger Schwenk,
    [*Margin-based Parallel Corpus Mining with Multilingual Sentence Embeddings*](https://arxiv.org/abs/1811.01136)
    arXiv, 3 Nov 2018.

[3] Mikel Artetxe and Holger Schwenk,
    [*Massively Multilingual Sentence Embeddings for Zero-Shot Cross-Lingual Transfer and Beyond*](https://arxiv.org/abs/1812.10464)
    arXiv, 26 Dec 2018.
