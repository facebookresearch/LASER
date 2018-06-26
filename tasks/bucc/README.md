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

Here, we apply this idea to the data provided by the shared task of the BUCC 
[Workshop on Building and Using Comparable Corporo](https://comparable.limsi.fr/bucc2018/bucc2018-task.html).
We provide results on the official language pairs English/French and
English/German, respectively.
In addition, we use the same system to extract French/German parallel sentences.

The same approach can be scaled up to huge collections of monolingual texts
(several billions) using more advanced features of the FAISS toolkit.

## Installation

* Please first download the BUCC shared task data
  [here](https://comparable.limsi.fr/bucc2017/cgi-bin/download-data-2018.cgi)
  and install it the directory "downloaded"
* Calculate the multilingual sentence embeddings for all languages by
  running the script
```bash
./bucc_embed.sh
```
* Calculate all pairwise distance and extract parallel data (this can take 1-2h)
```bash
./bucc_mine.sh
```

## Results

Optimized on F-score on the training corpus
(these results are slight improved with repect to [2]):

| Languages | Threshold | precision | Recall | F-score |
|-----------|-----------|-----------|--------|---------|
|   fr-en   |   0.519   |   81.85   |  69.14 |  74.96  |
|   de-en   |   0.499   |   82.71   |  70.56 |  76.16  |


Below, we compare our approach to the [official results of the 2018 edition
of the BUCC workshop](http://lrec-conf.org/workshops/lrec2018/W8/pdf/12_W8.pdf) [1].
More details on our approach are provided in [2].

|    System | en-fr | en-de |
|-----------|-------|-------|
|   VIC'17  |   79  |   84  |
|   RALI'17 |   20  |    -  |
|  LIMSI'17 |    -  |    -  |
|  VIC'18   |   81  |   86  |
|   H2'18   |   76  |    -  |
|   LASER   |  75.8 |  76.9 |

All numbers are F1-scores on the test set.

Mining parallel data solely based on a threshold of the multilingual distance
seems to work quite well. However, we have realized that sentences which
contain mainly enumerations of named entities are wrongly considered to be
mutual translations, although the named entities do not correspond.
Post processing to exclude those sentences is left for future research.

## References

[1] Pierre Zweigenbaum, Serge Sharoff and Reinhard Rapp,`
    [*Overview of the Third BUCC Shared Task: Spotting Parallel Sentences in Comparable Corpora*](http://lrec-conf.org/workshops/lrec2018/W8/pdf/12_W8.pdf),
    LREC, 2018.

[2] Holger Schwenk,
    [*Filtering and Mining Parallel Data in a Joint Multilingual Space*](https://arxiv.org/abs/1805.09822),
    ACL, July 2018
