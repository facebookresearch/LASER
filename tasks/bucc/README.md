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
We provide results on all official language pairs French, Spanish, Russian and
Chinese paired with English, respectively.
In addition, we use the same system to extract French/German parallel sentences.

The same approach can be scaled up to huge collections of monolingual texts
(several billions) using more advanced features of the FAISS toolkit.

## Installation

* Please first download the BUCC shared task data
  [here](https://comparable.limsi.fr/bucc2017/cgi-bin/download-data-2018.cgi)
  and install it the directory "downloaded"
* run `./bucc.sh`
```
** UPDATED on January 22nd 2018 **

## References

[1] Pierre Zweigenbaum, Serge Sharoff and Reinhard Rapp,`
    [*Overview of the Third BUCC Shared Task: Spotting Parallel Sentences in Comparable Corpora*](http://lrec-conf.org/workshops/lrec2018/W8/pdf/12_W8.pdf),
    LREC, 2018.

[2] Holger Schwenk,
    [*Filtering and Mining Parallel Data in a Joint Multilingual Space*](https://arxiv.org/abs/1805.09822),
    ACL, July 2018
