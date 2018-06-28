# LASER: application to cross-lingual document classification

This codes shows how to use multilingual sentence embeddings for
cross-lingual document classification, using the Reuters RCV2 corpus.
This corpus was introduced in [1] and used in several works to
evaluate cross-lingual document classification.

However, one drawback of this corpus are very unbalanced class-prior
probabilities and we recommend to rather use the
[MLDoc corpus](https://github.com/facebookresearch/MLDoc) [5]

We train a  document classifier on one language (e.g. English) and apply it then
to several other languages without using any resource of that language
(e.g. German, French and Spanish)

## Installation

* The Reuters RCV2 corpus uses a particular copyright and must be obtained
  [here](https://trec.nist.gov/data/reuters/reuters.html).
  The subsets used in the paper by Klementiev et al [1] and follow-up work
  can be obtained directly [from the authors](mailto:titovian@gmail.com)
* Calculate the multilingual sentence embeddings for all languages by
  running the script (please adapt the paths in the beginning of the script
  to your installation)
```bash
./reuters_embed.sh
```
* Train the classifier with the optimized settings
```bash
./reuters_train.sh && ./reuters_ana.sh
```
  it is also possible to rerun the optimization step if you want to try
  different settings (function OptimizeZeroShot())

## Results

We use a simple MLP classifier with one hidden layer and Adam optimization.

You should get the following results for zero-short cross-lingual transfer
(which can be displayed with the script ./mldoc\_ana.sh)
We also provide the published results of three other approaches.
To the best of our knowledge, those represent the current state-of-the-art on
the Reuters RCV2 corpus.

| Transfer direction | BAE-cr [2] | BRAVE-S [3] | PARA\_DOC [4] |   LASER   |
|--------------------|------------|-------------|---------------|-----------|
|    en -> de        |    91.8%   |    89.7%    |      92.7%    | **92.9%** |
|    de -> en        |    74.2%   |    80.1%    |    **91.5%**  |   83.0%   |
|    en -> fr        |  **84.6%** |    82.5%    |        -      |   73.4%   |
|    fr -> en        |    74.2%   |  **79.5%**  |        -      |   78.2%   |
|    en -> es        |    49.0%   |    60.2%    |        -      | **72.8%** |
|    es -> en        |    64.4%   |    70.4%    |        -      | **73.2%** |
 
All numbers are accuracies on the test set
(this corpus does not provide a development corpus for parameter tuning)


## References

[1] A. Klementiev and I. Titov and B. Bhattarai,
    Inducing Crosslingual Distributed Representations of Words,
    COLING, 2012.

[2] Sarath Chandar, Mitesh M. Khapra, Balaraman Ravindran, Vikas Raykar and Amrita Saha,
    Multilingual Deep Learning, NIPS deep learnign workshop, 2013.

[3] Aditua Mogadala and Achim Rettinger,
    Bilingual Word Embeddings from Parallel and Non-Parallel Corpora for Cross-Language Classification,
    NAACl, pages 692--702, 2016.

[4] Hieu Pham, Minh-Thang Luong and Christopher D. Manning,
    Learning Distributed Representations for Multilingual Text Sequences,
    Workshop on Vector Space Modeling for NLP, 2015.

[5] [*A Corpus for Multilingual Document Classification in Eight Languages*](http://www.lrec-conf.org/proceedings/lrec2018/pdf/658.pdf),
    LREC, pages 3548-3551, 2018.

