# LASER: application to cross-lingual document classification

This codes shows how to use the multilingual sentence embedding for
cross-lingual document classification, using the MLDoc corpus.

We train a  document classifier on one language (e.g. English) and apply it then
to several other languages without using any resource of that language
(e.g. French, German, Spanish and Italian)

## Installation

* Please first download the MLDoc corpus from
  [here](https://github.com/facebookresearch/MLDoc)
  and install it in the directory MLDoc
* Calculate the multilingual sentence embeddings for all languages by
  running the script (please adapt the paths in the beginning of the script
  to your installation)
```bash
./mldoc_embed.sh
```
* Train the classifier with the optimized settings
```bash
./mldoc_train.sh && ./mldoc_ana.sh
```
  It is also possible to rerun the optimization step if you want to try
  different settings (function OptimizeZeroShot())

## Results

We use a simple MLP classifier with one hidden layer and Adam optimization.

You should get the following results for zero-short cross-lingual transfer
(which can be displayed with the script ./mldoc\_ana.sh)
These results are better than those reported in [1] since the system has
been improved since publication.

| Train language |   Dev  |   De   |   En   |   Es   |   Fr   |   It   |
|----------------|--------|--------|--------|--------|--------|--------|
| German (de)    | 93.60% | 91.50% | 78.40% | 75.05% | 73.88% | 68.28% |
| English (en)   | 88.70% | 83.20% | 88.25% | 74.18% | 78.18% | 68.55% |
| Spanish (es)   | 84.70% | 75.33% | 67.55% | 85.38% | 72.00% | 65.18% |
| French (fr)    | 90.60% | 83.53% | 77.93% | 76.53% | 89.03% | 67.93% |
| Italian (it)   | 83.30% | 81.08% | 74.35% | 78.70% | 77.18% | 84.98% |

All numbers are accuracies on the test set
(except the column "Dev")


## References

Details on the corpus are described in this paper:

[1]  Holger Schwenk and Xian Li,
    [*A Corpus for Multilingual Document Classification in Eight Languages*](http://www.lrec-conf.org/proceedings/lrec2018/pdf/658.pdf),
    LREC, pages 3548-3551, 2018.

```
@InProceedings{Schwenk:2018:lrec_mldoc,
  author = {Holger Schwenk and Xian Li},
  title = {A Corpus for Multilingual Document Classification in Eight Languages},
  booktitle = {LREC},,
  pages = {3548--3551},
  year = {2018}
}
```
