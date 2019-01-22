# LASER  Language-Agnostic SEntence Representations

LASER is a library to calculate and use multilingual sentence embeddings.

# Tatoeba multilingual test set

We provide here the test set for 112 languages as we have used in the paper [1].
This data is extracted from the [Tatoeba corpus](https://tatoeba.org/eng/), dated Saturday 2018/11/17.

For each languages, we have selected 1000 English sentences and their translations, if available.
Please check [this paper](https://arxiv.org/abs/1812.10464) for a description of the languages, their families and scripts as well as baseline results.

Please note that the English sentences are not identical for all language pairs.
This means that the results are not directly comparable across languages.  In particular,
the sentences tend to have less variety for several low-resource languages,
e.g. "Tom needed water", "Tom needs water", "Tom is getting water", ....

# License

Please see [here](https://tatoeba.org/eng/terms_of_use) for the license of the Tatoeba corpus.

# References

[1] Mikel Artetxe, Holger Schwenk,
    Massively Multilingual Sentence Embeddings for Zero-Shot Cross-Lingual Transfer and Beyond,
[arXiv Dec 26 2018](https://arxiv.org/abs/1812.10464)


