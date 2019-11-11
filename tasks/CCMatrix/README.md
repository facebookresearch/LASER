# CCMatrix: Mining Billions of High-Quality Parallel Sentences on the WEB

## Parallel data

We show that margin-based bitext mining in LASER's multilingual sentence space can be applied to monolingual corpora of billions of sentences.  We are using ten snapshots of a curated common crawl corpus [1] totaling 32.7 billion unique sentences.  Using one unified approach for 38 languages, we were able to mine 3.5 billions parallel sentences, out of which 661 million are aligned with English.  17 language pairs have more then 30 million parallel sentences, 82 more then 10 million, and most more than one million, including direct alignments between many European or Asian languages.

This [*table shows the amount of mined parallel sentences for most of the language pairs (all sizes in million sentences)*](MatrixMine.pdf)


## Download

* You need to download first the 
  [*CCNet corpus*](https://github.com/facebookresearch/cc_net) corpus [1]
* **We will soon provide a script to extract the parallel data from this corpus.  Please be patient**

Please cite reference [2] if you use this data.


## Evaluation

To evaluate the quality of the mined bitexts, we train NMT systems for most of the language pairs and evaluate them on TED, WMT and WAT test sets. Using our mined bitexts only and no human translated parallel data, we achieve a new state-of-the-art for a single system on the WMT'19 test set for translation between English and German, Russian and Chinese, as well as German/French. In particular, our English/German system outperforms the best single one by close to 4 BLEU points and is almost on pair with best WMT'19 evaluation system which uses system combination and back-translation.  We also achieve excellent results for distant languages pairs like Russian/Japanese, outperforming the best submission at the 2019 workshop on Asian Translation (WAT).


## References

[1] Guillaume Wenzek, Marie-Anne Lachaux, Alexis Conneau, Vishrav Chaudhary, Francisco Guzmán, Armand Jouli and Edouard Grave,
    [*CCNet: Extracting High Quality Monolingual Datasets from Web Crawl Data*](https://arxiv.org/abs/1911.00359)

[2] Holger Schwenk, Guillaume Wenzek, Sergey Edunov, Edouard Grave and Armand Joulin,
    [*CCMatrix: Mining Billions of High-Quality Parallel Sentences on the WEB*](https://arxiv.org/abs/xxx.yyy)
    **THE PAPER WILL BE AVAILABLE ON MONDAY 11/11 8pm EST**
