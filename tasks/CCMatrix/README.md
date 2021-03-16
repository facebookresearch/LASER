# CCMatrix: Mining Billions of High-Quality Parallel Sentences on the WEB

## Parallel data

We show that margin-based bitext mining in LASER's multilingual sentence space can be applied to monolingual corpora of billions of sentences to produce high quality aligned translation data. We use thirty-two snapshots of a curated common crawl corpus [1] totaling 69 billion unique sentences. Using one unified approach for 80 languages, we were able to mine 10.8 billion parallel sentences, out of which only 2.9 billion are aligned with English. 

## Download

We open-source our scripts in this directory so that others may reproduce the data, evaluation and results reported in the CCMatrix paper.
```
pip3 install cc_net
python3 dl_cc_matrix.py
```

Please cite reference [2][3] if you use this data.


## Evaluation

Evaluation
We have assessed the quality of our mined data with bilingual models and multilingual models.

* Bilingual models [2]:  To evaluate the quality of the mined bitexts, we train NMT systems for most of the language pairs and evaluate them on TED, WMT and WAT test sets. Using our mined bitexts only and no human translated parallel data, we achieve a new state-of-the-art for a single system on the WMT'19 test set for translation between English and German, Russian and Chinese, as well as German/French. In particular, our English/German system outperforms the best single one by close to 4 BLEU points and is almost on pair with best WMT'19 evaluation system which uses system combination and back-translation. We also achieve excellent results for distant languages pairs like Russian/Japanese, outperforming the best submission at the 2019 workshop on Asian Translation (WAT).

* Multilingual models [3]:  CCMatrix data is used to train M2M-100, a large-scale Many-to-Many multilingual translation model. The thousands of directions we mine produce training data for direct translations without relying solely on English data. We mine using novel strategy which exploits language groupings and bridge languages to avoid mining every possible direction while maintaining good accuracy. By training on this data and scaling model capacity through model parallelism and language-specific parameters, M2M-100 outperforms English-Centric multilingual models trained on data where either the source or target language is English. The system improves over 10 BLEU on average compared to an English-Centric baseline when translating directly between non-English directions. M2M-100 is competitive to bilingual models from WMT and improves over existing publicly available multilingual translation systems. To download the data, follow our instructions above. To download the models and reproduce the training, click [*here*](https://github.com/pytorch/fairseq/tree/master/examples/m2m_100)

Please note that additional data filtering was applied before training the M2M-100 model, see [3] for details.
Also, we have improved mining against English which leads to more bitexts, in particular for mid- and low-resources languages.
This new data was not used for M2M-100.

## References

[1] Guillaume Wenzek, Marie-Anne Lachaux, Alexis Conneau, Vishrav Chaudhary, Francisco Guzmán, Armand Jouli and Edouard Grave,
    [*CCNet: Extracting High Quality Monolingual Datasets from Web Crawl Data*](https://arxiv.org/abs/1911.00359)

[2] Holger Schwenk, Guillaume Wenzek, Sergey Edunov, Edouard Grave and Armand Joulin,
    [*CCMatrix: Mining Billions of High-Quality Parallel Sentences on the WEB*](https://arxiv.org/abs/1911.04944)
    
[3] Angela Fan, Shruti Bhosale, Holger Schwenk, Zhiyi Ma, Ahmed El-Kishky, Siddharth Goyal, Mandeep Baines, Onur Celebi, Guillaume Wenzek, Vishrav Chaudhary, Naman Goyal, Tom Birch, Vitaliy Liptchinsky, Sergey Edunov, Edouard Grave, Michael Auli, and Armand Joulin. Beyond English-Centric Multilingual Machine Translation
