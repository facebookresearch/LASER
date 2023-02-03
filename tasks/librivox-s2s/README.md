# Librivox S2S: Automatically mined Speech-to-Speech translations 

## Abstract

We present an approach to encode a speech signal into a fixed-size representation which minimizes the cosine loss with the existing massively multilingual LASER text embedding space. Sentences are close in this embedding space, independently of their language and modality, either text or audio. Using a similarity metric in that multimodal embedding space, we perform mining of audio in German, French, Spanish and English from [*Librivox*](https://librivox.org/) against billions of sentences from Common Crawl. This yielded more than twenty thousand hours of aligned speech translations. To evaluate the automatically mined speech/text corpora, we train neural speech translation systems for several languages pairs. Adding the mined data, achieves significant improvements in the BLEU score on the CoVoST2 and the MUST-C test sets with respect to a very competitive baseline. Our approach can also be used to directly perform speech-to-speech mining, without the need to first transcribe or translate the data. We obtain more than one thousand three hundred hours of aligned speech in French, German, Spanish and English. This speech corpus has the potential to boost research in speech-to-speech translation which suffers from scarcity of natural end-to-end training data.

## Download

Manifest files for all languages directions are available [*here*](https://dl.fbaipublicfiles.com/librivox_s2s/manifests.zip).
S2S alignments are sorted by decreasing mining scores (first column). Audios files for each language direction can be downloaded separately. For each language direction, we give the amount of aligned hours in the source and target language.

- [*English-French*](https://dl.fbaipublicfiles.com/librivox_s2s/ena-fra.zip) (470h / 447h)
- [*English-German*](https://dl.fbaipublicfiles.com/librivox_s2s/dea-ena.zip) (363h / 324h)
- [*English-Spanish*](https://dl.fbaipublicfiles.com/librivox_s2s/ena-esa.zip) (425h / 442h)
- [*French-German*](https://dl.fbaipublicfiles.com/librivox_s2s/dea-fra.zip) (33h / 38h)
- [*French-Spanish*](https://dl.fbaipublicfiles.com/librivox_s2s/esa-fra.zip) (101h / 111h)
- [*German-Spanish*](https://dl.fbaipublicfiles.com/librivox_s2s/dea-esa.zip) (41h / 40h)

The aligned Speech-to-Speech segments are distributed under the same copyright than [*Librivox*](https://librivox.org/).

Please cite reference [1] if you use this data.
The mined speech-to-speech data was successfully used to train Speech-to-Speech translation systems [2].


## References

[1] Paul-Ambroise Duquenne, Hongyu Gong, Holger Schwenk,
    [*Multimodal and Multilingual Embeddings for Large-Scale Speech Mining,*](https://papers.nips.cc/paper/2021/hash/8466f9ace6a9acbe71f75762ffc890f1-Abstract.html), NeurIPS 2021, pages 15748-15761.

[2] Ann Lee, Hongyu Gong, Paul-Ambroise Duquenne, Holger Schwenk, Peng-Jen Chen, Changhan Wang, Sravya Popuri, Juan Pino, Jiatao Gu, Wei-Ning Hsu,
    [*Textless Speech-to-Speech Translation on Real Data*](https://arxiv.org/abs/2112.08352), arXiv Dec 15 2021. to appear at NAACL'22.

