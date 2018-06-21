# LASER  Language-Agnostic SEntence Representations

LASER is a library to calculate multilingual sentence embeddings.

Currently, we include an encoder which supports nine European languages:
* Germanic languages: English, German, Dutch, Danish
* Romanic languages: French, Spanish, Italian, Portuguese
* Uralic languages: Finish

All these languages are encoded by the same BLSTM encoder, and there is no need
to specify the input language (but tokenization is language specific).
According to our experience, the sentence encoder supports code-switching, i.e.
the same sentences can contain words in several different languages.

We have also some evidence that the encoder generalizes somehow to other
languages of the Germanic and Romanic language families (e.g. Swedish,
Norwegian, Afrikaans, Catalan or Corsican), although no data of these languages
was used during training.

A detailed description how the multilingual sentence embeddings are trained can
be found in [1,3].

## Dependencies
* Python 3 with [NumPy](http://www.numpy.org/)
* [PyTorch](http://pytorch.org/)
* [Faiss](https://github.com/facebookresearch/faiss) (for mining bitexts)
* tokenization from the Moses encoder and byte-pair-encoding

## Installation
* set the environment variable 'LASER' to the root of the installation, e.g.
  export LASER="${HOME}/projects/laser"
* download encoders from Amazon s3
* download third party software
```bash
./install_models.sh
./install_external_tools.sh
```
* download the data used in the examples tasks (see  description for each task)

## Applications

We showcase several applications of multilingual sentence embeddings
with code to reproduce our results (in the directory "tasks").

* Cross-lingual document classification using the
  [*Reuters*](https://github.com/fairinternal/mlenc/tree/master/tasks/reuters)
   and [*MLdoc*](https://github.com/fairinternal/mlenc/tree/master/tasks/mldoc) corpus [2]
* [*Mining parallel data*](https://github.com/fairinternal/mlenc/tree/master/tasks/bucc) in monolingual texts [3]
* [*Multilingual similarity search*](https://github.com/fairinternal/mlenc/tree/master/tasks/similarity)

For all tasks, we use exactly the same multilingual encoder, without any task specific optimization.

## License

This source code is licensed under the license found in the [`LICENSE`](LICENSE) file in the root directory of this source tree.

## References

[1] Holger Schwenk and Matthijs Douze,
    [*Learning Joint Multilingual Sentence Representations with Neural Machine Translation*](https://aclanthology.info/papers/W17-2619/w17-2619),
    ACL workshop on Representation Learning for NLP, 2017
```
@inproceedings{Schwenk:2017:repl4nlp,
  title={Learning Joint Multilingual Sentence Representations with Neural Machine Translation},
  author={Holger Schwenk and Matthijs Douze},
  booktitle={ACL workshop on Representation Learning for NLP},
  year={2017}
}
```

[2]  Holger Schwenk and Xian Li,
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

[3] Holger Schwenk,
    [*Filtering and Mining Parallel Data in a Joint Multilingual Space*](https://arxiv.org/abs/1805.09822),
    ACL, July 2018
