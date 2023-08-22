# LASER encoders

LASER Language-Agnostic SEntence Representations Toolkit

laser_encoders is the official Python package for the Facebook [LASER](https://github.com/facebookresearch/LASER) library. It provides a simple and convenient way to use LASER embeddings in Python. It allows you to calculate multilingual sentence embeddings using the LASER toolkit. These embeddings can be utilized for various natural language processing tasks, including document classification, bitext filtering, and mining.

## Dependencies

- Python >= 3.8
- [PyTorch](http://pytorch.org/)

## Installation

You can install laser_encoders using pip:

```sh
 pip install laser_encoders
```

## Downloading the pre-trained models

```sh
python laser_encoders download_models.py --lang=your_prefered_language #eg --lang="igbo"
```

This will by default download the models to the `~/.cache/laser_encoders` directory. 
To download the models to a specific location use 
```sh
python laser_encoders download_models.py --model-dir=path/to/model/directory
```
use the `--help` command to get the full list of various args that can be passed.

## Usage

The laser_encoders package provides an easy-to-use interface to calculate multilingual sentence embeddings. Here's a simple example of how you can use it:

```py
from laser_encoders.laser_tokenizer import LaserTokenizer
from laser_encoders.models import SentenceEncoder
from pathlib import Path

tokenizer = LaserTokenizer(spm_model=Path(path/to/spm_model))

tokenized_sentence = tokenizer.tokenize("This is a test sentence.")

encoder = SentenceEncoder(model_path=path/to/downloaded/model, spm_vocab=path/to/cvocab)
embeddings = encoder.encode_sentences([tokenized_sentence])
```
To tokenize a file use
```py
tokenized_sentence = tokenizer.tokenize_file(inp_fname=Path(path/to/input_file.txt), out_fname=Path(path/to/output_file.txt))
```
**Alternatively**, you can download and initialise the tokenizer and encoder with just one step
```py
from laser_encoders import initialize_encoder, initialize_tokenizer

# Load the LASER tokenizer
tokenizer = initialiize_tokenizer(lang="igbo")
tokenized_sentence = tokenizer.tokenize("nnọọ, kedu ka ị mere")

# Load the LASER sentence encoder
encoder = initialize_encoder(lang="igbo")

# Encode sentences into embeddings
embeddings = encoder.encode_sentences([tokenized_sentence])
```


## Now you can use these embeddings for downstream tasks

For more advanced usage and options, please refer to the official LASER repository documentation.

### Contributing

We welcome contributions from the developer community to enhance and improve laser_encoders. If you'd like to contribute, you can:

1. Submit bug reports or feature requests through GitHub issues.
1. Fork the repository, make changes, and submit pull requests for review.

### License

This package is released under the [LASER](https://github.com/facebookresearch/LASER/blob/main/LICENSE) Facebook BSD License.

### Contact

For any questions, feedback, or support, you can contact Facebook AI Research.

### Acknowledgments

This package is based on the [LASER](https://github.com/facebookresearch/LASER) project by Facebook AI Research.
