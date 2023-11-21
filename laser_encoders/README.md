# LASER encoders

LASER Language-Agnostic SEntence Representations Toolkit

laser_encoders is the official Python package for the Facebook [LASER](https://github.com/facebookresearch/LASER) library. It provides a simple and convenient way to use LASER embeddings in Python. It allows you to calculate multilingual sentence embeddings using the LASER toolkit. These embeddings can be utilized for various natural language processing tasks, including document classification, bitext filtering, and mining.

## Dependencies

- Python `>= 3.8`
- [PyTorch `>= 1.10.0`](http://pytorch.org/)
- sacremoses `>=0.1.0`
- sentencepiece `>=0.1.99`
- numpy `>=1.21.3`
- fairseq `>=0.12.2`

You can find a full list of requirements [here](https://github.com/facebookresearch/LASER/blob/main/pyproject.toml)

## Installation

You can install `laser_encoders` package from PyPI:

```sh
pip install laser_encoders
```

Alternatively, you can install it from a local clone of this repository, in editable mode:
```sh
pip install . -e
```

## Usage

Here's a simple example on how to obtain embeddings for sentences using the `LaserEncoderPipeline`:

>**Note:** By default, the models will be downloaded to the `~/.cache/laser_encoders` directory. To specify a different download location, you can provide the argument `model_dir=path/to/model/directory`

```py
from laser_encoders import LaserEncoderPipeline

# Initialize the LASER encoder pipeline
encoder = LaserEncoderPipeline(lang="igbo")

# Encode sentences into embeddings
embeddings = encoder.encode_sentences(["nnọọ, kedu ka ị mere"])
# If you want the output embeddings to be L2-normalized, set normalize_embeddings to True
normalized_embeddings = encoder.encode_sentences(["nnọọ, kedu ka ị mere"], normalize_embeddings=True)

```

If you prefer more control over the tokenization and encoding process, you can initialize the tokenizer and encoder separately:
```py
from laser_encoders import initialize_encoder, initialize_tokenizer

# Initialize the LASER tokenizer
tokenizer = initialize_tokenizer(lang="igbo")
tokenized_sentence = tokenizer.tokenize("nnọọ, kedu ka ị mere")

# Initialize the LASER sentence encoder
encoder = initialize_encoder(lang="igbo")

# Encode tokenized sentences into embeddings
embeddings = encoder.encode_sentences([tokenized_sentence])
```
>By default, the `spm` flag is set to `True` when initializing the encoder, ensuring the accompanying spm model is downloaded.

**Supported Languages:** You can specify any language from the [FLORES200](https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200) dataset. This includes both languages identified by their full codes (like "ibo_Latn") and simpler alternatives (like "igbo").

## Downloading the pre-trained models

If you prefer to download the models individually, you can use the following command:

```sh
python -m laser_encoders.download_models --lang=your_prefered_language  # e.g., --lang="igbo""
```

By default, the downloaded models will be stored in the `~/.cache/laser_encoders` directory. To specify a different download location, utilize the following command:

```sh
python -m laser_encoders.download_models --model-dir=path/to/model/directory
```

> For a comprehensive list of available arguments, you can use the `--help` command with the download_models script.

Once you have successfully downloaded the models, you can utilize the `SentenceEncoder` to tokenize and encode your text in your desired language. Here's an example of how you can achieve this:

```py
from laser_encoders.models import SentenceEncoder
from pathlib import Path

encoder = SentenceEncoder(model_path=path/to/downloaded/model, spm_model=Path(path/to/spm_model), spm_vocab=path/to/cvocab)
embeddings = encoder("This is a test sentence.")
```
If you want to perform tokenization seperately, you can do this below:
```py
from laser_encoders.laser_tokenizer import LaserTokenizer

tokenizer = LaserTokenizer(spm_model=Path(path/to/spm_model))

tokenized_sentence = tokenizer.tokenize("This is a test sentence.")

encoder = SentenceEncoder(model_path=path/to/downloaded/model, spm_vocab=path/to/cvocab)
embeddings = encoder.encode_sentences([tokenized_sentence])
```

For tokenizing a file instead of a string, you can use the following:

```py
tokenized_sentence = tokenizer.tokenize_file(inp_fname=Path(path/to/input_file.txt), out_fname=Path(path/to/output_file.txt))
```

### Now you can use these embeddings for downstream tasks

For more advanced usage and options, please refer to the official LASER repository documentation.

## LASER Versions and Associated Packages

For users familiar with the earlier version of LASER, you might have encountered the [`laserembeddings`](https://pypi.org/project/laserembeddings/) package. This package primarily dealt with LASER-1 model embeddings.

For the latest LASER-2,3 models, use the newly introduced `laser_encoders` package, which offers better performance and support for a wider range of languages.


## Contributing

We welcome contributions from the developer community to enhance and improve laser_encoders. If you'd like to contribute, you can:

1. Submit bug reports or feature requests through GitHub issues.
1. Fork the repository, make changes, and submit pull requests for review.

Please follow our [Contribution Guidelines](https://github.com/facebookresearch/LASER/blob/main/CONTRIBUTING.md) to ensure a smooth process.

### Code of Conduct

We expect all contributors to adhere to our [Code of Conduct](https://github.com/facebookresearch/LASER/blob/main/CODE_OF_CONDUCT.md).

### Contributors

The following people have contributed to this project:

- [Victor Joseph](https://github.com/CaptainVee)
- [Paul Okewunmi](https://github.com/Paulooh007)
- [Siddharth Singh Rana](https://github.com/NIXBLACK11)
- [David Dale](https://github.com/avidale/)
- [Holger Schwenk](https://github.com/hoschwenk)
- [Kevin Heffernan](https://github.com/heffernankevin)

### License

This package is released under the [LASER](https://github.com/facebookresearch/LASER/blob/main/LICENSE) BSD License.

