# LASER encoders

LASER Language-Agnostic SEntence Representations Toolkit

laser_encoders is the official Python package for the Facebook [LASER](https://github.com/facebookresearch/LASER) library. It provides a simple and convenient way to use LASER embeddings in Python. It allows you to calculate multilingual sentence embeddings using the LASER toolkit. These embeddings can be utilized for various natural language processing tasks, including document classification, bitext filtering, and mining.

## Dependencies

- Python `>= 3.8`
- [PyTorch `>= 2.0`](http://pytorch.org/)
- sacremoses `>=0.0.53`
- sentencepiece `>=0.1.99`
- numpy `>=1.25.0`
- fairseq `>=0.12.2`

You can find a full list of requirements [here](requirements.txt)

## Installation

You can install laser_encoders using pip:

```sh
 pip install laser_encoders
```

## Usage

Here's a simple example of how you can download and initialise the tokenizer and encoder with just one step.

**Note:** By default, the models will be downloaded to the` ~/.cache/laser_encoders` directory. To specify a different download location, you can provide the argument `model_dir=path/to/model/directory` to the initialize_tokenizer and initialize_encoder functions

```py
from laser_encoders import initialize_encoder, initialize_tokenizer

# Initialize the LASER tokenizer
tokenizer = initialize_tokenizer(lang="igbo")
tokenized_sentence = tokenizer.tokenize("nnọọ, kedu ka ị mere")

# Initialize the LASER sentence encoder
encoder = initialize_encoder(lang="igbo")

# Encode sentences into embeddings
embeddings = encoder.encode_sentences([tokenized_sentence])
```

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

Once you have successfully downloaded the models, you can utilize the `LaserTokenizer` to tokenize text in your desired language. Here's an example of how you can achieve this:

```py
from laser_encoders.laser_tokenizer import LaserTokenizer
from laser_encoders.models import SentenceEncoder
from pathlib import Path

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

## Contributing

We welcome contributions from the developer community to enhance and improve laser_encoders. If you'd like to contribute, you can:

1. Submit bug reports or feature requests through GitHub issues.
1. Fork the repository, make changes, and submit pull requests for review.

Please follow our [Contribution Guidelines](https://github.com/facebookresearch/LASER/blob/main/CONTRIBUTING.md) to ensure a smooth process.

### Code of Conduct

We expect all contributors to adhere to our [Code of Conduct](https://github.com/facebookresearch/LASER/blob/main/CODE_OF_CONDUCT.md).

### Contributors

The following people have contributed to this project:

- [CaptainVee](https://github.com/CaptainVee)

### License

This package is released under the [LASER](https://github.com/facebookresearch/LASER/blob/main/LICENSE) Facebook BSD License.

### Contact

For any questions, feedback, or support, you can contact Facebook AI Research.

### Acknowledgments

This package is based on the [LASER](https://github.com/facebookresearch/LASER) project by Facebook AI Research.

Special thanks to [heffernankevin](https://github.com/heffernankevin) and [avidale](https://github.com/avidale/), the maintainers of the LASER project for their valuable contributions and guidance.
