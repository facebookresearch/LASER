# LASER encoders
LASER Language-Agnostic SEntence Representations Toolkit

laser_encoders is a Python package that allows you to calculate multilingual sentence embeddings using the LASER toolkit. These embeddings can be utilized for various natural language processing tasks, including document classification, bitext filtering, and mining. It is the python wrapper for the [LASER](https://github.com/facebookresearch/LASER) repo


## Dependencies
* Python >= 3.8
* [PyTorch](http://pytorch.org/)

## Installation
You can install laser_encoders using pip:

```sh
 pip install laser_encoders
```

## Downloading the pre-trained models

```sh
python laser_encoders download_models.py 
```
This will by default download the models to the `~/.cache/laser_encoders` directory. Use python laser_encoders download_models.py path/to/model/directory to download the models to a specific location.


## Usage
The laser_encoders package provides an easy-to-use interface to calculate multilingual sentence embeddings. Here's a simple example of how you can use it:

```py
from laser_encoders import initialize_encoder, initialize_tokenizer

# Load the LASER sentence encoder
encoder = initialize_encoder()

# Encode sentences into embeddings
sentences = ["Hello, how are you?", "Bonjour, comment ça va?"]
embeddings = encoder.encode_sentences(sentences)
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
