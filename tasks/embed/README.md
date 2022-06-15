# LASER: calculation of sentence embeddings

Tool to calculate sentence embeddings for an arbitrary text file:
```
bash ./embed.sh INPUT-FILE OUTPUT-FILE [LANGUAGE (ISO3)]
```

Requires download of the wmt22 models, see [tasks/wmt22/README.md](https://github.com/facebookresearch/LASER/tree/main/tasks/wmt22).
The input will first be tokenized, and then sentence embeddings will be generated. If a `language` is specified, 
then `embed.sh` will look for a language-specific encoder (specified by a three-letter langauge code). Otherwise 
it will default to LASER2, which covers 93 languages (https://arxiv.org/pdf/1812.10464.pdf).

## Output format

The embeddings are stored in float32 matrices in raw binary format.
They can be read in Python by:
```
import numpy as np
dim = 1024
X = np.fromfile("my_embeddings.bin", dtype=np.float32, count=-1)                                                                          
X.resize(X.shape[0] // dim, dim)                                                                                                 
```
X is a N x 1024 matrix where N is the number of lines in the text file.
        
## Examples

In order to encode an input text in any of the 93 languages supported by LASER2 (e.g. afr, eng, fra):
```
./embed.sh input_file output_file
```

To use a language-specific encoder (if available), such as for example: Wolof, Hausa, or Oromo:
```
./embed.sh input_file output_file wol
```
```
./embed.sh input_file output_file hau
```
```
./embed.sh input_file output_file orm
```

