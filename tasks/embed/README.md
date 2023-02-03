# LASER: calculation of sentence embeddings

Tool to calculate sentence embeddings for an arbitrary text file:
```
bash ./embed.sh INPUT-FILE OUTPUT-FILE [LANGUAGE]
```

The input will first be tokenized, and then sentence embeddings will be generated. If a `language` is specified, 
then `embed.sh` will look for a language-specific LASER3 encoder using the format: `{model_dir}/laser3-{language}.{version}.pt`. 
Otherwise it will default to LASER2 which covers the same 93 languages as [the original LASER encoder](https://arxiv.org/pdf/1812.10464.pdf).

**NOTE:** please set the model location (`model_dir` in `embed.sh`) before running. We recommend to download the models from the NLLB 
release (see [here](/nllb/README.md)). Optionally you can also select the model version number for downloaded LASER3 models. This currently defaults to: `1` (initial release).

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

In order to encode an input text in any of the 93 languages supported by LASER2 (e.g. Afrikaans, English, French):
```
./embed.sh input_file output_file
```

To use a language-specific encoder (if available), such as for example: Wolof, Hausa, or Irish:
```
./embed.sh input_file output_file wol_Latn
```
```
./embed.sh input_file output_file hau_Latn
```
```
./embed.sh input_file output_file gle_Latn
```

