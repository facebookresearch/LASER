# LASER: calculation of sentence embeddings

This codes shows how to calculate sentence embeddings for
an arbitrary text file:
```
bash ./embed.sh INPUT-FILE LANGUAGE OUTPUT-FILE
```
The input will be tokenized, using the mode of the specified language, BPE will be applied
and the sentence embeddings will be calculated.

## Output format

The embeddings are stored in float32 matrices in raw binary format.
They can be read in Python by:
```
import numpy as np
dim = 1024
X = np.fromfile("my_embeddings.raw", dtype=np.float32, count=-1)                                                                          
X.resize(X.shape[0] // dim, dim)                                                                                                 
```
X is a N x 1024 matrix where N is the number of lines in the text file.
        
## Example
```
./embed.sh ${LASER}/data/tatoeba/v1/tatoeba.fra-eng.fra fr my_embeddings.raw
```
