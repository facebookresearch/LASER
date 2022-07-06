# LASER: xSIM (multilingual similarity search)

This README shows how to calculate the xsim (multilingual similarity) error rate for a given language pair.

xSIM returns the error rate for encoding bitexts into the same embedding space i.e., given a bitext 
with source language embeddings X, and target language embeddings Y, xSIM aligns the embeddings from 
X and Y based on a margin-based similarity, and then returns the percentage of incorrect alignments.

xSIM offers three margin-based scoring options (discussed in detail [here](https://arxiv.org/pdf/1811.01136.pdf)):
- distance
- ratio
- absolute

## Example usage

### Sample script

Simply run the example script `bash ./eval.sh` to download a sample dataset (flores200), a sample encoder (laser2), 
and calculate the sentence embeddings and the xSIM error rate for a set of (comma separated) languages.

You can also calculate xsim for encoders hosted on [HuggingFace sentence-transformers](https://huggingface.co/sentence-transformers). For example, to use LaBSE you can modify/add the following arguments in the sample script:
```
--src-encoder LaBSE
--use-hugging-face
--embedding-dimension 768
```
Note: for HuggingFace encoders there is no need to specify `--src-spm-model`.

### Python

Import xsim

```
from xsim import xSIM
```
Calculate xsim from either numpy float arrays (e.g. np.float32) or binary embedding files
```
# A: numpy arrays x and y

err, nbex = xSIM(x, y)

# B: binary embedding files x and y

fp16_flag = False     # set true if embeddings are saved in 16 bit
embedding_dim = 1024  # set dimension of saved embeddings
err, nbex = xSIM(
  x, 
  y, 
  dim=embedding_dim, 
  fp16=fp16_flag
)
```
Error type
```
# A: textual-based error (allows for duplicates)

tgt_text = "/path/to/target-text-file"
err, nbex = xSIM(x, y, eval_text=tgt_text)

# B: index-based error (default)

err, nbex = xSIM(x, y)
```
Margin selection
```
# A: ratio (default)
err, nbex = xSIM(x, y)

# B: distance
err, nbex = xSIM(x, y, margin='distance')

# C: absolute
err, nbex = xSIM(x, y, margin='absolute')
```
Finally, to calculate the error rate simply return: `100 * err / nbex` (number of errors over total examples).
