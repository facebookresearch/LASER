# LASER: xSIM++

This README shows how to calculate the xSIM++ error rate for a given language pair.

xSIM++ is an extension of [xSIM](https://github.com/facebookresearch/LASER/tree/main/tasks/xsim). In comparison to xSIM, this evaluates using target-side data with additional synthetic, hard-to-distinguish examples. You can find more details about it in the publication: [xSIM++: An Improved Proxy to Bitext Mining Performance for Low-Resource Languages](https://arxiv.org/abs/2306.12907).

## Example usage

Simply run the example script `bash ./eval.sh` to download a sample dataset (flores200), download synthetically augmented English evaluation data from Flores, a sample encoder (laser2), and calculate both the sentence embeddings and the xSIM++ error rate for a set of (comma separated) languages.

The evaluation command is similar to xSIM, however there is an additional option to provide the comma-separated list of augmented languages: `--tgt-aug-langs`. These refer
to languages in the chosen evaluation set which also have a separate augmented data file. In addition to the error rate, the script also provides a breakdown of the number of errors by type (e.g. incorrect entity/number etc.).

You can also calculate xsim++ for encoders hosted on [HuggingFace sentence-transformers](https://huggingface.co/sentence-transformers). For example, to use LaBSE you can modify/add the following arguments in the sample script:
```
--src-encoder LaBSE
--use-hugging-face
--embedding-dimension 768
```
Note: for HuggingFace encoders there is no need to specify `--src-spm-model`.
