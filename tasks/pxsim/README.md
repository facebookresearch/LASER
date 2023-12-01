# LASER: P-xSIM (dual approach multilingual similarity error rate)

This README shows how to calculate the P-xSIM error rate (Seamless Communication et al., 2023) for a given language pair.

P-xSIM returns the error rate for recreating gold alignments using a blended combination of two different approaches.
It works by performing a k-nearest-neighbor search and margin calculation (i.e. margin-based parallel alignment) using the
first approach, followed by the scoring of each candidate neighbor using an auxiliary model (the second approach). Finally,
the scores of both the margin-based alignment and the auxiliary model are combined together using a blended score defined as:

$$ \text{blended-score}(x, y) = \alpha \cdot \text{margin} + (1 - \alpha) \cdot \text{auxiliary-score} $$

where the parameter $\alpha$ controls the combination of both the margin-based and auxiliary scores. By default, the auxiliary-score will be calculated as the cosine between the source and candidate neighbors using the auxiliary embeddings. However, there is also an option to perform inference using a comparator model (Seamless Communication et al., 2023). In this instance, the auxiliary-score will be the AutoPCP outputs.

P-xSIM offers three margin-based scoring options (discussed in detail [here](https://arxiv.org/pdf/1811.01136.pdf)):
- distance
- ratio
- absolute

## Example usage

Simply run the example script `bash ./eval.sh` to download a sample dataset (flores200), sample encoders (laser2 and LaBSE),
and then perform P-xSIM. In this toy example, we use laser2 to provide the k-nearest-neighbors, followed by applying LaBSE as an
auxiliary model on each candidate neighbor, before then applying the blended scoring function defined above. Dependending on
your data sources, you may want to alter the approach used for either margin-based parallel alignment, or the scoring of each candidate neighbor
(i.e. the auxiliary model).

In addition to LaBSE in the example above, you can also calculate P-xSIM using any model hosted on [HuggingFace sentence-transformers](https://huggingface.co/sentence-transformers).
