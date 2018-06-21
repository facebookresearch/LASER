# Text normalization before sentence embeddings

The sentence embeddings have been trained on the Europarl corpus which uses
somehow a formal language. In particular, contracted forms are almost never
used, e.g. *can't* or *I'll*.

Therefore we normalize the input text and map all contractions to the full
form (e.g. *cannot* or *I will*). This is only used for English.

We also provide some simple regular expressions to replace numbers by one
token. Those are only meant as a starting point for further experiments.
