# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# LASER  Language-Agnostic SEntence Representations
# is a toolkit to calculate multilingual sentence embeddings
# and to use them for document classification, bitext filtering
# and mining
#
# --------------------------------------------------------
#
# This module contains all functions for binarization of text data

import mlenc_const as const

###############################################################################
#
# Load the hash table from file
# Format:  index (1..N), word, frequency count
#
# IMPORTANT: we will subtract 1 from all read inidices
#
###############################################################################


def LoadHashTable(htable_file, verbose=0):
    tok2idx = {}
    idx2tok = {}
    idx2cnt = {}
    if verbose & const.VERBOSE_LOAD > 0:
        print("Reading binarization hashtable from file '%s'"
              % htable_file.name)
    for line in htable_file:
        toks = line.split()     # array with idx, word, cont
        idx = int(toks[0]) - 1  # IMPORTANT: Python indices start at 0 !!!
        word = toks[1]
        idx2tok[idx] = word
        tok2idx[word] = idx
        idx2cnt[word] = toks[2]  # count
    if verbose & const.VERBOSE_LOAD > 0:
        print(" - read %d entries" % len(idx2tok))

    return tok2idx


###############################################################################
#
# Function to binarize provided data
#
###############################################################################

def BinarizeText(htable, data, order="left-to-right"):
    # resize to current bsize
    slen_max = data.text_bin.shape[0]
    bsize = len(data.text_bpe)
    data.text_bin.resize(slen_max, bsize)
    data.text_slen.resize(bsize)
    data.text_bin.fill(data.idx_pad)

    b = 0
    for line in data.text_bpe:
        words = line.split()
        data.text_slen[b] = len(words)
        if data.text_slen[b] > slen_max:
            # too long text will be a line with all <PAD>
            print("\r - WARNING skipping line with %d words"
                  % data.text_slen[b])
        else:
            w = 0
            offs = slen_max if order == "right-to-left" else data.text_slen[b]
            for word in words:
                if word in htable:
                    idx = htable[word]
                else:
                    idx = data.idx_unk
                data.text_bin[offs-1-w][b] = idx
                w += 1
        b += 1
