#!/usr/bin/python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# LASER  Language-Agnostic SEntence Representations
# is a toolkit to calculate multilingual sentence embeddings
# and to use them for document classification, bitext filtering
# and mining
#
# --------------------------------------------------------
#
# Python tool to search for paraphrases in FAISS index

import re
import sys
import os.path
import tempfile
import argparse
import faiss
import time
import pdb
import numpy as np
from collections import namedtuple

# get environment
assert os.environ.get('LASER'), 'Please set the enviornment variable LASER'
LASER = os.environ['LASER']

sys.path.append(LASER + '/source/lib')
from indexing import IndexLoad, IndexTextOpen, IndexTextQuery, SplitOpen, SplitAccess
from embed import SentenceEncoder, EncodeLoad, EncodeFile, EncodeTime
from text_processing import Token, BPEfastApply

SPACE_NORMALIZER = re.compile("\s+")
Batch = namedtuple('Batch', 'srcs tokens lengths')

# calculate L2 distance between [x]
# and the vectors referenced in idxs
# x should be already normalized
def IndexDistL2(X, E, D, I, thresh=1.0, dtype=np.float32, sort=True):
    nb, nK = I.shape
    dim = X.shape[1]
    dist_l2 = np.empty((nb, nK), dtype=np.float32)
    y = np.empty((1, dim), dtype=dtype)
    for i in range(nb):
        for k in range(nK):
            if D[i, k] <= thresh:
                # get embedding from disk
                np.copyto(y, SplitAccess(E, I[i, k]))
                faiss.normalize_L2(y)
                dist_l2[i, k] = 1.0 - np.dot(X[i], y[0])
            else:
                # exclude sentences which already have a huge FAISS distance
                # (getting embeddings from disk is very time consumming)
                dist_l2[i, k] = 1.0

        if sort:
            # re-sort according to L2
            idxs = np.argsort(dist_l2[i], axis=0)
            dist_l2[i] = dist_l2[i][idxs]
            I[i] = I[i][idxs]

    return dist_l2, I

###############################################################################
#
# Apply an absolute threshold on the distance
#
###############################################################################

def MarginAbs(em, ofp, params, args, stats):
    D, I = params.idx.search(em, args.kmax)
    thresh = args.threshold_faiss
    if args.embed:
        D, I = IndexDistL2(em, params.E, D, I, args.threshold_faiss)
        thresh = args.threshold_L2

    for n in range(D.shape[0]):

        prev = {}  # for deduplication
        for i in range(args.kmax):
            txt = IndexTextQuery(params.T, params.R, I[n, i])
            if (args.dedup and txt not in prev) and D[n, i] <= thresh:
                prev[txt] = 1
                ofp.write('{:d}\t{:7.5f}\t{}\n'
                          .format(stats.nbs, D[n, i], txt))
                stats.nbp += 1

        # display source sentece if requested
        if (args.include_source == 'matches' and len(prev) > 0):
            ofp.write('{:d}\t{:6.1f}\t{}\n'
                      .format(stats.nbs, 0.0, sentences[n].replace('@@ ', '')))

        if args.include_source == 'always':
            ofp.write('{:d}\t{:6.1f}\t{}\n'
                      .format(stats.nbs, 0.0, sentences[n].replace('@@ ', '')))
        stats.nbs += 1


###############################################################################
#
# Apply an threshold on the ratio between distance and average
#
###############################################################################

def MarginRatio(em, ofp, params, args, stats):
    D, I = params.idx.search(em, args.margin_k)
    thresh = args.threshold
    if args.embed:
        D, I = IndexDistL2(em, params.E, D, I, args.threshold_faiss)
        thresh = args.threshold_L2

    Mean = D.mean(axis=1)
    for n in range(D.shape[0]):
        if D[n, 0] / Mean[n] <= args.threshold:
            if args.include_source == 'matches':
                ofp.write('{:d}\t{:6.1f}\t{}\n'
                          .format(stats.nbs, 0.0, sentences[n].replace('@@ ', '')))
            txt = IndexTextQuery(params.T, params.R, I[n, 0])
            ofp.write('{:d}\t{:7.5f}\t{}\n'.format(stats.nbs, D[n, 0], txt))
            stats.nbp += 1

        stats.nbs += 1

    if args.include_source == 'always':
        ofp.write('{:d}\t{:6.1f}\t{}\n'
                  .format(stats.nbs, 0.0, sentences[n].replace('@@ ', '')))


###############################################################################

def MarginDist(em, ofp, params, args, stats):
    print('ERROR: MarginAbs not implemented')
    sys.exit(1)


###############################################################################

def buffered_read(fp, buffer_size):
    buffer = []
    for src_str in fp:
        buffer.append(src_str.strip())
        if len(buffer) >= buffer_size:
            yield buffer
            buffer = []

    if len(buffer) > 0:
        yield buffer


###############################################################################

parser = argparse.ArgumentParser('LASER: paraphrase tool')

parser.add_argument('--encoder', type=str, required=True,
    help='encoder to be used')
parser.add_argument('--encoding', default='utf-8',
    help='Character encoding for input/output')
parser.add_argument('--token-lang', type=str, default='--',
    help="Language of tokenizer ('--' for no tokenization)")
parser.add_argument('--bpe-codes', type=str, default=None, required=True,
    help='BPE codes')
parser.add_argument('--buffer-size', type=int, default=100,
    help='Buffer size (sentences)')
parser.add_argument('--max-tokens', type=int, default=12000,
    help='Maximum number of tokens to process in a batch')
parser.add_argument('--max-sentences', type=int, default=None,
    help='Maximum number of sentences to process in a batch')
parser.add_argument('--cpu', action='store_true',
    help='Use CPU instead of GPU')

parser.add_argument('--index', type=str, required=True,
    help='FAISS index')
parser.add_argument('--nprobe', type=int, default=128,
    help='FAISS: value of nprobe')
parser.add_argument('--text', type=str, required=True,
    help='File with indexed texts')
parser.add_argument(
    '--dim', type=int, default=1024,
    help='Dimension of specified sentence embeddings')
parser.add_argument(
    '--embed', type=str, default=None,
    help='Sentence embeddings, true L2 distance will be calculated when specified')

parser.add_argument('-i', '--input', type=str, required=True,
    help='Input text file')
parser.add_argument('-p', '--output', type=str, default='--',
    help='Output paraphrases')
parser.add_argument('--kmax', type=int, default=10,
    help='Max value of distance or margin of each paraphrase')
parser.add_argument('--dedup', type=int, default=1,
    help='Deduplicate list of paraphrases')
parser.add_argument('--include-source', default='never',
    choices=['never', 'matches', 'always'],
    help='Include source sentence in the list of paraphrases')
parser.add_argument('--margin',
    choices=['absolute', 'distance', 'ratio'],
    default='ratio', help='Margin function')
parser.add_argument('-T', '--threshold-margin', type=float, default=0.9,
    help='Threshold on margin')
parser.add_argument('--threshold-faiss', type=float, default=0.4,
    help='Threshold on FAISS distance')
parser.add_argument('--threshold-L2', type=float, default=0.2,
    help='Threshold on L2 distance')
parser.add_argument('--margin-k', type=int, default=4,
    help='Number of nearest neighbors for margin calculation')

parser.add_argument('--verbose', action='store_true',
    help='Detailed output')


print('\nLASER: paraphrase tool')
args = parser.parse_args()

# index,
# memory mapped texts, references and word counts
# encoder
params = namedtuple('params', 'idx T R W M E enc')

# open text and reference file
params.T, params.R, params.W, params.M = IndexTextOpen(args.text)

# Open on-disk embeddings for L2 distances
if args.embed:
    params.E = SplitOpen(args.embed, ['en'],
                                args.dim, np.float32, verbose=False)

# load FAISS index
params.idx = IndexLoad(args.index, args.nprobe)

# load sentence encoder
params.enc = EncodeLoad(args)


margin_methods = {'absolute': MarginAbs,
                  'distance': MarginDist,
                  'ratio': MarginRatio}

with tempfile.TemporaryDirectory() as tmpdir:
    ifile = args.input
    if args.token_lang != '--':
        ifile = os.path.join(tmpdir, 'tok')
        Token(args.input,
              ifile,
              lang=args.token_lang,
              romanize=True if args.token_lang == 'el' else False,
              lower_case=True, gzip=False,
              verbose=args.verbose, over_write=False)

    if args.bpe_codes:
        bpe_file = os.path.join(tmpdir, 'bpe')
        BPEfastApply(ifile,
                     bpe_file,
                     args.bpe_codes,
                     verbose=args.verbose, over_write=False)
        ifile = bpe_file

    print(' - processing (batch size is {:d})'.format(args.buffer_size))
    ifp = open(ifile, 'r', encoding=args.encoding, errors='surrogateescape')
    if args.output == '--':
        ofp = sys.stdout
    else:
        ofp = open(args.output, 'w', encoding=args.encoding, errors='surrogateescape')
    stats = namedtuple('stats', 'ns np')
    stats.nbs = 0
    stats.nbp = 0
    t = time.time()
    for sentences in buffered_read(ifp, args.buffer_size):
        embed = params.enc.encode_sentences(sentences)
        faiss.normalize_L2(embed)
        # call function for selected margin method
        margin_methods.get(args.margin)(embed, ofp, params, args, stats)
        if stats.nbs % 1000 == 0:
            print('\r - {:d} sentences {:d} paraphrases'
                  .format(stats.nbs, stats.nbp), end='')

    ifp.close()
    if args.output != '--':
        ofp.close()
    print('\r - {:d} sentences {:d} paraphrases'
          .format(stats.nbs, stats.nbp), end='')
    EncodeTime(t)
