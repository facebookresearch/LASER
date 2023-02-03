#!/usr/bin/python3
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
# Tool to calculate to embed a text file
# The functions can be also imported into another Python code

import os
import sys
import faiss
import argparse
import torch
import numpy as np

# get environment
assert os.environ.get('LASER'), 'Please set the enviornment variable LASER'
LASER = os.environ['LASER']

sys.path.append(LASER + '/source')
sys.path.append(LASER + '/source/tools')
from embed import SentenceEncoder, EncodeLoad, EncodeFile, EmbedLoad
from lib.text_processing import Token, BPEfastApply


###############################################################################
#
# Load texts and remove duplicates
#
###############################################################################

def TextLoadUnify(fname, args):
    if args.verbose:
        print(' - loading texts {:s}: '.format(fname), end='')
    fin = open(fname, encoding=args.encoding, errors='surrogateescape')
    inds = []
    sents = []
    sent2ind = {}
    n = 0
    nu = 0
    for line in fin:
        new_ind = len(sent2ind)
        inds.append(sent2ind.setdefault(line, new_ind))
        if args.unify:
            if inds[-1] == new_ind:
                sents.append(line[:-1])
                nu += 1
        else:
            sents.append(line[:-1])
            nu += 1
        n += 1
    if args.verbose:
        print('{:d} lines, {:d} unique'.format(n, nu))
    del sent2ind
    return inds, sents


###############################################################################
#
# Wrapper for knn on CPU/GPU
#
###############################################################################

def knn(x, y, k, use_gpu):
    return knnGPU(x, y, k) if use_gpu else knnCPU(x, y, k)


###############################################################################
#
# Perform knn on GPU
#
###############################################################################

def knnGPU(x, y, k, mem=5*1024*1024*1024):
    dim = x.shape[1]
    batch_size = mem // (dim*4)
    sim = np.zeros((x.shape[0], k), dtype=np.float32)
    ind = np.zeros((x.shape[0], k), dtype=np.int64)
    for xfrom in range(0, x.shape[0], batch_size):
        xto = min(xfrom + batch_size, x.shape[0])
        bsims, binds = [], []
        for yfrom in range(0, y.shape[0], batch_size):
            yto = min(yfrom + batch_size, y.shape[0])
            # print('{}-{}  ->  {}-{}'.format(xfrom, xto, yfrom, yto))
            idx = faiss.IndexFlatIP(dim)
            idx = faiss.index_cpu_to_all_gpus(idx)
            idx.add(y[yfrom:yto])
            bsim, bind = idx.search(x[xfrom:xto], min(k, yto-yfrom))
            bsims.append(bsim)
            binds.append(bind + yfrom)
            del idx
        bsims = np.concatenate(bsims, axis=1)
        binds = np.concatenate(binds, axis=1)
        aux = np.argsort(-bsims, axis=1)
        for i in range(xfrom, xto):
            for j in range(k):
                sim[i, j] = bsims[i-xfrom, aux[i-xfrom, j]]
                ind[i, j] = binds[i-xfrom, aux[i-xfrom, j]]
    return sim, ind


###############################################################################
#
# Perform knn on CPU
#
###############################################################################

def knnCPU(x, y, k):
    dim = x.shape[1]
    idx = faiss.IndexFlatIP(dim)
    idx.add(y)
    sim, ind = idx.search(x, k)
    return sim, ind


###############################################################################
#
# Scoring
#
###############################################################################

def score(x, y, fwd_mean, bwd_mean, margin):
    return margin(x.dot(y), (fwd_mean + bwd_mean) / 2)


def score_candidates(x, y, candidate_inds, fwd_mean, bwd_mean, margin, verbose=False):
    if verbose:
        print(' - scoring {:d} candidates'.format(x.shape[0]))
    scores = np.zeros(candidate_inds.shape)
    for i in range(scores.shape[0]):
        for j in range(scores.shape[1]):
            k = candidate_inds[i, j]
            scores[i, j] = score(x[i], y[k], fwd_mean[i], bwd_mean[k], margin)
    return scores


###############################################################################
#
# Main
#
###############################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LASER: Mine bitext')
    parser.add_argument('src',
        help='Source language corpus')
    parser.add_argument('trg',
        help='Target language corpus')
    parser.add_argument('--encoding', default='utf-8',
        help='Character encoding for input/output')
    parser.add_argument('--src-lang', required=True,
        help='Source language id')
    parser.add_argument('--trg-lang', required=True,
        help='Target language id')
    parser.add_argument('--output', required=True,
        help='Output file')
    parser.add_argument('--threshold', type=float, default=0,
        help='Threshold on extracted bitexts')

    # mining params
    parser.add_argument('--mode',
        choices=['search', 'score', 'mine'], required=True,
        help='Execution mode')
    parser.add_argument('-k', '--neighborhood',
        type=int, default=4,
        help='Neighborhood size')
    parser.add_argument('--margin',
        choices=['absolute', 'distance', 'ratio'], default='ratio',
        help='Margin function')
    parser.add_argument('--retrieval',
        choices=['fwd', 'bwd', 'max', 'intersect'], default='max',
        help='Retrieval strategy')
    parser.add_argument('--unify', action='store_true',
        help='Unify texts')
    parser.add_argument('--gpu', action='store_true',
        help='Run knn on all available GPUs')
    parser.add_argument('--verbose', action='store_true',
        help='Detailed output')

    # embeddings
    parser.add_argument('--src-embeddings', required=True,
        help='Precomputed source sentence embeddings')
    parser.add_argument('--trg-embeddings', required=True,
        help='Precomputed target sentence embeddings')
    parser.add_argument('--dim', type=int, default=1024,
        help='Embedding dimensionality')
    args = parser.parse_args()

    print('LASER: tool to search, score or mine bitexts')
    use_gpu = torch.cuda.is_available() and args.gpu
    if use_gpu:
        print(' - knn will run on all available GPUs (recommended)')
    else:
        print(' - knn will run on CPU (slow)')

    src_inds, src_sents = TextLoadUnify(args.src, args)
    trg_inds, trg_sents = TextLoadUnify(args.trg, args)

    def unique_embeddings(emb, ind, verbose=False):
        aux = {j: i for i, j in enumerate(ind)}
        if verbose:
            print(' - unify embeddings: {:d} -> {:d}'.format(len(emb), len(aux)))
        return emb[[aux[i] for i in range(len(aux))]]

    # load the embeddings
    x = EmbedLoad(args.src_embeddings, args.dim, verbose=args.verbose)
    if args.unify:
        x = unique_embeddings(x, src_inds, args.verbose)
    faiss.normalize_L2(x)
    y = EmbedLoad(args.trg_embeddings, args.dim, verbose=args.verbose)
    if args.unify:
        y = unique_embeddings(y, trg_inds, args.verbose)
    faiss.normalize_L2(y)

    # calculate knn in both directions
    if args.retrieval != 'bwd':
        if args.verbose:
            print(' - perform {:d}-nn source against target'.format(args.neighborhood))
        x2y_sim, x2y_ind = knn(x, y, min(y.shape[0], args.neighborhood), use_gpu)
        x2y_mean = x2y_sim.mean(axis=1)

    if args.retrieval != 'fwd':
        if args.verbose:
            print(' - perform {:d}-nn target against source'.format(args.neighborhood))
        y2x_sim, y2x_ind = knn(y, x, min(x.shape[0], args.neighborhood), use_gpu)
        y2x_mean = y2x_sim.mean(axis=1)

    # margin function
    if args.margin == 'absolute':
        margin = lambda a, b: a
    elif args.margin == 'distance':
        margin = lambda a, b: a - b
    else:  # args.margin == 'ratio':
        margin = lambda a, b: a / b

    fout = open(args.output, mode='w', encoding=args.encoding, errors='surrogateescape')

    if args.mode == 'search':
        if args.verbose:
            print(' - Searching for closest sentences in target')
            print(' - writing alignments to {:s}'.format(args.output))
        scores = score_candidates(x, y, x2y_ind, x2y_mean, y2x_mean, margin, args.verbose)
        best = x2y_ind[np.arange(x.shape[0]), scores.argmax(axis=1)]

        nbex = x.shape[0]
        ref = np.linspace(0, nbex-1, nbex).astype(int)  # [0, nbex)
        err = nbex - np.equal(best.reshape(nbex), ref).astype(int).sum()
        print(' - errors: {:d}={:.2f}%'.format(err, 100*err/nbex))
        for i in src_inds:
            print(trg_sents[best[i]], file=fout)

    elif args.mode == 'score':
        for i, j in zip(src_inds, trg_inds):
            s = score(x[i], y[j], x2y_mean[i], y2x_mean[j], margin)
            print(s, src_sents[i], trg_sents[j], sep='\t', file=fout)

    elif args.mode == 'mine':
        if args.verbose:
            print(' - mining for parallel data')
        fwd_scores = score_candidates(x, y, x2y_ind, x2y_mean, y2x_mean, margin, args.verbose)
        bwd_scores = score_candidates(y, x, y2x_ind, y2x_mean, x2y_mean, margin, args.verbose)
        fwd_best = x2y_ind[np.arange(x.shape[0]), fwd_scores.argmax(axis=1)]
        bwd_best = y2x_ind[np.arange(y.shape[0]), bwd_scores.argmax(axis=1)]
        if args.verbose:
            print(' - writing alignments to {:s}'.format(args.output))
            if args.threshold > 0:
                print(' - with threshold of {:f}'.format(args.threshold))
        if args.retrieval == 'fwd':
            for i, j in enumerate(fwd_best):
                print(fwd_scores[i].max(), src_sents[i], trg_sents[j], sep='\t', file=fout)
        if args.retrieval == 'bwd':
            for j, i in enumerate(bwd_best):
                print(bwd_scores[j].max(), src_sents[i], trg_sents[j], sep='\t', file=fout)
        if args.retrieval == 'intersect':
            for i, j in enumerate(fwd_best):
                if bwd_best[j] == i:
                    print(fwd_scores[i].max(), src_sents[i], trg_sents[j], sep='\t', file=fout)
        if args.retrieval == 'max':
            indices = np.stack((np.concatenate((np.arange(x.shape[0]), bwd_best)),
                                np.concatenate((fwd_best, np.arange(y.shape[0])))), axis=1)
            scores = np.concatenate((fwd_scores.max(axis=1), bwd_scores.max(axis=1)))
            seen_src, seen_trg = set(), set()
            for i in np.argsort(-scores):
                src_ind, trg_ind = indices[i]
                if not src_ind in seen_src and not trg_ind in seen_trg:
                    seen_src.add(src_ind)
                    seen_trg.add(trg_ind)
                    if scores[i] > args.threshold:
                        print(scores[i], src_sents[src_ind], trg_sents[trg_ind], sep='\t', file=fout)

    fout.close()
