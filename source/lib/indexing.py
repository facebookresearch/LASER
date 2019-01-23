#!/usr/bin/python3
# Copyright (c) 2018-present, Facebook, Inc.
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
# tools for indexing and search with FAISS

import faiss
import os.path
import sys
import numpy as np


###############################################################################
# create an FAISS index on the given data

def IndexCreate(dname, idx_type,
                verbose=False, normalize=True, save_index=False, dim=1024):

    assert idx_type == 'FlatL2', 'only FlatL2 index is currently supported'
    x = np.fromfile(dname, dtype=np.float32, count=-1)
    nbex = x.shape[0] // dim
    print(' - embedding: {:s} {:d} examples of dim {:d}'
          .format(dname, nbex, dim))
    x.resize(nbex, dim)
    print(' - creating FAISS index')
    idx = faiss.IndexFlatL2(dim)
    if normalize:
        faiss.normalize_L2(x)
    idx.add(x)
    if save_index:
        iname = 'TODO'
        print(' - saving index into ' + iname)
        faiss.write_index(idx, iname)
    return x, idx


###############################################################################
# search closest vector for all languages pairs and calculate error rate

def IndexSearchMultiple(data, idx, verbose=False):
    nl = len(data)
    nbex = data[0].shape[0]
    err = np.zeros((nl, nl)).astype(float)
    ref = np.linspace(0, nbex-1, nbex).astype(int)  # [0, nbex)
    if verbose:
        print('Calculating similarity error:')
    for i1 in range(nl):
        for i2 in range(nl):
            if i1 != i2:
                D, I = idx[i2].search(data[i1], 1)
                err[i1, i2] \
                    = (nbex - np.equal(I.reshape(nbex), ref)
                       .astype(int).sum()) / nbex
                if verbose:
                    print(' - similarity error {:s}/{:s}: {:5d}={:5.2f}%'
                          .format(args.langs[i1], args.langs[i2],
                                  err[i1, i2], 100.0 * err[i1, i2]))
    return err


###############################################################################
# print confusion matrix

def IndexPrintConfusionMatrix(err, langs):
    nl = len(langs)
    assert nl == err.shape[0], 'size of errror matrix doesn not match'
    print('Confusion matrix:')
    print('{:8s}'.format('langs'), end='')
    for i2 in range(nl):
        print('{:8s} '.format(langs[i2]), end='')
    print('{:8s}'.format('avg'))
    for i1 in range(nl):
        print('{:3s}'.format(langs[i1]), end='')
        for i2 in range(nl):
            print('{:8.2f}%'.format(100 * err[i1, i2]), end='')
        print('{:8.2f}%'.format(100 * err[i1, :].sum() / (nl-1)))

    print('avg', end='')
    for i2 in range(nl):
        print('{:8.2f}%'.format(100 * err[:, i2].sum() / (nl-1)), end='')

    # global average
    print('{:8.2f}%'.format(100 * err.sum() / (nl-1) / (nl-1)))


###############################################################################
# Load an FAISS index

def IndexLoad(idx_name, nprobe):
    print(' - loading FAISS index', idx_name)
    index = faiss.read_index(idx_name)
    print(' - found {:d} sentences of dim {:d}'.format(index.ntotal, index.d))
    print(' - setting nbprobe to {:d}'.format(nprobe))
    faiss.GpuParameterSpace().set_index_parameter(index, 'nprobe', nprobe)
    return index


###############################################################################
# Opens a text file with the sentences corresponding to the indices used
# by an FAISS index
# We also need the reference files with the byte offsets to the beginning
# of each sentence
# optionnally:  array with number of words per sentence
# All arrays are memory mapped

def IndexTextOpen(txt_fname):
    print(' - opening text corpus', txt_fname)
    txt_mmap = np.memmap(txt_fname, mode='r', dtype=np.uint8)
    fname = txt_fname.replace('.txt', '.ref.bin32')
    if os.path.isfile(fname):
        print(' - opening sentence start offsets {} (32 bit)'.format(fname))
        ref_mmap = np.memmap(fname, mode='r', dtype=np.uint32)
    else:
        fname = txt_fname.replace('.txt', '.ref.bin64')
        if os.path.isfile(fname):
            print(' - opening sentence start offsets {} (64 bit)'
                  .format(fname))
            ref_mmap = np.memmap(fname, mode='r', dtype=np.uint64)
        else:
            print('ERROR: no file with sentence start offsets found')
            sys.exit(1)
    print(' - found {:d} sentences'.format(ref_mmap.shape[0]))

    nbw_mmap = None
    fname = txt_fname.replace('.txt', '.nw.bin8')
    if os.path.isfile(fname):
        print(' - opening word counts', fname)
        nbw_mmap = np.memmap(fname, mode='r', dtype=np.uint8)

    return txt_mmap, ref_mmap, nbw_mmap


###############################################################################
# Return the text for the given index

def IndexTextQuery(txt_mmap, ref_mmap, idx):
    p = int(ref_mmap[idx])  # get starting byte position
    i = 0
    dim = 10000  # max sentence length in bytes
    b = bytearray(dim)
    #  find EOL
    while txt_mmap[p+i] != 10 and i < dim:
        b[i] = txt_mmap[p+i]
        i += 1

    return b[0:i].decode('utf-8')


###############################################################################
# Search the [k] nearest vectors of [x] in the given index
# and return the text line

def IndexSearchKNN(index, x, T, R, kmax=1, Dmax=1.0, dedup=True):
    D, I = index.search(x, kmax)
    prev = {}  # for depuplication
    res = []
    for n in range(x.shape[0]):
        for i in range(kmax):
            txt = IndexTextQuery(T, R, I[n, i])
            if (dedup and txt not in prev) and D[n, i] <= Dmax:
                prev[txt] = 1
                res.append([txt, D[n, i]])
    return res
