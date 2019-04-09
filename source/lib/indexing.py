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
# tools for indexing and search with FAISS

import faiss
import os.path
import sys
import numpy as np

#-------------------------------------------------------------
# Get list of fnames:
#  - we loop over the list of given languages
#  - for each language, we also check if there are splitted files .%03d

def SplitFnames(par_fname, langs):
    fnames = []
    for l in langs:
        fname = par_fname + '.' + l
        if os.path.isfile(fname):
            fnames.append(fname)
        for i in range(1000):
            fname = par_fname + '.' + l + '.{:03d}'.format(i)
            if os.path.isfile(fname):
                fnames.append(fname)
    if len(fnames) == 0:
        print("ERROR: no embeddings found in {:s}*".format(par_fname))
        sys.exit(1)
    return fnames

def SplitOpen(par_fname, langs, dim, dtype, verbose=False):
    M = []
    nf = 0
    nc = 0
    print('Reading sentence embeddings')
    print(' - memory mapped files {:s}'.format(par_fname))
    for fname in SplitFnames(par_fname, langs):
        n = int(os.path.getsize(fname) / dim / np.dtype(dtype).itemsize)
        if verbose:
            print(' - {:s}: {:d} x {:d}'.format(fname, n, dim))
        Mi = np.memmap(fname, mode='r', dtype=dtype, shape=(n, dim))
        nc += n
        nf += 1
        M.append(Mi)
    print(' - total of {:d} files: {:d} x {:d}'.format(nf, nc, dim))
    return M

def SplitAccess(M, idx):
    i = idx
    for Mi in M:
        n = Mi.shape[0]
        if i < n:
            return Mi[i,:]
        i -= n
    print('ERROR: index {:d} is too large form memory mapped files'.format(idx))
    sys.exit(1)


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

def IndexSearchMultiple(data, idx, verbose=False, texts=None, print_errors=False):
    nl = len(data)
    nbex = data[0].shape[0]
    err = np.zeros((nl, nl)).astype(float)
    ref = np.linspace(0, nbex-1, nbex).astype(int)  # [0, nbex)
    if verbose:
        if texts is None: 
            print('Calculating similarity error (indices):')
        else:
            print('Calculating similarity error (textual):')
    for i1 in range(nl):
        for i2 in range(nl):
            if i1 != i2:
                D, I = idx[i2].search(data[i1], 1)
                if texts: # do textual comparison
                    e1 = 0
                    for p in range(I.shape[0]):
                        if texts[i2][p] != texts[i2][I[p,0]]:
                            e1 += 1
                            if print_errors:
                                print('Error {:s}\n      {:s}'
                                      .format(texts[i2][p].strip(), texts[i2][I[p,0]].strip()))
                    err[i1, i2] = e1 / nbex
                else:  # do index based comparision
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
    print('{:8.2f}%'.format(100 * err.sum() / (nl-1) / nl))


###############################################################################
# Load an FAISS index

def IndexLoad(idx_name, nprobe, gpu=False):
    print('Reading FAISS index')
    print(' - index: {:s}'.format(idx_name))
    index = faiss.read_index(idx_name)
    print(' - found {:d} sentences of dim {:d}'.format(index.ntotal, index.d))
    print(' - setting nbprobe to {:d}'.format(nprobe))
    if gpu:
        print(' - transfer index to %d GPUs ' % faiss.get_num_gpus())
        #co = faiss.GpuMultipleClonerOptions()
        #co.shard = True
        index = faiss.index_cpu_to_all_gpus(index) # co=co
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
    print('Reading text corpus')
    print(' - texts: {:s}'.format(txt_fname))
    txt_mmap = np.memmap(txt_fname, mode='r', dtype=np.uint8)
    fname = txt_fname.replace('.txt', '.ref.bin32')
    if os.path.isfile(fname):
        print(' - sentence start offsets (32 bit): {}'.format(fname))
        ref_mmap = np.memmap(fname, mode='r', dtype=np.uint32)
    else:
        fname = txt_fname.replace('.txt', '.ref.bin64')
        if os.path.isfile(fname):
            print(' - sentence start offsets (64 bit): {}'.format(fname))
            ref_mmap = np.memmap(fname, mode='r', dtype=np.uint64)
        else:
            print('ERROR: no file with sentence start offsets found')
            sys.exit(1)
    print(' - found {:d} sentences'.format(ref_mmap.shape[0]))

    nbw_mmap = None
    fname = txt_fname.replace('.txt', '.nw.bin8')
    if os.path.isfile(fname):
        print(' - word counts: {:s}'.format(fname))
        nbw_mmap = np.memmap(fname, mode='r', dtype=np.uint8)

    M = None
    fname = txt_fname.replace('.txt', '.meta')
    if os.path.isfile(fname):
        M = []
        n = 0
        print(' - metafile: {:s}'.format(fname))
        with open(fname, 'r') as fp:
            for line in fp:
                fields = line.strip().split()
                if len(fields) != 2:
                    print('ERROR: format error in meta file')
                    sys.exit(1)
                n += int(fields[1])
                M.append({'lang': fields[0], 'n': n})
        print(' - found {:d} languages:'.format(len(M)), end='')
        for L in M:
            print(' {:s}'.format(L['lang']), end='')
        print('')

    return txt_mmap, ref_mmap, nbw_mmap, M


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
# and return the text lines

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
