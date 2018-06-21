#!/usr/bin/python
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
# Python tool to calculate the pairwise similarity error
# between a set of texts which are mutual translations.
# If the closest vector in the target language is in a different
# position in the file, we count an error. Therefore, the texts
# should not contain duplicates.

import argparse
import numpy as np
import faiss
import sys

parser = argparse.ArgumentParser()
parser.add_argument(
   "--fname", help="basename for embeddings and output files")
parser.add_argument(
    '--langs', '-l', type=str, required=True, nargs='*', action='append',
    help="List of languages to test on (option can be given multiple times)")
parser.add_argument(
    "--dim", type=int, default=1024,
    help="dimension of sentence embeddings")
parser.add_argument(
    "--norm", type=int, const=True, nargs="?",
    help="renormalize embeddings")
args = parser.parse_args()


print("Processing")
data = []
faiss_idx = []
for l in args.langs:
    fnamel = "%s.%s" % (args.fname, l[0])
    print(" - reading " + fnamel)
    x = np.fromfile(fnamel, dtype=np.float32, count=-1)
    nbex = x.shape[0] // args.dim
    print(" - found %d examples of dim %d" % (nbex, args.dim))
    x.resize(nbex, args.dim)
    if args.norm:
        print(" - normalizing")
        faiss.normalize_L2(x)
    data.append(x)
    print(" - creating FAISS index")
    idx = faiss.IndexFlatL2(args.dim)
    idx.add(x)
    faiss_idx.append(idx)

nl = len(data)
err = np.zeros((nl, nl)).astype(int)
ref = np.linspace(0, nbex-1, nbex).astype(int)  # [0, nbex)
for i1 in range(nl):
    for i2 in range(nl):
        if i1 != i2:
            D, I = faiss_idx[i2].search(data[i1], 1)
            err[i1, i2] \
                = nbex - np.equal(I.reshape(nbex), ref).astype(int).sum()
            print(" - similarity error %s/%s: %5d=%5.2f%%"
                  % (args.langs[i1][0], args.langs[i2][0],
                     err[i1, i2], 100.0 * err[i1, i2] / nbex))

print("\nConfusion matrix:")
for i1 in range(nl):
    sys.stdout.write(str.format("%8s " % args.langs[i1][0]))
print()
for i1 in range(nl):
    sys.stdout.write(args.langs[i1][0])
    for i2 in range(nl):
        sys.stdout.write(str.format("%8.2f%%" % (100.0 * err[i1, i2] / nbex)))
    print()
