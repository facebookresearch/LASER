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
# Python tools to calculate all pairwsie distances between
# two sets of sentence emebddings.
# The indicies of the k-closest ones are dumped into a file

import argparse
import numpy as np
import faiss

cnt = -1       # set to -1 to read all
maxn = 500000  # maximum number of vectors for brute-force search with FAISS

parser = argparse.ArgumentParser()
parser.add_argument("--fname",
                    help="basename for embeddings and output files")
parser.add_argument("--lang1", help="first language")
parser.add_argument("--lang2", help="second language")
parser.add_argument("--dim", type=int, default=1024,
                    help="dimension of sentence embeddings")
parser.add_argument("--k", type=int, default=1, help="K-nn")
parser.add_argument("--norm", type=int, const=True, nargs="?",
                    help="renormalize embeddings")
args = parser.parse_args()


print("Reading files")

fnamel = "%s.%s" % (args.fname, args.lang1)
print(" - " + fnamel)
x = np.fromfile(fnamel, dtype=np.float32, count=cnt)
xn = x.shape[0] // args.dim
print(" - found %d elements of dim %d = %d lines" % (x.shape[0], args.dim, xn))
x.resize(xn, args.dim)
if xn > maxn:
    printf("- WARNING: your are using many vectors.")
    printf("           It is recommended to use advanced features"
           " of FAISS to speed-up processing")

fnamel = "%s.%s" % (args.fname, args.lang2)
print(" - " + fnamel)
y = np.fromfile(fnamel, dtype=np.float32, count=cnt)
yn = y.shape[0] // args.dim
print(" - found %d elements of dim %d = %d lines" % (y.shape[0], args.dim, yn))
y.resize(yn, args.dim)
if yn > maxn:
    printf("- WARNING: your are using an huge index.")
    printf("           It is recommended to use advanced features"
           " of FAISS to speed-up processing")


print("Searching %d vectors in %d with FAISS for k=%d" % (xn, yn, args.k))
if args.norm:
    print(" - normalize X")
    faiss.normalize_L2(x)
    print(" - normalize Y")
    faiss.normalize_L2(y)

print(" - create index on Y")
idx = faiss.IndexFlatL2(args.dim)
idx.add(y)
print(" - find k-nn of X in Y")
D, I = idx.search(x, args.k)

if args.norm:
    fnamel = "%s.cos.k%d" % (args.fname, args.k)
else:
    fnamel = "%s.l2.k%d" % (args.fname, args.k)

print(" - dumping results into %s" % fnamel)
np.savetxt(fnamel + ".dist", D, "%8f")
np.savetxt(fnamel + ".idx", I, "%d")
