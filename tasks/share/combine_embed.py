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
# Simple Python tool to combine the sentence embeddings of multiple
# which have been split

import sys
import argparse
import numpy as np


###############################################################################
#
# Create command line arguments
#
###############################################################################

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description="Combine split embeddings")

parser.add_argument(
    '--inp', '-i', type=argparse.FileType('r'), required=True, metavar='PATH',
    help="Input file with embeddings (numpy memory mapped file)")
parser.add_argument(
    '--dim', '-d', type=int, default=1024, metavar='INT',
    help="Dimension of sentence embedding.")
parser.add_argument(
    '--out', '-o', required=True, type=argparse.FileType('w'), metavar='PATH',
    help="Output file with embeddings (numpy memory mapped file)")
parser.add_argument(
    '--sid', '-s', type=argparse.FileType('r', encoding='UTF-8'),
    required=True, metavar='PATH',
    help="File with sentence IDs.")

args = parser.parse_args()

print("Combing sentence embeddings which had been split into several lines")

# read the input embeddings
ems = np.fromfile(args.inp.name,
                  dtype=np.float32, count=-1).reshape(-1, args.dim)
ninp = ems.shape[0]
print(" - input: %s, %d lines" % (args.inp.name, ninp))

# get all sentence IDs
sid = np.empty(ninp, dtype=np.int32)
i = 0
for line in args.sid:
    sid[i] = int(line)
    i += 1
nout = sid.max()
print(" - sentence IDs: %s, %d lines" % (args.sid.name, nout))

# combine the sentence vectors
print(" - combining")
emb = np.zeros((nout, args.dim), dtype=np.float32)
cnt = np.zeros(nout, dtype=np.int32)
for i in range(ninp):
    idx = sid[i] - 1    # sentence IDs are 1..N !!
    emb[idx] += ems[i]  # cumulate sentence vectors
    cnt[idx] += 1

if (cnt == 0).astype(int).sum() > 0:
    print("ERROR: error in sentence splits: missing lines")
    sys.exit(1)

# normalize
for i in range(nout):
    emb[i] /= cnt[i]

print(" - output: %s" % args.out.name)
emb.tofile(args.out)
