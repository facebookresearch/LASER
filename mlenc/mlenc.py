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
# Main (version which reads networks in PyTorch format)

import sys
import argparse
import time
import torch
import numpy as np
from collections import namedtuple

# LASER modules
import mlenc_const as const
from mlenc_binarize import LoadHashTable, BinarizeText
from apply_bpe import BPE
import mlenc_blstm as encoders

###############################################################################
#
# Create command line arguments
#
###############################################################################


def create_parser():
    parser = argparse.ArgumentParser(
                formatter_class=argparse.RawDescriptionHelpFormatter,
                description="Calculate multilingual sentence encodings")

    parser.add_argument(
        '--text', '-i', type=argparse.FileType('r', encoding='UTF-8'),
        default=sys.stdin,
        metavar='PATH',
        help="Input text file (default: standard input).")
    parser.add_argument(
        '--preserve-case', '-C', action='store_true',
        help="Preserve case of input texts (default is all lower case)")
    parser.add_argument(
        '--bpe_codes', '-c', type=argparse.FileType('r', encoding='UTF-8'),
        metavar='PATH',
        required=True,
        help="File with BPE codes (created by learn_bpe.py).")
    parser.add_argument(
        '--hash_table', '-t', type=argparse.FileType('r', encoding='UTF-8'),
        metavar='PATH',
        required=True,
        help="File with hash table for binarization.")
    parser.add_argument(
        '--model', '-n', type=argparse.FileType('r'), metavar='PATH',
        required=True,
        help="File with trained model used for encoding.")

    parser.add_argument(
        '--output_bpe', '-p', type=argparse.FileType('w', encoding='UTF-8'),
        metavar='PATH',
        help="Output file with BPE processed text")
    parser.add_argument(
        '--output_bin', '-b', type=argparse.FileType('w'), metavar='PATH',
        help="Output file with binarized text (torch 2D int tensor)")
    parser.add_argument(
        '--output_enc', '-e', type=argparse.FileType('w'), metavar='PATH',
        help="Output file for encoded text_enc (torch 2D float tensor)")
    parser.add_argument(
        '--gpu', '-g', type=int, default=0, metavar='INT',
        help="GPU device (use -1 for CPU)")
    parser.add_argument(
        '--bsize', '-B', type=int, default=128, metavar='INT',
        help="Batch size for parallel processing")
    parser.add_argument(
        '--lstm_batched', '-X', type=int, default=1, metavar='INT',
        help="Use batch mode for LSTM (faster)")
    parser.add_argument(
        '--max_len', '-m', type=int, default=100, metavar='INT',
        help="Maximum number of words in one line"
             " (longer lines will be skipped)")
    parser.add_argument(
        '--separator', '-s', type=str, default='@@', metavar='STR',
        help="Separator between non-final subword units"
             " (default: '%(default)s'))")
    parser.add_argument(
        '--verbose', '-v', type=int, default=3, metavar='INT',
        help="Verbose level: 0=quite, 1: processing, 2: network 4: debugging")

    return parser


###############################################################################
#
# Function to handle one batch: BPE, binarization + encoding
#
###############################################################################

def HandleBatch(model, data, stats, lstm_batched=1):
    BinarizeText(model.htable, data,
                 order="left-to-right" if lstm_batched > 0
                       else "right-to-left")

    # update stats
    stats.nb_sents += data.text_bin.shape[1]
    stats.nb_unk += (data.text_bin == data.idx_unk).sum()
    stats.nb_words += data.text_slen.sum()
    stats.nb_skip += (data.text_slen > data.text_bin.shape[0]).sum()

    if data.file_bin:
        data.text_bin.tofile(data.file_bin)
    if lstm_batched > 0:
        model.net.EncodeBatch(data)
    else:
        model.net.EncodeBatch1(data)
    if data.file_enc:
        data.text_enc.tofile(data.file_enc)

    if args.verbose & const.VERBOSE_DEBUG:
        print("Debugging output:")
        for b in range(data.text_bin.shape[1]):
            slen = data.text_slen[:][b]
            sys.stdout.write("  %d\tbin (reversed):" % b)
            if lstm_batched > 0:
                for l in range(0, slen):
                    sys.stdout.write(" %d" % data.text_bin[l][b])
            else:
                for l in range(-slen, 0):
                    sys.stdout.write(" %d" % data.text_bin[l][b])
            print("  len=%d" % slen)
            print("     \tembed norm: %.14f"
                  % np.linalg.norm(data.text_enc[:][b]))

    if (args.verbose & const.VERBOSE_INFO) and (stats.nb_sents % 100000 == 0):
        sys.stdout.write("\r - sentences %d" % stats.nb_sents)
        sys.stdout.flush()


###############################################################################
#
# Main program
#
###############################################################################

parser = create_parser()
args = parser.parse_args()

# all models are grouped into one structure
model = namedtuple("model", ["htable", "bpe", "net"])
print("\nLoading models")
loaded = torch.load(args.model.name)
model.htable = LoadHashTable(args.hash_table, args.verbose)
model.bpe = BPE(args.bpe_codes, separator=args.separator)
model.net = encoders.BLSTM(args.model.name, gpu=args.gpu, verbose=args.verbose)

# all data structures are grouped into one structure
data = namedtuple("data",
                  ["text_bpe", "text_slen", "text_bin", "text_enc"
                   "file_bpe", "file_bin", "file_enc"
                   "idx_pad", "idx_unk"])
data.text_slen = np.empty(args.bsize, dtype=np.int32)
data.text_bin = np.empty((args.max_len, args.bsize), dtype=np.int32)
data.text_enc = np.empty((args.bsize, model.net.nembed), dtype=np.float32) \
                         if model.net else 0
data.idx_unk = model.htable['<UNK>']
data.idx_pad = model.htable['<PAD>']

# open requested output files
if args.verbose & const.VERBOSE_INFO > 0:
    print("\nOutput files:")
if args.output_bpe:
    if args.verbose & const.VERBOSE_INFO > 0:
        print(" - BPE encoding in file '%s'" % args.output_bpe.name)
    data.file_bpe = args.output_bpe
else:
    data.file_bpe = 0

if args.output_bin:
    if args.verbose & const.VERBOSE_INFO > 0:
        print(" - binarized text in file '%s'" % args.output_bin.name)
    data.file_bin = args.output_bin
else:
    data.file_bin = 0

if args.output_enc:
    if args.verbose & const.VERBOSE_INFO > 0:
        print(" - encoded text in file '%s'" % args.output_enc.name)
    data.file_enc = args.output_enc
else:
    data.file_enc = 0

stats = namedtuple("stats", ["nb_sents", "nb_skip", "nb_unk", "nb_words"])
stats.nb_sents = 0
stats.nb_skip = 0
stats.nb_unk = 0
stats.nb_words = 0

if args.verbose & const.VERBOSE_INFO > 0:
    print("\nProcessing '%s'" % args.text.name)
    if args.preserve_case:
        print(" - preserving case of input texts")
    else:
        print(" - lower casing input texts")
    print(" - batch size is %d" % args.bsize)
start_t = time.process_time()
data.text_bpe = []  # paragraphe of BPE encoded sentences
for line in args.text:
    if not args.preserve_case:
        line = line.strip().lower()
    line = model.bpe.segment(line).strip()
    data.text_bpe.append(line)
    if data.file_bpe:
        data.file_bpe.write(line + "\n")

    if len(data.text_bpe) == args.bsize:
        HandleBatch(model, data, stats, lstm_batched=args.lstm_batched)
        data.text_bpe = []

# process last (partial) batch
if len(data.text_bpe) > 0:
    HandleBatch(model, data, stats, lstm_batched=args.lstm_batched)

if args.verbose & const.VERBOSE_INFO > 0:
    print("\r - sentences: %d, skipped: %d=%.2f%%,  words: %d, unk: %d=%.2f%%"
          % (stats.nb_sents, stats.nb_skip, 100.0*stats.nb_skip/stats.nb_sents,
             stats.nb_words, stats.nb_unk, 100.0*stats.nb_unk/stats.nb_words))
    dur = time.process_time() - start_t
    print(" - time: %.1f sec, %.1f sents/sec" % (dur, stats.nb_sents/dur))
