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
# Quora Q&A paraphrase detection

import os
import sys
import argparse
import faiss
import numpy as np

# get environment
assert os.environ.get('LASER'), 'Please set the enviornment variable LASER'
LASER = os.environ['LASER']

sys.path.append(LASER + '/source')
sys.path.append(LASER + '/source/lib')
from embed import SentenceEncoder, EncodeLoad, EncodeFile
from text_processing import Token, BPEfastApply
from indexing import IndexCreate, IndexSearchMultiple, IndexPrintConfusionMatrix

###############################################################################

parser = argparse.ArgumentParser('LASER: similarity search')
parser.add_argument('--base-dir', type=str, default='.',
    help='Base directory for all data files')
parser.add_argument('--data', type=str, required=True,
    help='Direcory and basename of input data (language name will be added)')
parser.add_argument('--output', type=str, required=True,
    help='Directory and basename of created data (language name will be added)')
parser.add_argument('--textual', action='store_true',
    help='Use textual comparison instead of indicies')
parser.add_argument(
    '--lang', '-l', nargs='+', required=True,
    help="List of languages to test on")

# preprocessing
parser.add_argument('--bpe-codes', type=str, required=True,
    help='Fast BPPE codes and vocabulary')
parser.add_argument('--verbose', action='store_true',
    help='Detailed output')

# options for encoder
parser.add_argument('--encoder', type=str, required=True,
    help='encoder to be used')
parser.add_argument('--buffer-size', type=int, default=100,
    help='Buffer size (sentences)')
parser.add_argument('--max-tokens', type=int, default=12000,
    help='Maximum number of tokens to process in a batch')
parser.add_argument('--max-sentences', type=int, default=None,
    help='Maximum number of sentences to process in a batch')
parser.add_argument('--cpu', action='store_true',
    help='Use CPU instead of GPU')

args = parser.parse_args()

print('LASER: similarity search')

print('\nProcessing:')
all_texts = []
if args.textual:
    print(' - using textual comparision')
    for l in args.lang:
        with open(os.path.join(args.base_dir, args.data + '.' + l),
                  encoding='utf-8', errors='surrogateescape') as f:
            texts = f.readlines()
            print(' -   {:s}: {:d} lines'.format(args.data + '.' + l, len(texts)))
            all_texts.append(texts)

enc = EncodeLoad(args)

out_dir = os.path.dirname(args.output)
if not os.path.exists(out_dir):
    print(' - creating directory {}'.format(out_dir))
    os.mkdir(out_dir)

all_data = []
all_index = []
for l in args.lang:
    Token(os.path.join(args.base_dir, args.data + '.' + l),
          os.path.join(args.base_dir, args.output + '.tok.' + l),
          lang=l,
          romanize=True if l == 'el' else False,
          lower_case=True,
          verbose=args.verbose, over_write=False)
    BPEfastApply(os.path.join(args.base_dir, args.output + '.tok.' + l),
                 os.path.join(args.base_dir, args.output + '.bpe.' + l),
                 args.bpe_codes,
                 verbose=args.verbose, over_write=False)
    EncodeFile(enc,
               os.path.join(args.base_dir, args.output + '.bpe.' + l),
               os.path.join(args.base_dir, args.output + '.enc.' + l),
               verbose=args.verbose, over_write=False)
    d, idx = IndexCreate(os.path.join(args.base_dir, args.output + '.enc.' + l),
                         'FlatL2',
                         verbose=args.verbose, save_index=False)
    all_data.append(d)
    all_index.append(idx)

err = IndexSearchMultiple(all_data, all_index, args.lang, texts=all_texts,
                          verbose=False, print_errors=False)
IndexPrintConfusionMatrix(err, args.lang)
