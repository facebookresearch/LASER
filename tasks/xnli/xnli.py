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
# XNLI

import os
import sys
import argparse
import pdb
import faiss
import numpy as np

# get environment
assert os.environ.get('LASER'), 'Please set the enviornment variable LASER'
LASER = os.environ['LASER']

sys.path.append(LASER + '/source')
sys.path.append(LASER + '/source/tools')
from embed import SentenceEncoder, EncodeLoad, EncodeFile
from text_processing import Token, BPEfastApply


################################################################################

parser = argparse.ArgumentParser('LASER: training and evaluation for XNLI')
parser.add_argument('--tsv', type=str, default='tsv',
    help='Directory of the TSV file')
parser.add_argument('--data_dir', type=str, default='.',
    help='Base directory for created files')
parser.add_argument('--bpe_codes', type=str, required=True,
    help='Directory of the tokenized data')
parser.add_argument('--verbose', action='store_true',
    help='Detailed output')

# options for encoder
parser.add_argument('--encoder', type=str, required=True,
    help='encoder to be used')
parser.add_argument(
    '--lang', '-L', nargs='+', default=None,
    help="List of languages to test on")
parser.add_argument('--buffer-size', type=int, default=10000,
    help='Buffer size (sentences)')
parser.add_argument('--max-tokens', type=int, default=12000,
    help='Maximum number of tokens to process in a batch')
parser.add_argument('--max-sentences', type=int, default=None,
    help='Maximum number of sentences to process in a batch')
parser.add_argument('--cpu', action='store_true',
    help='Use CPU instead of GPU')

args = parser.parse_args()

print('LASER: training and evaluation for XNLI')

if not os.path.exists(args.data_dir):
    os.mkdir(args.data_dir)

enc = EncodeLoad(args)

languages_train = ('en',)
languages = ('en', 'ar', 'bg', 'de', 'el', 'es', 'fr', 'hi', 'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh')

print('\nProcessing train:')
for lang in languages_train:
    for part in ('prem', 'hyp'):
        cfname = os.path.join(args.data_dir, 'xnli.train.' + part + '.')
        Token(cfname + lang,
              cfname + 'tok.' + lang,
              lang=lang,
              romanize=True if lang=='el' else False,
              lower_case=True, gzip=True,
              verbose=args.verbose, over_write=False)
        BPEfastApply(cfname + 'tok.' + lang,
                     cfname + 'bpe.' + lang,
                     args.bpe_codes,
                     verbose=args.verbose, over_write=False)
        EncodeFile(enc,
                   cfname + 'bpe.' + lang,
                   cfname + 'enc.' + lang,
                   verbose=args.verbose, over_write=False,
                   buffer_size=args.buffer_size) 

for corpus in ('xnli.dev', 'xnli.test'):
    print('\nProcessing {}:'.format(corpus))
    for part in ('prem', 'hyp'):
        cfname = os.path.join(args.data_dir, corpus + '.' + part + '.')
        for lang in languages:
            Token(cfname + lang,
                  cfname + 'tok.' + lang,
                  lang=lang,
                  romanize=True if lang=='el' else False,
                  lower_case=True, gzip=False,
                  verbose=args.verbose, over_write=False)
            BPEfastApply(cfname + 'tok.' + lang,
                         cfname + 'bpe.' + lang,
                         args.bpe_codes,
                         verbose=args.verbose, over_write=False)
            EncodeFile(enc,
                       cfname + 'bpe.' + lang,
                       cfname + 'enc.' + lang,
                       verbose=args.verbose, over_write=False,
                       buffer_size=args.buffer_size) 

