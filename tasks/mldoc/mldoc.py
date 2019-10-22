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
# Calculate embeddings of MLDoc corpus


import os
import sys
import argparse

# get environment
assert os.environ.get('LASER'), 'Please set the enviornment variable LASER'
LASER = os.environ['LASER']

sys.path.append(LASER + '/source')
sys.path.append(LASER + '/source/tools')
from embed import SentenceEncoder, EncodeLoad, EncodeFile
from text_processing import Token, BPEfastApply, SplitLines, JoinEmbed


###############################################################################

parser = argparse.ArgumentParser('LASER: calculate embeddings for MLDoc')
parser.add_argument(
    '--mldoc', type=str, default='MLDoc',
    help='Directory of the MLDoc corpus')
parser.add_argument(
    '--data_dir', type=str, default='embed',
    help='Base directory for created files')

# options for encoder
parser.add_argument(
    '--encoder', type=str, required=True,
    help='Encoder to be used')
parser.add_argument(
    '--bpe_codes', type=str, required=True,
    help='Directory of the tokenized data')
parser.add_argument(
    '--lang', '-L', nargs='+', default=None,
    help="List of languages to test on")
parser.add_argument(
    '--buffer-size', type=int, default=10000,
    help='Buffer size (sentences)')
parser.add_argument(
    '--max-tokens', type=int, default=12000,
    help='Maximum number of tokens to process in a batch')
parser.add_argument(
    '--max-sentences', type=int, default=None,
    help='Maximum number of sentences to process in a batch')
parser.add_argument(
    '--cpu', action='store_true',
    help='Use CPU instead of GPU')
parser.add_argument(
    '--verbose', action='store_true',
    help='Detailed output')
args = parser.parse_args()

print('LASER: calculate embeddings for MLDoc')

if not os.path.exists(args.data_dir):
    os.mkdir(args.data_dir)

enc = EncodeLoad(args)

print('\nProcessing:')
for part in ('train1000', 'dev', 'test'):
    # for lang in "en" if part == 'train1000' else args.lang:
    for lang in args.lang:
        cfname = os.path.join(args.data_dir, 'mldoc.' + part)
        Token(cfname + '.txt.' + lang,
              cfname + '.tok.' + lang,
              lang=lang,
              romanize=(True if lang == 'el' else False),
              lower_case=True, gzip=False,
              verbose=args.verbose, over_write=False)
        SplitLines(cfname + '.tok.' + lang,
                   cfname + '.split.' + lang,
                   cfname + '.sid.' + lang)
        BPEfastApply(cfname + '.split.' + lang,
                     cfname + '.split.bpe.' + lang,
                     args.bpe_codes,
                     verbose=args.verbose, over_write=False)
        EncodeFile(enc,
                   cfname + '.split.bpe.' + lang,
                   cfname + '.split.enc.' + lang,
                   verbose=args.verbose, over_write=False,
                   buffer_size=args.buffer_size)
        JoinEmbed(cfname + '.split.enc.' + lang,
                  cfname + '.sid.' + lang,
                  cfname + '.enc.' + lang)
