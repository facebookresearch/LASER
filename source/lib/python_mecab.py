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

import os
import sys
import argparse
import MeCab

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description="Mecab Japanese tokenizer wrapped in python to avoid C++ Mecab issues")
parser.add_argument(
    '--input', '-i', type=argparse.FileType('r', encoding='UTF-8'),
    default=sys.stdin,
    metavar='PATH',
    help="Input text file (default: standard input).")
parser.add_argument(
    '--output', '-o', type=argparse.FileType('w', encoding='UTF-8'),
    default=sys.stdout,
    metavar='PATH',
    help="Output text file (default: standard output).")

args = parser.parse_args()
wakati = MeCab.Tagger("-Owakati")

for line in args.input:
    line = wakati.parse(line)
    args.output.write(line)
