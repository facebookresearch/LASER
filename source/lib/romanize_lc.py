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
# Romanize and lower case text

import os
import sys
import argparse
from transliterate import translit, get_available_language_codes

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description="Calculate multilingual sentence encodings")
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
parser.add_argument(
    '--language', '-l', type=str,
    metavar='STR', default="none",
    help="perform transliteration into Roman characters"
         " from the specified language (default none)")
parser.add_argument(
    '--preserve-case', '-C', action='store_true',
    help="Preserve case of input texts (default is all lower case)")

args = parser.parse_args()

for line in args.input:
    if args.language != "none":
        line = translit(line, args.language, reversed=True)
    if not args.preserve_case:
        line = line.lower()
    args.output.write(line)
