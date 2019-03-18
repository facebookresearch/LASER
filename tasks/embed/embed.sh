#!/bin/bash
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
# bash script to calculate sentence embeddings for arbitrary
# text file

if [ -z ${LASER+x} ] ; then
  echo "Please set the environment variable 'LASER'"
  exit
fi

if [ $# -ne 3 ] ; then
  echo "usage embed.sh input-file language output-file"
  exit
fi

ifile=$1
lang=$2
ofile=$3

# encoder
model_dir="${LASER}/models"
encoder="${model_dir}/bilstm.93langs.2018-12-26.pt"
bpe_codes="${model_dir}/93langs.fcodes"

cat $ifile \
  | python3 ${LASER}/source/embed.py \
    --encoder ${encoder} \
    --token-lang ${lang} \
    --bpe-codes ${bpe_codes} \
    --output ${ofile} \
    --verbose
