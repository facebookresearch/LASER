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
  exit 1
fi

if [ $# -lt 2 ] ; then
  echo "usage: embed.sh input-file output-file [language (iso3)]"
  exit 1
fi

infile=$1
outfile=$2
language=${3:-"eng"}

version=1

# defaulting to WMT'22 models (see tasks/wmt22)
model_dir="$LASER/models/wmt22"

model_file=${model_dir}/laser3-$language.v$version.pt

if [ -s $model_file ]; then
    encoder=$model_file
else
    echo "couldn't find $model_file. defaulting to laser2"
    encoder="${model_dir}/laser2.pt"
fi

python3 ${LASER}/source/embed.py \
    --input ${infile}    \
    --encoder ${encoder} \
    --output ${outfile}  \
    --verbose
