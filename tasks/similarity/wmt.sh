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
# evaluate similarity search on WMT newstest2011

if [ -z ${LASER+x} ] ; then
  echo "Please set the environment variable 'LASER'"
  exit
fi

# encoder
model_dir="${LASER}/models"
encoder="${model_dir}/bilstm.93langs.2018-12-26.pt"
bpe_codes="${model_dir}/93langs.fcodes"

edir="embed"


if [ ! -d dev ] ; then
  echo " - Download WMT data"
  wget -q http://www.statmt.org/wmt13/dev.tgz
  tar --wildcards -xf dev.tgz "dev/newstest2012.??"
  /bin/rm dev.tgz
fi

python3 ${LASER}//source/similarity_search.py \
    --bpe-codes ${bpe_codes} --encoder ${encoder} \
    --base-dir . \
    --data dev/newstest2012 --output ${edir}/newstest2012 \
    --lang cs de en es fr --verbose
