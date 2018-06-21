#!/bin/bash
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
# bash script to calculate pairwise similarity errors  on a corpus
# of mutual translations.
# Here, we use the WMT newstest of 2009


# include common functions
source ../share/tools.sh

# where to find LASER tools
mldir="${LASER}/tools"

# embedding parameters
bname="dev/newstest2009"; langs=("de" "en" "es" "fr")
maxlen=2500		# maxmimum number of words after sentence splitting and BPE
mod="eparl20kj"		# which BLSTM model to use
gpu=0			# which GPU to use, use -1 for CPU mode


echo "Calculating similarity errors"
if [ ! -f dev.tgz ] ; then
  echo " - downloading WMT'18 dev sets"
  wget -q http://data.statmt.org/wmt18/translation-task/dev.tgz
fi

if [ ! -d dev ] ; then
  echo " - extracting data"
  tar zxf dev.tgz
fi

echo "Processing data:"
for lang in ${langs[@]} ; do
  Tokenize ${bname} ${lang}
  Embed ${bname}.${tok} ${lang} ${mod} ${maxlen} ${gpu}
done

python ${mldir}/faiss/similarity_error.py \
  --fname ${bname}.${tok}.enc --norm \
  --langs ${langs[0]} --langs ${langs[1]} --langs ${langs[2]} --langs ${langs[3]}

