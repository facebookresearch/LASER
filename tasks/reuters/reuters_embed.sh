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
#--------------------------------------------------------
#
# bash script to calculate sentence embeddings for the Reuters RCV2 corpus


# general config
data="."
ddir=${data}/data	# raw texts of Reuters RCV2
edir=${data}/embed	# normalized texts and embeddings
langs=("de" "en" "fr" "es")

# parameters
lbl="lbl"
txt="txt"
maxlen=2800	# maxmimum number of words after sentence splitting and BPE
mod="eparl20kj"	# which BLSTM model to use
gpu=0		# which GPU to use, use -1 for CPU mode

# include common functions
source ../share/tools.sh

###################################################################
#
# Extract files with labels and texts from the Reuters corpus
#
###################################################################

ExtractReuters () {
  fname=$1
  lang=$2
  if [ ! -f ${edir}/${fname}.${lbl}.${lang} ] ; then
    echo " - extract labels in ${fname}.${lang}"
    cut -d'	' -f1 ${ddir}/${fname}.${lang} \
      | sed -e 's/C/1/' -e 's/E/2/' -e 's/G/3/' -e 's/M/4/' \
      > ${edir}/${fname}.${lbl}.${lang}
  fi
  if [ ! -f ${edir}/${fname}.${txt}.${lang} ] ; then
    echo " - extract texts in ${fname}.${lang}"
    # remove text which is not useful for classification
    cut -d'	' -f2 ${ddir}/${fname}.${lang} \
     | sed -e 's/ Co \./ Co./g' -e s'/ Inc \. / Inc. /g' \
           -e 's/([cC]) Reuters Limited 199[0-9]\.//g' \
      > ${edir}/${fname}.${txt}.${lang}
  fi
}


###################################################################
#
# Create all files
#
###################################################################

# create output directories
for d in ${edir} ; do
  MKDIR ${d}
done

echo -e "\nProcessing Reuters RCV2 data in ${data}"
for l in ${langs[@]} ; do
  for part in "reuters.train1000" "reuters.test" ; do
    ExtractReuters ${part} ${l}
    Tokenize ${edir}/${part}.${txt} ${l}
    SplitLines ${edir}/${part}.${txt}.${tok} ${l}	# creates files .split.LANG and .sid.LANG
    Embed ${edir}/${part}.${txt}.${tok}.split ${l} ${mod} ${maxlen} ${gpu}
    JoinLines ${edir}/${part}.${txt}.${tok}.%s.${lang}
  done
done
