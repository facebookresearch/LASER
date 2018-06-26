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
# bash script to calculate sentence embeddings for the MLDoc corpus


# general config
data="."
ddir=${data}/MLDoc	# raw texts of MLdoc
edir=${data}/embed	# normalized texts and embeddings
langs=("de" "en" "fr" "es" "it" )

# parameters
lbl="lbl"
txt="txt"
maxlen=2500		# maxmimum number of words after sentence splitting and BPE
mod="eparl20kj"		# which BLSTM model to use
gpu=0			# which GPU to use, use -1 for CPU mode

# include common functions
source ../share/tools.sh

###################################################################
#
# Extract files with labels and texts from the MLdoc corpus
#
###################################################################

ExtractMLdoc () {
  fname=$1
  lang=$2
  if [ ! -d ${ddir} ] ; then
    echo "Please install the MLDoc corpus first"
    exit
  fi

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

echo -e "\nProcessing MLDoc data"
for l in ${langs[@]} ; do
  for part in "mldoc.train1000" "mldoc.dev" "mldoc.test" ; do
    ExtractMLdoc ${part} ${l}
    Tokenize ${edir}/${part}.${txt} ${l}
    # create files .split.LANG and .sid.LANG
    SplitLines ${edir}/${part}.${txt}.${tok} ${l}
    Embed ${edir}/${part}.${txt}.${tok}.split ${l} ${mod} ${maxlen} ${gpu}
    JoinLines ${edir}/${part}.${txt}.${tok}.%s.${lang}
  done
done
