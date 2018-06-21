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
# bash script to calculate sentence embeddings for the BUCC corpus


# general config
bucc="bucc2018"
data="."
xdir=${data}/downloaded	# tar files as distrubuted by the BUCC evaluation
ddir=${data}/${bucc}	# raw texts of BUCC
edir=${data}/embed	# normalized texts and embeddings
langs=("fr" "de" )
lang0="en"		# English is always the 2nd language

# parameters
id="id"
txt="txt"
maxlen=2500		# maxmimum number of words after sentence splitting and BPE
mod="eparl20kj"		# which BLSTM model to use
gpu=0			# which GPU to use, use -1 for CPU mode

# include common functions
source ../share/tools.sh


###################################################################
#
# Extract files with labels and texts from the BUCC corpus
#
###################################################################

GetData () {
  fn1=$1; fn2=$2; lang=$3
  outf="${edir}/${bucc}.${lang}-${lang0}.${fn2}"
  for ll  in ${lang0} ${lang} ; do
    inf="${ddir}/${fn1}.${ll}"
    if [ ! -f ${outf}.${txt}.${ll} ] ; then
      echo " - extract files ${outf} in ${ll}"
      cat ${inf} | cut -f1 > ${outf}.${id}.${ll}
      cat ${inf} | cut -f2 > ${outf}.${txt}.${ll}
    fi
  done
}

ExtractBUCC () {
  slang=$1
  tlang=${lang0}

  pushd ${data} > /dev/null
  if [ ! -d ${ddir}/${slang}-${tlang} ] ; then
    for tf in ${xdir}/${bucc}-${slang}-${tlang}.*.tar.bz2 ; do
      echo " - extract from tar `basename ${tf}`"
      tar jxf $tf
    done
  fi

  GetData "${slang}-${tlang}/${slang}-${tlang}.sample" "dev" ${l}
  GetData "${slang}-${tlang}/${slang}-${tlang}.training" "train" ${l}
  GetData "${slang}-${tlang}/${slang}-${tlang}.test" "test" ${l}
  popd > /dev/null
}


###################################################################
#
# Main loop
#
###################################################################

echo -e "\nProcessing BUCC data in ${data}"

# create output directories
for d in ${ddir} ${edir} ; do
  MKDIR ${d}
done

for l in ${langs[@]} ; do
  ExtractBUCC ${l}
  bname="${bucc}.${l}-${lang0}"

  # Tokenzie and embed train
  # (we don't use the very dev data)
  for part in "${bname}.train" "${bname}.test"; do
    for ll  in ${lang0} ${lang} ; do
      Tokenize ${edir}/${part}.${txt} ${ll}
      Embed ${edir}/${part}.${txt}.${tok} ${ll} ${mod} ${maxlen} ${gpu}
    done
  done
done
