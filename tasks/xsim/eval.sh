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
#-------------------------------------------------------
#
# This bash script installs the flores200 dataset, downloads laser2, and then
# performs xsim (multilingual similarity) evaluation with ratio margin

if [ -z ${LASER} ] ; then 
  echo "Please set the environment variable 'LASER'"
  exit
fi

ddir="${LASER}/data"
cd $ddir  # move to data directory

if [ ! -d $ddir/flores200 ] ; then
    echo " - Downloading flores200..."
    wget --trust-server-names -q https://tinyurl.com/flores200dataset
    tar -xf flores200_dataset.tar.gz
    /bin/mv flores200_dataset flores200
    /bin/rm flores200_dataset.tar.gz
else
    echo " - flores200 already downloaded"
fi

mdir="${LASER}/models"
if [ ! -d ${mdir} ] ; then
  echo " - creating model directory: ${mdir}"
  mkdir -p ${mdir}
fi

function download {
    file=$1
    if [ -f ${mdir}/${file} ] ; then
        echo " - ${mdir}/$file already downloaded";
    else
        echo " - Downloading $s3/${file}";
        wget -q $s3/${file};
    fi 
}

cd $mdir  # move to model directory

# available encoders
s3="https://dl.fbaipublicfiles.com/nllb/laser"

if [ ! -f ${mdir}/laser2.pt ] ; then
    echo " - Downloading $s3/laser2.pt"
    wget --trust-server-names -q https://tinyurl.com/nllblaser2
else
    echo " - ${mdir}/laser2.pt already downloaded"
fi
download "laser2.spm"
download "laser2.cvocab"

corpus_part="devtest"
corpus="flores200"

# note: example evaluation script expects format: basedir/corpus/corpus_part/lang.corpus_part

echo " - calculating xsim"
python3 $LASER/source/eval.py                \
    --base-dir $ddir                         \
    --corpus $corpus                         \
    --corpus-part $corpus_part               \
    --margin ratio                           \
    --src-encoder   $LASER/models/laser2.pt  \
    --src-spm-model $LASER/models/laser2.spm \
    --src-langs afr_Latn,fin_Latn,fra_Latn,hin_Deva,tha_Thai,eng_Latn      \
    --nway --verbose
