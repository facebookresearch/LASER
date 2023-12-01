#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# LASER  Language-Agnostic SEntence Representations
# is a toolkit to calculate multilingual sentence embeddings
# and to use them for various tasks such as document classification,
# and bitext filtering
#
#-------------------------------------------------------
#
# This bash script downloads the flores200 dataset, laser2, and then
# performs pxsim evaluation

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

cd -

mdir="${LASER}/models"
if [ ! -d ${mdir} ] ; then
  echo " - creating model directory: ${mdir}"
  mkdir -p ${mdir}
fi

function download {
    file=$1
    save_dir=$2
    if [ -f ${save_dir}/${file} ] ; then
        echo " - ${save_dir}/$file already downloaded";
    else
        cd $save_dir
        echo " - Downloading $s3/${file}";
        wget -q $s3/${file};
        cd -
    fi
}

# available encoders
s3="https://dl.fbaipublicfiles.com/nllb/laser"

if [ ! -f ${mdir}/laser2.pt ] ; then
    cd $mdir
    echo " - Downloading $s3/laser2.pt"
    wget --trust-server-names -q https://tinyurl.com/nllblaser2
    cd -
else
    echo " - ${mdir}/laser2.pt already downloaded"
fi
download "laser2.spm" $mdir
download "laser2.cvocab" $mdir

# encode FLORES200 texts using both LASER2 and LaBSE
for lang in eng_Latn wol_Latn; do
    infile=$LASER/data/flores200/devtest/$lang.devtest
    python3 ${LASER}/source/embed.py            \
        --input     $infile                     \
        --encoder   $mdir/laser2.pt             \
        --spm-model $mdir/laser2.spm            \
        --output    $lang.devtest.laser2        \
        --verbose

    python3 ${LASER}/source/embed.py            \
        --input     $infile                     \
        --encoder   LaBSE                       \
        --use-hugging-face                      \
        --output    $lang.devtest.labse         \
        --verbose
done

# run pxsim using LaBSE as an auxiliary scoring model
echo " - calculating p-xsim"
python3 $LASER/source/pxsim.py run              \
    --src_emb wol_Latn.devtest.laser2           \
    --tgt_emb eng_Latn.devtest.laser2           \
    --src_aux_emb wol_Latn.devtest.labse        \
    --tgt_aux_emb eng_Latn.devtest.labse        \
    --alpha 0.1                                 \
    --k 32                                      \
    --aux_emb_dim 768
