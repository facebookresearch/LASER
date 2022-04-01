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
# This bash script installs WMT'22 sentence encoders from Amazon s3

if [ -z ${LASER} ] ; then 
  echo "Please set the environment variable 'LASER'"
  exit
fi

mdir="${LASER}/models/wmt22"
version=1  # model version

echo "Downloading networks..."

if [ ! -d ${mdir} ] ; then
  echo " - creating model directory: ${mdir}"
  mkdir -p ${mdir}
fi

function download {
    file=$1
    if [ -f ${mdir}/${file} ] ; then
        echo " - ${mdir}/$file already downloaded";
    else
        echo " - $s3/${file}";
        wget -q $s3/${file};
    fi   
}

cd ${mdir}  # move to model directory

# available encoders
s3="https://dl.fbaipublicfiles.com/laser/models"

# [afr, eng, and fra] are supported by the same LASER2 model (93 langs total)
download "laser2.pt"
download "laser2.spm"
download "laser2.cvocab"

# other WMT '22 supported languages (-afr,eng,fra)
langs=(amh fuv hau ibo kam \
       kin lin lug luo nso \
       nya orm sna som ssw \
       swh tsn tso umb wol \
       xho yor zul)

for lang in ${langs[@]}; do
    download "laser3-$lang.v$version.pt";
    if [ $lang == "fuv" ] || [ $lang == "amh" ] ; then
        download "laser3-$lang.v$version.spm";
        download "laser3-$lang.v$version.cvocab";
    fi 
done