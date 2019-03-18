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
# This bash script installs sentence encoders from Amazon s3
#

if [ -z ${LASER} ] ; then 
  echo "Please set the environment variable 'LASER'"
  exit
fi

mdir="${LASER}/models"

# available encoders
s3="https://dl.fbaipublicfiles.com/laser/models"
networks=("bilstm.eparl21.2018-11-19.pt" \
          "eparl21.fcodes" "eparl21.fvocab" \
          "bilstm.93langs.2018-12-26.pt" \
          "93langs.fcodes" "93langs.fvocab")


echo "Downloading networks"

if [ ! -d ${mdir} ] ; then
  echo " - creating directory ${mdir}"
  mkdir -p ${mdir}
fi

cd ${mdir}
for f in ${networks[@]} ; do
  if [ -f ${f} ] ; then
    echo " - ${mdir}/${f} already downloaded"
  else
    echo " - ${f}"
    wget -q ${s3}/${f}
  fi
done
