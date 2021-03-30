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
echo "Downloading networks to ${mdir}"
mkdir -p ${mdir}
wget -P ${mdir} -nc -nd \
  "https://dl.fbaipublicfiles.com/laser/models/bilstm.eparl21.2018-11-19.pt" \
  "https://dl.fbaipublicfiles.com/laser/models/eparl21.fcodes" \
  "https://dl.fbaipublicfiles.com/laser/models/eparl21.fvocab" \
  "https://dl.fbaipublicfiles.com/laser/models/bilstm.93langs.2018-12-26.pt" \
  "https://dl.fbaipublicfiles.com/laser/models/93langs.fcodes" \
  "https://dl.fbaipublicfiles.com/laser/models/93langs.fvocab"
