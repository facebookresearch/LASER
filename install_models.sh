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
#-------------------------------------------------------
#
# This bash script installs sentence encoders from Amazon s3
#

if [ -z ${LASER+x} ] ; then 
  echo "Please set the environment variable 'LASER'"
  exit
fi

bdir="${LASER}"
mdir="${bdir}/models"

# available encoders
net_s3="https://s3.amazonaws.com/laser-toolkit/models/networks"
net_local="${mdir}/networks"
networks=("blstm.ep7.9langs-v1.bpej20k.model.py")

# corresponding BPE codes and binarization vocabularies
bin_s3="https://s3.amazonaws.com/laser-toolkit/models/binarize"
bin_local="${mdir}/binarize"
binarize=("ep7.9langs-v1.bpej20k.bin.9xx" "ep7.9langs-v1.bpej20k.codes.9xx")


#--------------------------------------

MKDIR () {
  df=$1
  if [ ! -d ${df} ] ; then
    echo " - creating directory ${df}"
    mkdir -p ${df}
  fi
}

#--------------------------------------

echo "Downloading vocabularies"
MKDIR ${bin_local}
cd ${bin_local}
for f in ${binarize[@]} ; do
  if [ -f ${f} ] ; then
    echo " - ${bin_local}/${f} already downloaded"
  else
    echo " - ${bin_local}/${f}"
    wget -q ${bin_s3}/${f}
  fi
done

#--------------------------------------

echo "Downloading networks"
MKDIR ${net_local}
cd ${net_local}
for f in ${networks[@]} ; do
  if [ -f ${f} ] ; then
    echo " - ${net_local}/${f} already downloaded"
  else
    echo " - ${net_local}/${f}"
    wget -q ${net_s3}/${f}
  fi
done
