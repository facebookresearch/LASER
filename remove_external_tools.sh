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
# This bash script removes all installed third party software 
#

if [ -z ${LASER+x} ] ; then 
  echo "Please set the environment variable 'LASER'"
  exit
fi

bdir="${LASER}"
tools_ext="${bdir}/tools-external"

/bin/rm -rf ${tools_ext}
