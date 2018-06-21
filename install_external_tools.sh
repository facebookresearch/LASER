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
# This bash script installs third party software 
#

if [ -z ${LASER+x} ] ; then 
  echo "Please set the environment variable 'LASER'"
  exit
fi

bdir="${LASER}"
tools_ext="${bdir}/tools-external"
shdir="${bdir}/tasks/share"

# download some selected file from the Moses release 4 package
moses_git="https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts"
moses_files=("tokenizer/tokenizer.perl" "tokenizer/detokenizer.perl" \
             "tokenizer/normalize-punctuation.perl" \
             "tokenizer/deescape-special-chars.perl" \
             "tokenizer/lowercase.perl" \
             "tokenizer/basic-protected-patterns" \
            )

moses_non_breakings="share/nonbreaking_prefixes/nonbreaking_prefix"
moses_non_breaking_langs=("de" "en" "es" "fi" "fr" "it" "nl" "pt" )


###################################################################
#
# Generic helper functions
#
###################################################################

MKDIR () {
  dname=$1
  if [ ! -d ${dname} ] ; then
    echo " - creating directory ${dname}"
    mkdir -p ${dname}
  fi
}


###################################################################
#
# Tokenization tools_ext form Moses
#
###################################################################

InstallMosesTools () {
  wdir="${tools_ext}/moses-tokenizer"
  MKDIR ${wdir}
  cd ${wdir}

  for f in ${moses_files[@]} ; do
    if [ ! -f `basename ${f}` ] ; then
      echo " - download ${f}"
      wget -q ${moses_git}/${f} .
    fi
  done
  chmod 755 *perl

  # download non-breaking prefixes per language
  wdir="${tools_ext}/share/nonbreaking_prefixes"
  MKDIR ${wdir}
  cd ${wdir}

  for l in ${moses_non_breaking_langs[@]} ; do
    f="${moses_non_breakings}.${l}"
    if [ ! -f `basename ${f}` ] ; then
      echo " - download ${f}"
      wget -q ${moses_git}/${f} .
    fi
  done
}


###################################################################
#
# BPE 
#
###################################################################

InstallBPE () {
  cd ${tools_ext}
  if [ ! -d subword_nmt ] ; then
    echo " - download BPE software for github"
    git clone https://github.com/rsennrich/subword-nmt.git

    # for easy python import
    cd ${LASER}/mlenc
    ln -s ${tools_ext}/subword-nmt/apply_bpe.py
  fi
}




###################################################################
#
# main
#
###################################################################

InstallMosesTools
InstallBPE
