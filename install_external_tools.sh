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
# This bash script installs third party software 
#

if [ -z ${LASER} ] ; then 
  echo "Please set the environment variable 'LASER'"
  exit
fi

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


bdir="${LASER}"
tools_ext="${bdir}/tools-external"
MKDIR $tools_ext

###################################################################
#
# Tokenization tools from Moses
# It is important to use the official release V4 and not the current one
# to obtain the same results than the published ones.
# (the behavior of the tokenizer for end-of-sentence abbreviations has changed)
#
###################################################################

InstallMosesTools () {
  moses_git="https://raw.githubusercontent.com/moses-smt/mosesdecoder/RELEASE-4.0/scripts"
  moses_files=("tokenizer/tokenizer.perl" "tokenizer/detokenizer.perl" \
               "tokenizer/normalize-punctuation.perl" \
               "tokenizer/remove-non-printing-char.perl" \
               "tokenizer/deescape-special-chars.perl" \
               "tokenizer/lowercase.perl" \
               "tokenizer/basic-protected-patterns" \
              )

  wdir="${tools_ext}/moses-tokenizer/tokenizer"
  MKDIR ${wdir}
  cd ${wdir}

  for f in ${moses_files[@]} ; do
    if [ ! -f `basename ${f}` ] ; then
      echo " - download ${f}"
      wget -q ${moses_git}/${f}
    fi
  done
  chmod 755 *perl

  # download non-breaking prefixes per language
  moses_non_breakings="share/nonbreaking_prefixes/nonbreaking_prefix"
  moses_non_breaking_langs=( \
      "ca" "cs" "de" "el" "en" "es" "fi" "fr" "ga" "hu" "is" \
      "it" "lt" "lv" "nl" "pl" "pt" "ro" "ru" "sk" "sl" "sv" \
      "ta" "yue" "zh" )
  wdir="${tools_ext}/moses-tokenizer/share/nonbreaking_prefixes"
  MKDIR ${wdir}
  cd ${wdir}

  for l in ${moses_non_breaking_langs[@]} ; do
    f="${moses_non_breakings}.${l}"
    if [ ! -f `basename ${f}` ] ; then
      echo " - download ${f}"
      wget -q ${moses_git}/${f} 
    fi
  done
}


###################################################################
#
# FAST BPE 
#
###################################################################

InstallFastBPE () {
  cd ${tools_ext}
  if [ ! -x fastBPE/fast ] ; then
    echo " - download fastBPE software from github"
    wget https://github.com/glample/fastBPE/archive/master.zip
    unzip master.zip
    /bin/rm master.zip
    mv fastBPE-master fastBPE
    cd fastBPE
    echo " - compiling"
    g++ -std=c++11 -pthread -O3 fastBPE/main.cc -IfastBPE -o fast
    if [ $? -eq 1 ] ; then
      echo "ERROR: compilation failed, please install manually"; exit
    fi
    python setup.py install
  fi
}

###################################################################
#
# SENTENCEPIECE 
#
###################################################################

InstallSentencePiece () {
  cd ${tools_ext}
  if [ ! -d sentencepiece-master ] ; then
    echo " - download sentencepiece from github"
    wget https://github.com/google/sentencepiece/archive/master.zip
    unzip master.zip
    /bin/rm master.zip
    if [ ! -s /usr/local/bin/spm_encode ] ; then
      echo " - building code "
      cd sentencepiece-master
      mkdir build
      cd build
      cmake ..
      make -j 10
    fi
  fi
}


###################################################################
#
# Install Japanese tokenizer Mecab
# We do not use automatic installation with "pip" but directly add the soruce directory
#
###################################################################

InstallMecab () {
  cd ${tools_ext}
  if [ ! -x mecab/mecab/bin/mecab ] ; then
    echo " - download mecab from github"
    wget https://github.com/taku910/mecab/archive/master.zip
    unzip master.zip 
    #/bin/rm master.zip
    if [ ! -s mecab/bin/mecab ] ; then
      mkdir mecab
      cd mecab-master/mecab
      echo " - installing code"
      ./configure --prefix ${tools_ext}/mecab && make && make install 
      if [ $? -q 1 ] ; then
        echo "ERROR: installation failed, please install manually"; exit
      fi
    fi
    if [ ! -d mecab/lib/mecab/dic/ipadic ] ; then
      cd ${tools_ext}/mecab-master/mecab-ipadic
      echo " - installing dictionaries"
      ./configure --prefix ${tools_ext}/mecab --with-mecab-config=${tools_ext}/mecab/bin/mecab-config \
        && make && make install 
      if [ $? -eq 1 ] ; then
        echo "ERROR: compilation failed, please install manually"; exit
      fi
    fi
  fi
}


###################################################################
#
# main
#
###################################################################

echo "Installing external tools"

InstallMosesTools
InstallFastBPE
InstallSentencePiece

#InstallMecab
echo ""
echo "automatic installation of the Japanese tokenizer mecab may be tricky"
echo "Please install it manually from https://github.com/taku910/mecab"
echo ""
echo "The installation directory should be ${LASER}/tools-external/mecab"
echo ""
