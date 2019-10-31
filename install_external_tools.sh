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

bdir="${LASER}"
tools_ext="${bdir}/tools-external"


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
# Tokenization tools from Moses
# It is important to use the official release V4 and not the current one
# to obtain the same results than the published ones.
# (the behavior of the tokenizer for end-of-sentence abbreviations has changed)
#
###################################################################

InstallMosesTools () {
  wdir="${tools_ext}/moses-tokenizer/tokenizer"
  MKDIR ${wdir}
  moses_tokenizer_root_url="https://raw.githubusercontent.com/moses-smt/mosesdecoder/RELEASE-4.0/scripts/tokenizer"

  wget -P ${wdir} -nc -nd -nv \
    "${moses_tokenizer_root_url}/tokenizer.perl" \
    "${moses_tokenizer_root_url}/detokenizer.perl" \
    "${moses_tokenizer_root_url}/normalize-punctuation.perl" \
    "${moses_tokenizer_root_url}/remove-non-printing-char.perl" \
    "${moses_tokenizer_root_url}/deescape-special-chars.perl" \
    "${moses_tokenizer_root_url}/lowercase.perl" \
    "${moses_tokenizer_root_url}/basic-protected-patterns"

  chmod 755 ${wdir}/*perl

  # download non-breaking prefixes per language

  wdir="${tools_ext}/moses-tokenizer/share/nonbreaking_prefixes"
  MKDIR ${wdir}
  moses_nonbreaking_prefixes_root_url="https://raw.githubusercontent.com/moses-smt/mosesdecoder/RELEASE-4.0/scripts/share/nonbreaking_prefixes/nonbreaking_prefix"

  wget -P ${wdir} -nc -nd -nv \
    "${moses_nonbreaking_prefixes_root_url}.ca" \
    "${moses_nonbreaking_prefixes_root_url}.cs" \
    "${moses_nonbreaking_prefixes_root_url}.de" \
    "${moses_nonbreaking_prefixes_root_url}.el" \
    "${moses_nonbreaking_prefixes_root_url}.en" \
    "${moses_nonbreaking_prefixes_root_url}.es" \
    "${moses_nonbreaking_prefixes_root_url}.fi" \
    "${moses_nonbreaking_prefixes_root_url}.fr" \
    "${moses_nonbreaking_prefixes_root_url}.ga" \
    "${moses_nonbreaking_prefixes_root_url}.hu" \
    "${moses_nonbreaking_prefixes_root_url}.is" \
    "${moses_nonbreaking_prefixes_root_url}.it" \
    "${moses_nonbreaking_prefixes_root_url}.lt" \
    "${moses_nonbreaking_prefixes_root_url}.lv" \
    "${moses_nonbreaking_prefixes_root_url}.nl" \
    "${moses_nonbreaking_prefixes_root_url}.pl" \
    "${moses_nonbreaking_prefixes_root_url}.pt" \
    "${moses_nonbreaking_prefixes_root_url}.ro" \
    "${moses_nonbreaking_prefixes_root_url}.ru" \
    "${moses_nonbreaking_prefixes_root_url}.sk" \
    "${moses_nonbreaking_prefixes_root_url}.sl" \
    "${moses_nonbreaking_prefixes_root_url}.sv" \
    "${moses_nonbreaking_prefixes_root_url}.ta" \
    "${moses_nonbreaking_prefixes_root_url}.yue" \
    "${moses_nonbreaking_prefixes_root_url}.zh"

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
    wget https://github.com/glample/fastBPE/archive/1fd33189c126dae356b9e187d93d93302fa45cef.zip
    unzip 1fd33189c126dae356b9e187d93d93302fa45cef.zip
    /bin/rm 1fd33189c126dae356b9e187d93d93302fa45cef.zip
    mv fastBPE-1fd33189c126dae356b9e187d93d93302fa45cef fastBPE
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
# Install Japanese tokenizer Mecab
# We do not use automatic installation with "pip" but directly add the soruce directory
#
###################################################################

InstallMecab () {
  cd ${tools_ext}
  if [ ! -x mecab/mecab/bin/mecab ] ; then
    echo " - download mecab from github"
    wget https://github.com/taku910/mecab/archive/3a07c4eefaffb4e7a0690a7f4e5e0263d3ddb8a3.zip
    unzip 3a07c4eefaffb4e7a0690a7f4e5e0263d3ddb8a3.zip
    #/bin/rm master.zip
    if [ ! -s mecab/bin/mecab ] ; then
      mkdir mecab
      cd mecab-3a07c4eefaffb4e7a0690a7f4e5e0263d3ddb8a3/mecab
      echo " - installing code"
      ./configure --prefix ${tools_ext}/mecab && make && make install 
      if [ $? -q 1 ] ; then
        echo "ERROR: installation failed, please install manually"; exit
      fi
    fi
    if [ ! -d mecab/lib/mecab/dic/ipadic ] ; then
      cd ${tools_ext}/mecab-3a07c4eefaffb4e7a0690a7f4e5e0263d3ddb8a3/mecab-ipadic
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

#InstallMecab
echo ""
echo "automatic installation of the Japanese tokenizer mecab may be tricky"
echo "Please install it manually from https://github.com/taku910/mecab"
echo ""
echo "The installation directory should be ${LASER}/tools-external/mecab"
echo ""
